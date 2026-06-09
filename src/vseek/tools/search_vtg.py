"""Video search tool backed by UniversalVTG temporal grounding.

Instead of cosine-similarity retrieval, this tool calls the VTG server which
runs the full temporal grounding model to localize relevant moments in the
video, then returns frames from those localized segments.
"""

import json
import logging
import os
from typing import Any, Optional, List, Dict, Tuple
from uuid import uuid4
from pathlib import Path

import cv2
import base64
import numpy as np
import requests
from verl.tools.base_tool import BaseTool
from verl.utils.rollout_trace import rollout_trace_op
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from vseek.data.frame import VideoFrames

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class VTGSearchTool(BaseTool):
    """Video search tool using UniversalVTG temporal grounding.

    This tool communicates with a VTG server that runs the full temporal
    grounding model. Given a text query, the model localizes relevant
    temporal segments in the video and returns the corresponding frames.

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the video search tool
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        self.server_url = config.get("server_url", "http://localhost:9002")
        self.timeout = config.get("timeout", 60)
        self.topk = config.get("topk", 4)
        self.dataset_name = config.get("dataset_name", "lvb")
        self.window_size = config.get("window_size", 8)
        self.index_path = config.get("index_path", "data/index")
        self.cache_limit = config.get("cache_limit", 64)
        self.max_frames_per_turn = config.get("max_frames_per_turn", 16)

        dir_name = f"{self.dataset_name}_window_{self.window_size}"
        self.data_root = Path(self.index_path).joinpath(dir_name)

        self.cached_frames_dict = {}
        self.cached_ids = []

        logger.info(f"Initialized VTGSearchTool with server_url={self.server_url}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "reward": [],
            "search_results": [],
        }
        return instance_id, ToolResponse()

    async def _temporal_ground(
        self,
        query: str,
        topk: int,
        video_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
        precomputed_frames: Optional[List[Dict[str, List[str]]]] = None,
    ) -> Tuple[List[Any], List[int], Dict[str, Any]]:
        """Run temporal grounding via the VTG server.

        Args:
            query: Text query describing what to search for
            topk: Number of top window results to return
            video_id: Video identifier
            dataset_name: Dataset name
            precomputed_frames: Optional pre-encoded frames by window

        Returns:
            Tuple of (frames, window_indices, metadata)
        """
        try:
            # Load video frames for frame extraction
            if precomputed_frames is None:
                if video_id not in self.cached_frames_dict:
                    self.cached_frames_dict[video_id] = VideoFrames.load(
                        str(self.data_root.joinpath(video_id))
                    )
                    self.cached_ids.append(video_id)
                    if len(self.cached_ids) > self.cache_limit:
                        pop_id = self.cached_ids.pop(0)
                        del self.cached_frames_dict[pop_id]

            payload = {
                "query": query,
                "topk": topk,
                "video_id": video_id,
                "dataset_name": dataset_name,
            }

            resp = requests.get(
                f"{self.server_url}/ground",
                params=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("error"):
                return [], [], {"status": "error", "error": data["error"], "search_mode": "temporal_grounding"}

            window_indices = sorted(data.get("window_indices", []))
            segments = data.get("segments", [])
            scores = data.get("scores", [])

            # Extract frames from the localized windows
            frames = []
            if precomputed_frames is not None:
                remapped = {}
                for frame in precomputed_frames:
                    remapped[frame["window_idx"]] = frame["encoded_frames"]
                for wi in window_indices:
                    frames += remapped.get(wi, [])
            else:
                vid_frames = self.cached_frames_dict.get(video_id)
                if vid_frames:
                    for wi in window_indices:
                        chunk = vid_frames.get_frame_chunk(wi)
                        if chunk:
                            frames += chunk

            metadata = {
                "search_mode": "temporal_grounding",
                "status": "success",
                "segments": segments,
                "scores": scores,
                "duration": data.get("duration"),
            }

            # Uniformly sample if too many frames
            if len(frames) > self.max_frames_per_turn:
                idxs = np.linspace(0, len(frames) - 1, self.max_frames_per_turn, dtype=int)
                frames = [frames[i] for i in idxs]

            return frames, window_indices, metadata

        except Exception as e:
            logger.exception("_temporal_ground failed: %s", e)
            return [], [], {"status": "error", "error": str(e), "search_mode": "temporal_grounding"}

    async def _encode_frame(self, frame, max_width=256, max_height=256, quality=85):
        height, width = frame.shape[:2]
        scale = min(max_width / width, max_height / height)
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        ret, buffer = cv2.imencode(".jpg", frame, encode_params)
        if not ret:
            raise ValueError("Could not encode frame")
        return base64.b64encode(buffer).decode("utf-8")

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute temporal grounding search.

        Args:
            instance_id: The instance ID of the tool
            parameters: Tool parameters containing query

        Returns: tool_response, tool_reward_score, tool_metrics
        """
        query = parameters.get("query")

        if query is None or not isinstance(query, str) or not query:
            error_msg = "Error: 'query' must be a non-empty string."
            logger.error(f"[VTGSearchTool] {error_msg} Received parameters: {parameters}")
            return ToolResponse(text=json.dumps({"error": error_msg})), 0.0, {}

        video_id = kwargs.get("video_id")
        precomputed_frames = kwargs.get("precomputed_frames")
        dataset_name = kwargs.get("dataset")

        try:
            frames, window_indices, metadata = await self._temporal_ground(
                query=query,
                topk=self.topk,
                video_id=video_id,
                dataset_name=dataset_name,
                precomputed_frames=precomputed_frames,
            )

            if metadata.get("error"):
                return ToolResponse(text=json.dumps({"error": metadata["error"]})), 0.0, {"error": metadata["error"]}

            self._instance_dict[instance_id]["search_results"].append({
                "query": query,
                "window_indices": window_indices,
                "metadata": metadata,
            })

            metrics = {
                "mode": "temporal_grounding",
                "num_results": len(window_indices),
                "status": metadata.get("status", "success"),
                "num_segments": len(metadata.get("segments", [])),
            }

            if precomputed_frames is None:
                frames = [await self._encode_frame(frame) for frame in frames]
            else:
                frames = [base64.b64encode(frame).decode("utf-8") if isinstance(frame, bytes) else frame for frame in frames]

            return ToolResponse(image=frames), 0.0, metrics

        except Exception as e:
            error_result = json.dumps({"error": f"VTG search failed: {e}"})
            logger.error(f"[VTGSearchTool] Execution failed: {e}")
            return ToolResponse(text=error_result), 0.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
