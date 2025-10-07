# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
import asyncio
import aiohttp
import requests
import torch
from typing import Any, Optional, List, Dict, Tuple
from uuid import uuid4
from pathlib import Path
from omegaconf import DictConfig
from verl.tools.base_tool import BaseTool
from verl.utils.rollout_trace import rollout_trace_op
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from vseek.data.frame import VideoFrames
from data.lvb import LongVideoBench

from vseek.video_embedding.video_clip import ViClip
from vseek.setting import ViClipSetting

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

VICLIP_SETTING = ViClipSetting()


class VideoSearchTool(BaseTool):
    """Video search tool for retrieving relevant video frames based on text queries.
    
    This tool provides video search functionality by communicating with a background
    server that handles video frame retrieval. It uses ViClip embeddings for semantic
    similarity search and returns relevant frame indices.
    
    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the video search tool
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize VideoSearchTool with configuration and schema.

        Args:
            config: Configuration dictionary containing tool settings
            tool_schema: OpenAI function tool schema definition

        Example tool_schema:
            type: function
            function:
                name: video_search
                description: Searches video frames by natural language or subtitle string.
                parameters:
                type: object
                properties:
                    query:
                    type: string
                    description: Text to search. Used for both modes.
                    mode:
                    type: string
                    enum: [base, subtitle]
                    description: base for natural language search, subtitle to match subtitles.
                required: [query, mode]
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Background server configuration
        self.server_url = config.get("server_url", "http://localhost:9000")
        self.timeout = config.get("timeout", 30)
        self.topk = config.get("topk", 4)
        self.dataset_name = config.get("dataset_name", "lvb")
        self.window_size = config.get("window_size", 4)
        self.index_path = config.get("index_path", "data/index")
        self.cache_limit = config.get("cache_limit", 64)
        # Initialize all the frames as per the dataset
        
        dir_name = f"{self.dataset_name}_window_{self.window_size}"
        print(f"dir_name: {dir_name}")
        self.data_root = Path(self.index_path).joinpath(dir_name)
        
        if self.dataset_name == "lvb":
            # convert to dict to DictConfig
            dataset_config = DictConfig(config)
            dataset = LongVideoBench(dataset_config)
            self.entries = dataset.load_data()
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        
        logger.info(f"Loading video index from: {self.data_root}")
        self.cached_frames_dict = {
            # entry["metadata"]["video_id"]: VideoFrames.load(str(data_root.joinpath(entry["metadata"]["video_id"])))
            # for entry in self.entries
        }
        self.cached_ids = []
        
        # ViClip model for text embeddings
        #print(f"Initialized VideoSearchTool with config: {config}")
        logger.info(f"Initialized VideoSearchTool with config: {config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the OpenAI tool schema.
        Ensures the schema requires a query and a mode ("base" or "subtitle").
        """       
        return self.tool_schema


    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a tool instance.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The instance id of the tool.
            tool_creation_response: The response of the tool when creating the instance.
        """
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "reward": [],
            "search_results": [],
        }
        return instance_id, ToolResponse()

    async def _search_video_frames(self, query: str, topk: int = 5, video_id: Optional[str] = None, search_type: str = "video_frames") -> Tuple[List[Any], List[int], Dict[str, Any]]:
        """Search for relevant video frames using background server.

        Args:
            query: Text query describing what to search for
            topk: Number of top results to return

        Returns:
            Tuple of (frame_indices, metadata)
        """
        try:
            if video_id not in self.cached_frames_dict:
                self.cached_frames_dict[video_id] = VideoFrames.load(str(self.data_root.joinpath(video_id)))
                self.cached_ids.append(video_id)
                if len(self.cached_ids) > self.cache_limit:
                    pop_id = self.cached_ids.pop(0)
                    del self.cached_frames_dict[pop_id]
            # Prepare request payload
            payload = {
                    "query": query,
                    "topk": topk,
                    "search_type": search_type,
                    "video_id": video_id
            }
            print(f"payload: {payload}")
            # Make async HTTP request to background server
            if search_type == "video_frames":
                resp = requests.get(f"{self.server_url}/search", params=payload, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                frame_indices = sorted(data.get("frame_indices", []))
            else:
                resp = requests.get(f"{self.server_url}/search_subtitle", params=payload, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                frame_indices = sorted(data.get("subtitle_indices", []))

            vid = data.get("video_id") or video_id
            frames = [self.cached_frames_dict[vid].get_frame_chunk(i) for i in frame_indices] if vid in self.cached_frames_dict else []
            metadata = data.get("metadata", {})
            metadata.setdefault("search_mode", "subtitle" if search_type != "video_frames" else "language")
            return frames, frame_indices, metadata

        except Exception as e:
            logger.exception("_search_video_frames failed: %s", e)
            return [], [], {"status": "error", "error": str(e), "search_mode": search_type}

    async def _search_subtitles(self, subtitle: str, topk: int = 5, video_id: Optional[str] = None, search_type: str = "subtitles") -> Tuple[List[Any], List[int], Dict[str, Any]]:
        """Search for frame indices by subtitle text."""
        return await self._search_video_frames(subtitle, topk=topk, video_id=video_id, search_type=search_type)

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute the video search tool.

        Args:
            instance_id: The instance ID of the tool
            parameters: Tool parameters containing query and optional topk

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The response containing frame indices and metadata
            tool_reward_score: The step reward score of the tool
            tool_metrics: The metrics of the tool
        """
        # Inputs
        query = parameters.get("query")
        mode = parameters.get("mode") or parameters.get("type") or parameters.get("search_type")

        # Backward compatibility: if subtitle provided and query missing, use it as query with subtitle mode
        if query is None and subtitle_legacy is not None:
            query = subtitle_legacy
            if not mode:
                mode = "subtitle"

        if not isinstance(query, str) or not query:
            error_msg = "Error: 'query' must be a non-empty string."
            logger.error(f"[VideoSearchTool] {error_msg} Received parameters: {parameters}")
            return ToolResponse(text=json.dumps({"error": error_msg})), 0.0, {}

        if not isinstance(mode, str) or mode not in {"base", "subtitle"}:
            error_msg = "Error: 'mode' must be 'base' or 'subtitle'."
            logger.error(f"[VideoSearchTool] {error_msg} Received parameters: {parameters}")
            return ToolResponse(text=json.dumps({"error": error_msg})), 0.0, {}

        search_type = "video_frames" if mode == "base" else "subtitles"
        # Maybe needs to be passed through kwargs
        topk = kwargs.get("topk")
        video_id = kwargs.get("video_id")
        
        print(f"query: {query}, mode: {mode}, topk: {topk}, video_id: {video_id}")

        # type checks done above

        try:
            
            if search_type == "video_frames":
                frames, frame_indices, metadata = await self._search_video_frames(
                    query=query, 
                    topk=topk,
                    video_id=video_id,
                    search_type=search_type)
            else:
                frames, frame_indices, metadata = await self._search_subtitles(
                    subtitle=query, 
                    topk=topk, 
                    video_id=video_id, 
                    search_type=search_type)
            # Store results in instance dictionary
            self._instance_dict[instance_id]["search_results"].append({
                "query": query,
                "frame_indices": frame_indices,
                "metadata": metadata
            })

            # Prepare response
            response_data = {
                "mode": "base" if search_type == "video_frames" else "subtitle",
                "query": query,
                "frame_indices": frame_indices,
                "topk": topk,
                "metadata": metadata,
            }

            # Convert metadata to metrics
            metrics = {
                "mode": response_data["mode"],
                "num_results": len(frame_indices),
                "status": metadata.get("status", "success"),
                "error": metadata.get("error"),
            }

            return ToolResponse(image=frames, text=json.dumps(response_data)), 0.0, metrics

        except Exception as e:
            error_result = json.dumps({"error": f"Video search execution failed: {e}"})
            logger.error(f"[VideoSearchTool] Execution failed: {e}")
            return ToolResponse(text=error_result), 0.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        """Calculate reward based on search results quality."""
        return 0.0
        if instance_id in self._instance_dict:
            search_results = self._instance_dict[instance_id]["search_results"]
            if search_results:
                # Simple reward based on number of results found
                last_result = search_results[-1]
                num_results = len(last_result["frame_indices"])
                return f"Found {num_results} relevant frames for query: {last_result['query']}"
        return "No search results available"

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance."""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
