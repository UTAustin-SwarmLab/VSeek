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
import torch
from typing import Any, Optional, List, Dict
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.utils.rollout_trace import rollout_trace_op
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

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
            {
                "type": "function",
                "function": {
                    "name": "video_search",
                    "description": "Searches for relevant video frames based on text queries.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Text query describing what to search for in the video"
                            },
                            "topk": {
                                "type": "integer",
                                "description": "Number of top results to return (default: 5)"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Background server configuration
        self.server_url = config.get("server_url", "http://localhost:8000")
        self.timeout = config.get("timeout", 30)
        self.topk = config.get("topk", 5)
        
        # ViClip model for text embeddings
        self.viclip = ViClip(
            pretrained_model_path=VICLIP_SETTING.vclip_model_path,
            gpu_number=VICLIP_SETTING.gpu_number,
        )

        logger.info(f"Initialized VideoSearchTool with config: {config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the OpenAI tool schema."""
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

    async def _search_video_frames(self, query: str, topk: int = 5) -> tuple[List[int], Dict[str, Any]]:
        """Search for relevant video frames using background server.

        Args:
            query: Text query describing what to search for
            topk: Number of top results to return

        Returns:
            Tuple of (frame_indices, metadata)
        """
        try:
            # Prepare request payload
            payload = {
                "query": query,
                "topk": topk,
                "search_type": "video_frames"
            }

            # Make async HTTP request to background server
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(
                    f"{self.server_url}/search",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        frame_indices = result.get("frame_indices", [])
                        metadata = result.get("metadata", {})
                        return frame_indices, metadata
                    else:
                        error_msg = f"Server returned status {response.status}: {await response.text()}"
                        logger.error(f"Video search failed: {error_msg}")
                        return [], {"error": error_msg, "status": "error"}

        except asyncio.TimeoutError:
            error_msg = f"Request timeout after {self.timeout} seconds"
            logger.error(f"Video search timeout: {error_msg}")
            return [], {"error": error_msg, "status": "timeout"}
        except aiohttp.ClientError as e:
            error_msg = f"Client error: {str(e)}"
            logger.error(f"Video search client error: {error_msg}")
            return [], {"error": error_msg, "status": "client_error"}
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(f"Video search unexpected error: {error_msg}")
            return [], {"error": error_msg, "status": "error"}

    async def _search_with_embeddings(self, query: str, embeddings: List[torch.Tensor], topk: int = 5) -> List[int]:
        """Search for relevant video frames using ViClip embeddings.

        Args:
            query: Text query describing what to search for
            embeddings: List of visual embeddings (tensors)
            topk: Number of top results to return

        Returns:
            List of frame indices sorted by similarity (highest first)
        """
        try:
            # Get text embedding for the search query
            text_embedding = self.viclip.get_text_embedding(query)

            # Get the device of the text embedding (likely GPU)
            device = text_embedding.device

            # Ensure text embedding is 1D [embedding_dim]
            if text_embedding.dim() > 1:
                text_embedding = text_embedding.squeeze()

            # Stack all visual embeddings into a single tensor
            # Move to same device as text embedding and normalize
            visual_embeddings = []
            for emb in embeddings:
                emb_device = emb.to(device)  # Move to same device as text embedding
                # Ensure embedding is 1D
                if emb_device.dim() > 1:
                    emb_device = emb_device.squeeze()
                emb_norm = emb_device / emb_device.norm(dim=-1, keepdim=True)
                visual_embeddings.append(emb_norm)

            visual_embeddings_tensor = torch.stack(visual_embeddings)  # Shape: [N, embedding_dim]

            # Compute cosine similarities on GPU
            # text_embedding is already normalized from get_text_embedding
            # visual_embeddings_tensor: [N, D], text_embedding: [D] -> similarities: [N]
            similarities = torch.matmul(visual_embeddings_tensor, text_embedding)

            # Get indices sorted by similarity (descending order)
            sorted_indices = torch.argsort(similarities, descending=True)

            return sorted_indices.cpu().tolist()[:topk]

        except Exception as e:
            logger.error(f"Embedding search failed: {str(e)}")
            return []

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
        query = parameters.get("query")
        topk = parameters.get("topk", self.topk)

        if not query or not isinstance(query, str):
            error_msg = "Error: 'query' is missing or not a string in parameters."
            logger.error(f"[VideoSearchTool] {error_msg} Received parameters: {parameters}")
            return ToolResponse(text=json.dumps({"error": error_msg})), 0.0, {}

        try:
            # Check if embeddings are provided for local search
            embeddings = kwargs.get("embeddings")
            
            if embeddings:
                # Use local embedding search
                frame_indices = await self._search_with_embeddings(query, embeddings, topk)
                metadata = {
                    "search_type": "local_embeddings",
                    "query": query,
                    "topk": topk,
                    "num_results": len(frame_indices),
                    "status": "success"
                }
            else:
                # Use background server search
                frame_indices, metadata = await self._search_video_frames(query, topk)

            # Store results in instance dictionary
            self._instance_dict[instance_id]["search_results"].append({
                "query": query,
                "frame_indices": frame_indices,
                "metadata": metadata
            })

            # Prepare response
            response_data = {
                "query": query,
                "frame_indices": frame_indices,
                "topk": topk,
                "metadata": metadata
            }

            # Convert metadata to metrics
            metrics = {
                "query": query,
                "num_results": len(frame_indices),
                "search_type": metadata.get("search_type", "background_server"),
                "status": metadata.get("status", "success"),
                "error": metadata.get("error")
            }

            return ToolResponse(text=json.dumps(response_data)), 0.0, metrics

        except Exception as e:
            error_result = json.dumps({"error": f"Video search execution failed: {e}"})
            logger.error(f"[VideoSearchTool] Execution failed: {e}")
            return ToolResponse(text=error_result), 0.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        """Calculate reward based on search results quality."""
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
