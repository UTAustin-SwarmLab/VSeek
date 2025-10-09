import os
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
from flask import Flask, request, jsonify
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

from vseek.video_embedding.video_clip import ViClip
from data.lvb import LongVideoBench
from vseek.data.frame import VideoFrames
import traceback
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Assuming that the dataset has been completely processed and indexed
class VideoSearchServer:
    def __init__(self, args):
        
        
        # Allow missing server_url with a sensible default
        try:
            self.server_url = args.retriever.server_url
        except Exception:
            self.server_url = "http://0.0.0.0:9000"
        self.index_path = args.retriever.index_path
        self.topk = args.retriever.topk
        self.retriever_name = args.retriever.retriever_name
        self.retriever_model = args.retriever.retriever_model
        self.faiss_gpu = args.retriever.faiss_gpu
        self.gpu_number = args.retriever.gpu_number
        self.retrieval_model_path = args.retriever.retrieval_model_path
        self.dataset_name = args.retriever.dataset_name
        self.window_size = args.retriever.window_size
        
        # Initialize retriever
        if self.retriever_name == "ViClip":
            self.retriever = ViClip(
                pretrained_model_path=self.retrieval_model_path,
                gpu_number=self.gpu_number,
            )
        else:
            raise ValueError(f"Unsupported retriever: {self.retriever_name}")
        
        # Initialize dataset
        if self.dataset_name == "lvb":
            dataset = LongVideoBench(args)
            self.entries = dataset.load_data()
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        
        # Initialize video data storage
        self.video_embeddings = {}
        self.video_subtitle_embeddings = {}
        self.video_subtitles = {}
        
        self.init_index()

    def init_index(self):
        """Initialize the video index by loading embeddings and subtitles."""
        dir_name = f"{self.dataset_name}_window_{self.window_size}"
        data_root = Path(self.index_path).joinpath(dir_name)
        
        logger.info(f"Loading video index from: {data_root}")
        
        total_success = 0
        for entry in tqdm(self.entries, desc="Processing LVB entries"):
            video_id = entry["metadata"]["video_id"]
            if video_id not in self.video_embeddings or video_id not in self.video_subtitles or video_id not in self.video_subtitle_embeddings:
                pkl_path = data_root.joinpath(f"{video_id}")

                if not pkl_path.exists():
                    # Skip if the preprocessed frames are not available
                    # You can generate them via LongVideoBench.save_it_as_vseek_data()
                    logger.warning(f"Missing preprocessed frames: {pkl_path}")
                    continue

                try:
                    video_frames = VideoFrames.load(str(pkl_path))
                    self.video_embeddings[video_id] = video_frames.embeddings
                    self.video_subtitles[video_id] = video_frames.window_by_subtitle
                    self.video_subtitle_embeddings[video_id] = video_frames.subtitle_embeddings
                    logger.debug(f"Loaded video {video_id} with {len(video_frames.embeddings)} embeddings")
                    total_success += 1
                except Exception as e:
                    logger.error(f"Failed to load video {video_id}: {e}")
                    continue
        logger.info(f"Loaded {total_success} videos")
        logger.info(f"Loaded {len(self.video_embeddings)} videos")


    def search_video(self):
        """Search for the most similar visual embeddings to the text query.

        Returns:
            JSON response with frame indices and metadata
        """
        try:
            # Get parameters from request
            if request.method == "GET":
                query = request.args.get("query")
                topk = int(request.args.get("topk", self.topk))
                video_id = request.args.get("video_id")
            else:  # POST
                data = request.get_json()
                query = data.get("query")
                topk = data.get("topk", self.topk)
                video_id = data.get("video_id")
            
            if not query:
                return jsonify({"error": "Query parameter is required"}), 400
            
            # Get text embedding for the search query
            text_embedding = self.retriever.get_text_embedding(query)

            # Get the device of the text embedding (likely GPU)
            device = text_embedding.device

            # Ensure text embedding is 1D [embedding_dim]
            if text_embedding.dim() > 1:
                text_embedding = text_embedding.squeeze()

            # Get embeddings for the specified video or all videos
            if video_id:
                if video_id not in self.video_embeddings:
                    return jsonify({"error": f"Video {video_id} not found"}), 404
                embeddings = self.video_embeddings[video_id]
            else:
                # Search across all videos - concatenate all embeddings
                embeddings = []
                for vid_embeddings in self.video_embeddings.values():
                    embeddings.extend(vid_embeddings)

            if not embeddings:
                return jsonify({"error": "No embeddings available"}), 404

            # Stack all visual embeddings into a single tensor
            # Move to same device as text embedding and normalize
            visual_embeddings = []
            visual_embeddings_keys = []
            for key, emb in embeddings.items():
                emb_device = emb.to(device)  # Move to same device as text embedding
                # Ensure embedding is 1D
                if emb_device.dim() > 1:
                    emb_device = emb_device.squeeze()
                emb_norm = emb_device / emb_device.norm(dim=-1, keepdim=True)
                visual_embeddings.append(emb_norm)
                visual_embeddings_keys.append(key)
                
            visual_embeddings_tensor = torch.stack(visual_embeddings)  # Shape: [N, embedding_dim]

            # Compute cosine similarities on GPU
            # text_embedding is already normalized from get_text_embedding
            # visual_embeddings_tensor: [N, D], text_embedding: [D] -> similarities: [N]
            similarities = torch.matmul(visual_embeddings_tensor, text_embedding)

            # Get indices sorted by similarity (descending order)
            sorted_indices = torch.argsort(similarities, descending=True)

            # Return top-k results
            frame_indices = sorted_indices.cpu().tolist()[:topk]
            window_indices = [visual_embeddings_keys[i] for i in frame_indices]
            # Return frame indices in ascending order
            
            response_data = {
                "query": query,
                "frame_indices": window_indices,
                "topk": topk,
                "video_id": video_id,
                "total_embeddings": len(embeddings),
                "metadata": {
                    "search_type": "video_frames",
                    "status": "success"
                }
            }
            
            return jsonify(response_data)

        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return jsonify({"error": f"Search failed: {str(e)}"}), 500
    
    def search_subtitle(self):
        """Search for the most similar subtitle to the text query.

        Returns:
            JSON response with subtitle indices and metadata
        """
        try:
            # Get parameters from request
            if request.method == "GET":
                query = request.args.get("query")
                topk = int(request.args.get("topk"))
                video_id = request.args.get("video_id")
            else:  # POST
                data = request.get_json()
                query = data.get("query")
                topk = data.get("topk")
                video_id = data.get("video_id")
            
            if not query:
                return jsonify({"error": "Query parameter is required"}), 400
            
            # Get text embedding for the search query
            text_embedding = self.retriever.get_text_embedding(query)

            # Get the device of the text embedding (likely GPU)
            device = text_embedding.device

            # Ensure text embedding is 1D [embedding_dim]
            if text_embedding.dim() > 1:
                text_embedding = text_embedding.squeeze()

            # Normalize the text embedding
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

            # Get subtitles for the specified video or all videos
            if video_id:
                if video_id not in self.video_subtitles:
                    return jsonify({"error": f"Video {video_id} not found for subtitles"}), 404
                subtitles = self.video_subtitles[video_id]
            else:
                # Search across all videos - concatenate all subtitles
                subtitles = []
                for vid_subtitles in self.video_subtitles.values():
                    subtitles.extend(vid_subtitles)

            if not subtitles:
                return jsonify({"error": "No subtitles available"}), 404

            # Get text embeddings for all subtitles
            subtitle_embeddings = self.video_subtitle_embeddings[video_id]
            logger.debug(f"Processing {len(subtitles)} subtitles")
            subtitle_embeddings_norm = []
            subtitle_embeddings_keys = []
            
            
            for subtitle,emb in subtitle_embeddings.items():
                emb_device = emb.to(device)  # Move to same device as query embedding
                # Ensure embedding is 1D
                if emb_device.dim() > 1:
                    emb_device = emb_device.squeeze()
                emb_norm = emb_device / emb_device.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                subtitle_embeddings_norm.append(emb_norm)
                subtitle_embeddings_keys.append(subtitle)
            # for sub in subtitle_list:
            #     emb_device = self.retriever.get_text_embedding(sub).to(device)
            #     # Ensure embedding is 1D
            #     if emb_device.dim() > 1:
            #         emb_device = emb_device.squeeze()
            #     emb_norm = emb_device / emb_device.norm(dim=-1, keepdim=True)
            #     subtitle_embeddings_norm.append(emb_norm)

            subtitle_embeddings_tensor = torch.stack(subtitle_embeddings_norm)  # Shape: [N, embedding_dim]

            # Compute cosine similarities on GPU
            # subtitle_embeddings_tensor: [N, D], text_embedding: [D] -> similarities: [N]
            similarities = torch.matmul(subtitle_embeddings_tensor, text_embedding)

            # Get indices sorted by similarity (descending order)
            print(f"similarities: {similarities}")
            print(f"sorted similarities: {torch.sort(similarities, descending=True)}")
            sorted_indices = torch.argsort(similarities, descending=True)
            print(f"sorted indices: {sorted_indices}")
            sorted_indices = sorted_indices.cpu().tolist()
            total_windows_retrieved = 0
            windows_retrieved = []

            closest_subtitles = []
            for index in sorted_indices:
                closest_subtitle = subtitle_embeddings_keys[index]
                closest_subtitles.append(closest_subtitle)
                windows_retrieved += self.video_subtitles[video_id][closest_subtitle]
                windows_retrieved = list(set(windows_retrieved))
                total_windows_retrieved = len(windows_retrieved)
                if total_windows_retrieved >= topk:
                    break
            windows_retrieved = [b//self.window_size for b in windows_retrieved]

            # Return top-k results
            
            response_data = {
                "query": query,
                "subtitle_indices": windows_retrieved,
                "topk": topk,
                "video_id": video_id,
                "total_subtitles": len(subtitles),
                "closest_subtitles": closest_subtitles,
                "metadata": {
                    "search_type": "subtitles",
                    "status": "success"
                }
            }
            
            return jsonify(response_data)

        except Exception as e:
            logger.error(f"Subtitle search failed: {str(e)}")
            print(traceback.format_exc())
            return jsonify({"error": f"Subtitle search failed: {str(e)}"}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "video_search_server"})


@app.route("/", methods=["GET"])
def root():
    """Root endpoint with API information."""
    return jsonify({
        "message": "Video Search Server",
        "version": "1.0.0",
        "endpoints": {
            "search": "/search",
            "search_subtitle": "/search_subtitle",
            "health": "/health"
        }
    })


def create_server(cfg: DictConfig):
    """Create and configure the video search server."""
    return VideoSearchServer(cfg)

global server


@hydra.main(version_base=None, config_path="../config/retriever", config_name="config")
def main(cfg: DictConfig):
    # Initialize server instance (will load index)
    server = create_server(cfg)  # or however you construct it
    app.add_url_rule("/search", view_func=server.search_video, methods=["GET"])
    app.add_url_rule("/search_subtitle", view_func=server.search_subtitle, methods=["GET"])

    logger.info("Starting Video Search Server")
    logger.info(f"Retriever: {server.retriever_name}")
    logger.info(f"Dataset: {server.dataset_name}")
    logger.info(f"Index path: {server.index_path}")
    host = server.server_url.split(":")[1].split("/")[-1]
    #convert port to int
    port = int(server.server_url.split(":")[2])
    app.run(host=host, port=port)


if __name__ == "__main__":
    main()

