"""Video temporal grounding server using UniversalVTG.

This server hosts the full UniversalVTG temporal grounding model. Instead of
cosine-similarity retrieval over pre-indexed window embeddings, it runs the
complete temporal grounding pipeline: given a text query and video features,
it localizes relevant temporal segments and returns frame indices corresponding
to those segments.

Requires pre-extracted PE visual features per video (stored as .pt files).

Start with::

    python -m vseek.tools.server_vtg \
        retriever.port=9002 \
        retriever.universalvtg_path=/path/to/universalvtg \
        retriever.features_path=/path/to/pe_features \
        retriever.experiment_path=/path/to/experiments/universalvtg
"""

import os
import sys
import logging
import traceback
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from flask import Flask, request, jsonify
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

from data.videomme import VideoMME
from data.lvbench import LVBench
from data.mlvu import MLVU
from data.lvb import LongVideoBench
from vseek.data.frame import VideoFrames

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

FEATURES_FPS = 2.0


class VTGSearchServer:
    """Temporal grounding server backed by the full UniversalVTG model.

    Unlike the embedding-similarity servers, this one runs the actual temporal
    grounding model at query time to localize relevant moments in the video.
    """

    def __init__(self, args):
        try:
            self.server_url = args.retriever.server_url
        except Exception:
            self.server_url = "http://0.0.0.0:9002"

        self.features_path = args.retriever.features_path
        self.index_path = args.retriever.index_path
        self.topk = args.retriever.topk
        self.gpu_number = args.retriever.gpu_number
        self.dataset_name = args.retriever.dataset_name
        self.window_size = args.retriever.window_size
        self.desired_interval_in_sec = args.retriever.desired_interval_in_sec

        universalvtg_path = args.retriever.universalvtg_path
        experiment_path = getattr(args.retriever, "experiment_path",
                                  os.path.join(universalvtg_path, "experiments/universalvtg"))
        checkpoint_name = getattr(args.retriever, "checkpoint_name", "best")
        device = f"cuda:{self.gpu_number}" if torch.cuda.is_available() else "cpu"

        # Add universalvtg to path for imports
        uvtg = Path(universalvtg_path).resolve()
        for p in [str(uvtg / "perception_models"), str(uvtg)]:
            if p not in sys.path:
                sys.path.insert(0, p)

        from universal_vtg_inference import UniversalVTG

        self.vtg_model = UniversalVTG(
            experiment_name=experiment_path,
            checkpoint_name=checkpoint_name,
            device=device,
        )
        self.device = device

        # VTG prediction parameters
        self.top_k_segments = getattr(args.retriever, "top_k_segments", 5)
        self.score_thresh = getattr(args.retriever, "score_thresh", 0.1)

        # Load datasets
        if isinstance(self.dataset_name, str):
            self.dataset_name = [self.dataset_name]
        logger.info(f"dataset_name: {self.dataset_name}")

        self.entries: Dict[str, list] = {}
        for dataset_name in self.dataset_name:
            if dataset_name == "lvb":
                self.entries[dataset_name] = LongVideoBench(args).load_data()
            elif dataset_name == "videomme":
                self.entries[dataset_name] = VideoMME(args).load_data()
            elif dataset_name == "lvbench":
                self.entries[dataset_name] = LVBench(args).load_data()
            elif dataset_name == "mlvu":
                self.entries[dataset_name] = MLVU(args).load_data()
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

        # Load pre-extracted PE visual features
        self.video_features: Dict[str, Dict[str, torch.Tensor]] = {}
        self.video_durations: Dict[str, Dict[str, float]] = {}
        for dn in self.dataset_name:
            self.video_features[dn] = {}
            self.video_durations[dn] = {}

        self._load_features()

        # Subtitle retrieval harness (BGE text encoder) — optional, fully
        # decoupled from ViClip. Loads the self-contained per-video index built
        # by scripts/data_ops/embed_subtitles_bge.py. Subtitle vectors are
        # precomputed; the encoder here is used only to embed the live query.
        self.subtitle_enabled = False
        sub_cfg = getattr(args.retriever, "subtitle", None)
        if sub_cfg is not None and getattr(sub_cfg, "enabled", False):
            from vseek.video_embedding.bge_text import BGETextEncoder

            self.subtitle_index_path = sub_cfg.index_path
            self.subtitle_encoder = BGETextEncoder(
                model_name=sub_cfg.encoder_model,
                query_instruction=sub_cfg.query_instruction,
                device=device,
            )
            self.subtitle_index: Dict[str, Dict[str, dict]] = {}
            self._load_subtitle_index()
            self.subtitle_enabled = True

    def _load_subtitle_index(self):
        """Load per-video BGE subtitle indices (subtitles + embeddings + map)."""
        root = Path(self.subtitle_index_path)
        for dataset_name in self.entries.keys():
            self.subtitle_index[dataset_name] = {}
            ds_dir = root / dataset_name
            if not ds_dir.exists():
                logger.warning(f"No BGE subtitle index for {dataset_name}: {ds_dir}")
                continue
            loaded = 0
            seen = set()
            for entry in self.entries[dataset_name]:
                video_id = entry["metadata"]["video_id"]
                if video_id in seen:
                    continue
                seen.add(video_id)
                f = ds_dir / f"{video_id}.pt"
                if not f.exists():
                    continue
                try:
                    data = torch.load(f, map_location="cpu", weights_only=False)
                    emb = data["embeddings"]
                    if emb.numel() == 0:
                        continue
                    self.subtitle_index[dataset_name][video_id] = {
                        "subtitles": data["subtitles"],
                        "embeddings": emb.to(self.device),
                        "window_by_subtitle": data["window_by_subtitle"],
                        "window_size": int(data.get("window_size", self.window_size)),
                    }
                    loaded += 1
                except Exception as e:
                    logger.error(f"Failed to load subtitle index for {video_id}: {e}")
            logger.info(f"Loaded {loaded} subtitle indices for {dataset_name}")

    def _load_features(self):
        """Load pre-extracted PE visual features for all videos."""
        features_root = Path(self.features_path)

        for dataset_name in self.entries.keys():
            total_loaded = 0
            seen_ids = set()

            for entry in self.entries[dataset_name]:
                video_id = entry["metadata"]["video_id"]
                if video_id in seen_ids:
                    continue
                seen_ids.add(video_id)

                feat_path = features_root / dataset_name / f"{video_id}.pt"
                if not feat_path.exists():
                    # Try without dataset subfolder
                    feat_path = features_root / f"{video_id}.pt"
                if not feat_path.exists():
                    logger.warning(f"Missing features for {video_id}: {feat_path}")
                    continue

                try:
                    data = torch.load(feat_path, map_location="cpu", weights_only=True)
                    if isinstance(data, dict):
                        features = data["features"]  # (D, T)
                        duration = data.get("duration", features.shape[-1] / FEATURES_FPS)
                    else:
                        features = data  # assume (D, T) tensor
                        duration = features.shape[-1] / FEATURES_FPS

                    self.video_features[dataset_name][video_id] = features
                    self.video_durations[dataset_name][video_id] = duration
                    total_loaded += 1
                except Exception as e:
                    logger.error(f"Failed to load features for {video_id}: {e}")

            logger.info(f"Loaded {total_loaded} video features for {dataset_name}")

    def _segments_to_frame_indices(
        self,
        segments: torch.Tensor,
        duration: float,
        total_frames: int,
    ) -> list[int]:
        """Convert temporal segments (seconds) to frame indices.

        Args:
            segments: (N, 2) tensor of [start, end] in seconds
            duration: Video duration in seconds
            total_frames: Total number of indexed frames (at desired_interval_in_sec)

        Returns:
            Sorted list of unique frame indices
        """
        frame_indices = set()
        for seg in segments:
            start_sec = max(0.0, seg[0].item())
            end_sec = min(duration, seg[1].item())
            # Convert seconds to frame indices (frames are at desired_interval_in_sec)
            start_frame = int(start_sec / self.desired_interval_in_sec)
            end_frame = int(end_sec / self.desired_interval_in_sec)
            start_frame = max(0, min(start_frame, total_frames - 1))
            end_frame = max(0, min(end_frame, total_frames - 1))
            for fi in range(start_frame, end_frame + 1):
                frame_indices.add(fi)
        return sorted(frame_indices)

    def _frame_indices_to_window_indices(self, frame_indices: list[int]) -> list[int]:
        """Convert frame indices to window indices."""
        window_indices = set()
        for fi in frame_indices:
            window_indices.add(fi // self.window_size)
        return sorted(window_indices)

    def temporal_ground(self):
        """Run temporal grounding for a text query on a video.

        Returns:
            JSON response with temporal segments, frame indices, and window indices.
        """
        try:
            if request.method == "GET":
                query = request.args.get("query")
                topk = int(request.args.get("topk", self.topk))
                video_id = request.args.get("video_id")
                dataset_name = request.args.get("dataset_name")
            else:
                data = request.get_json()
                query = data.get("query")
                topk = data.get("topk", self.topk)
                video_id = data.get("video_id")
                dataset_name = data.get("dataset_name")

            if not query:
                return jsonify({"error": "Query parameter is required"}), 400
            if not video_id:
                return jsonify({"error": "video_id parameter is required"}), 400
            if not dataset_name:
                return jsonify({"error": "dataset_name parameter is required"}), 400

            if video_id not in self.video_features.get(dataset_name, {}):
                return jsonify({"error": f"Video {video_id} not found in {dataset_name}"}), 404

            vid_features = self.video_features[dataset_name][video_id]
            duration = self.video_durations[dataset_name][video_id]

            # Run temporal grounding
            results = self.vtg_model.predict(
                vid_features,
                query,
                top_k=self.top_k_segments,
                score_thresh=self.score_thresh,
                feature_fps=FEATURES_FPS,
                duration=duration,
            )

            segments = results["segments"][0]  # (N, 2)
            scores = results["scores"][0]      # (N,)

            if len(segments) == 0:
                return jsonify({
                    "query": query,
                    "video_id": video_id,
                    "segments": [],
                    "scores": [],
                    "frame_indices": [],
                    "window_indices": [],
                    "metadata": {"search_type": "temporal_grounding", "status": "no_results"},
                })

            # Total indexed frames for this video
            total_frames = int(duration / self.desired_interval_in_sec)

            # Convert segments to frame indices, taking top-k segments
            # and limiting total frame count by topk windows
            segments_list = segments.cpu().tolist()
            scores_list = scores.cpu().tolist()

            # Get frame indices from all predicted segments
            frame_indices = self._segments_to_frame_indices(segments, duration, total_frames)
            window_indices = self._frame_indices_to_window_indices(frame_indices)

            # Limit to topk windows (sorted by temporal order)
            if len(window_indices) > topk:
                # Prioritize windows that overlap with higher-scored segments
                window_scores = {}
                for seg, score in zip(segments_list, scores_list):
                    start_w = int(seg[0] / self.desired_interval_in_sec) // self.window_size
                    end_w = int(seg[1] / self.desired_interval_in_sec) // self.window_size
                    for w in range(start_w, end_w + 1):
                        window_scores[w] = max(window_scores.get(w, 0.0), score)
                # Sort by score descending, take topk
                scored_windows = sorted(window_scores.items(), key=lambda x: x[1], reverse=True)
                window_indices = sorted([w for w, _ in scored_windows[:topk]])

            return jsonify({
                "query": query,
                "video_id": video_id,
                "segments": segments_list,
                "scores": scores_list,
                "frame_indices": frame_indices,
                "window_indices": window_indices,
                "topk": topk,
                "duration": duration,
                "metadata": {"search_type": "temporal_grounding", "status": "success"},
            })

        except Exception as e:
            logger.error(f"Temporal grounding failed: {e}")
            logger.debug(traceback.format_exc())
            return jsonify({"error": f"Temporal grounding failed: {e}"}), 500

    def search_subtitle(self):
        """Retrieve windows by matching the query against subtitle text (BGE).

        Embeds the query with the BGE encoder, scores it against the video's
        precomputed subtitle embeddings, then walks matches in descending
        similarity accumulating their windows until at least ``topk`` are found.
        Returns window indices in the same space as /ground. Videos without a
        subtitle index (e.g. mlvu/lvbench) return an empty result, not an error.
        """
        try:
            if not getattr(self, "subtitle_enabled", False):
                return jsonify({"error": "Subtitle search is disabled on this server"}), 400

            if request.method == "GET":
                query = request.args.get("query")
                topk = int(request.args.get("topk", self.topk))
                video_id = request.args.get("video_id")
                dataset_name = request.args.get("dataset_name")
            else:
                data = request.get_json()
                query = data.get("query")
                topk = int(data.get("topk", self.topk))
                video_id = data.get("video_id")
                dataset_name = data.get("dataset_name")

            if not query:
                return jsonify({"error": "Query parameter is required"}), 400
            if not video_id:
                return jsonify({"error": "video_id parameter is required"}), 400
            if not dataset_name:
                return jsonify({"error": "dataset_name parameter is required"}), 400

            idx = self.subtitle_index.get(dataset_name, {}).get(video_id)
            if not idx:
                return jsonify({
                    "query": query,
                    "video_id": video_id,
                    "subtitle_indices": [],
                    "closest_subtitles": [],
                    "metadata": {"search_type": "subtitles", "status": "no_subtitles"},
                })

            q = self.subtitle_encoder.encode_query(query).to(self.device)  # [D], normalized
            sims = torch.matmul(idx["embeddings"], q)  # [N] (subtitle embs are normalized)
            order = torch.argsort(sims, descending=True).cpu().tolist()

            window_size = idx["window_size"]
            wbs = idx["window_by_subtitle"]
            subtitles = idx["subtitles"]
            frame_indices: list[int] = []
            closest: list[str] = []
            for i in order:
                sub = subtitles[i]
                closest.append(sub)
                frame_indices += wbs.get(sub, [])
                frame_indices = list(dict.fromkeys(frame_indices))  # dedup, keep order
                if len(frame_indices) >= topk:
                    break
            window_indices = sorted({fi // window_size for fi in frame_indices})

            return jsonify({
                "query": query,
                "video_id": video_id,
                "subtitle_indices": window_indices,
                "closest_subtitles": closest,
                "topk": topk,
                "metadata": {"search_type": "subtitles", "status": "success"},
            })

        except Exception as e:
            logger.error(f"Subtitle search failed: {e}")
            logger.debug(traceback.format_exc())
            return jsonify({"error": f"Subtitle search failed: {e}"}), 500


# ------------------------------------------------------------------
# Flask routes
# ------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "service": "vtg_search_server"})


@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "message": "UniversalVTG Temporal Grounding Server",
        "version": "1.0.0",
        "endpoints": {
            "ground": "/ground",
            "search_subtitle": "/search_subtitle",
            "health": "/health",
        },
    })


def create_server(cfg: DictConfig):
    return VTGSearchServer(cfg)


def main():
    # Load config via hydra-core BEFORE adding universalvtg to sys.path,
    # because universalvtg has a local `hydra/` submodule that shadows hydra-core.
    with hydra.initialize(version_base=None, config_path="../config/retriever"):
        cfg = hydra.compose(config_name="config_vtg", overrides=sys.argv[1:])

    # Now add universalvtg to sys.path and swap hydra-core for the local
    # hydra/ submodule (Mamba/Hydra model). We no longer need hydra-core.
    uvtg_path = Path(cfg.retriever.universalvtg_path).resolve()
    for p in [str(uvtg_path / "perception_models"), str(uvtg_path)]:
        if p not in sys.path:
            sys.path.insert(0, p)
    # Evict cached hydra-core so `from hydra.modules...` resolves to local submodule
    for mod_name in [k for k in sys.modules if k == "hydra" or k.startswith("hydra.")]:
        del sys.modules[mod_name]

    server = create_server(cfg)
    app.add_url_rule("/ground", view_func=server.temporal_ground, methods=["GET", "POST"])
    app.add_url_rule("/search_subtitle", view_func=server.search_subtitle, methods=["GET", "POST"])

    logger.info("Starting UniversalVTG Temporal Grounding Server")
    logger.info(f"Dataset: {server.dataset_name}")
    logger.info(f"Features path: {server.features_path}")
    logger.info(f"Loaded videos: {sum(len(v) for v in server.video_features.values())}")
    host = server.server_url.split(":")[1].split("/")[-1]
    port = int(server.server_url.split(":")[2])
    app.run(host=host, port=port)


if __name__ == "__main__":
    main()
