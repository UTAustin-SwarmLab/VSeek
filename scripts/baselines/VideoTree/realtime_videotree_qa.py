import base64
import os
import re
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from openai import OpenAI
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from transformers import CLIPModel, CLIPProcessor


MODEL_VARIANTS: dict[str, dict[str, Optional[str]]] = {
    "gpt5": {
        "model_name": "gpt-5",
        "base_url": None,
    },
    "qwen4b_instruct": {
        "model_name": "Qwen/Qwen3-VL-4B-Instruct",
        "base_url": "http://127.0.0.1:8000/v1",
    },
}


@dataclass
class RuntimeVideoTreeConfig:
    # Supported convenient variants:
    # - "gpt5"
    # - "qwen4b_instruct" (OpenAI-compatible local endpoint)
    model_variant: str = "gpt5"
    model_name: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    n_passes: int = 4
    temperature: float = 0.1
    max_output_tokens: int = 256
    extraction_fps: float = 1.0
    max_frames_to_read: int = 512
    max_keyframes: int = 64
    image_max_width: int = 384
    image_max_height: int = 384
    image_quality: int = 85
    embedding_model_name: str = "openai/clip-vit-base-patch32"
    relevance_prompt_model: Optional[str] = None
    qa_prompt_model: Optional[str] = None

    # Adaptive breadth expansion (strict parity with adaptive_breath_expansion.py)
    init_cluster_num: int = 8
    max_cluster_num: int = 32
    default_adpative_rate: int = 2
    iter_threshold: int = 4

    # Depth expansion (strict parity with depth_expansion.py)
    depth_num_subclusters: int = 4
    depth_num_subsubclusters: int = 4

    # Image budget when calling VLM (after keyframe index is built)
    max_images_per_call: int = 64

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 7


class RuntimeVideoTreeQA:
    """
    Runtime VideoTree QA with strict pipeline parity.

    Input entry:
      {
        "question": "...",
        "answer": "...",            # ground truth answer
        "video": "/path/to.mp4"   # or key "video_path"
        "choices": ["A", ...]      # optional (recommended for strict MCQ parity)
      }

    Strict parity stages (but run online, no offline files):
      1) Adaptive breadth expansion:
           - cluster frame features
           - choose representative frame per cluster
           - ask LLM for per-cluster relevance scores in [1,2,3]
           - increase clusters until enough "3" scores or max clusters reached
      2) Relevance-based depth expansion:
           - hierarchical sub/sub-sub clustering conditioned on relevance
           - choose representative indices in temporal order
      3) QA:
           - prompt VLM with question + selected keyframes for N passes
    """

    def __init__(self, config: RuntimeVideoTreeConfig):
        self.config = config
        if config.model_variant not in MODEL_VARIANTS:
            raise ValueError(
                f"Unsupported model_variant='{config.model_variant}'. "
                f"Available: {sorted(MODEL_VARIANTS.keys())}"
            )
        variant_cfg = MODEL_VARIANTS[config.model_variant]
        # Variant provides defaults; explicit fields can still override.
        if not config.model_name:
            config.model_name = variant_cfg["model_name"] or "gpt-5"
        if config.base_url is None and variant_cfg["base_url"] is not None:
            config.base_url = variant_cfg["base_url"]

        api_key = config.api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"
        if "gpt" in config.model_name.lower() and (config.api_key is None and os.getenv("OPENAI_API_KEY") is None):
            raise ValueError("OPENAI_API_KEY is required when using GPT models.")

        self.client = OpenAI(api_key=api_key, base_url=config.base_url)
        self.clip_processor = CLIPProcessor.from_pretrained(config.embedding_model_name)
        self.clip_model = CLIPModel.from_pretrained(config.embedding_model_name).to(config.device)
        self.clip_model.eval()
        self.relevance_model_name = config.relevance_prompt_model or config.model_name
        self.qa_model_name = config.qa_prompt_model or config.model_name

    @staticmethod
    def _parse_answer(text: str) -> str:
        if not text:
            return ""
        answer_tag = re.findall(
            r"<\s*answer\s*>([\s\S]*?)<\s*/\s*answer\s*>",
            text,
            flags=re.IGNORECASE,
        )
        target = answer_tag[-1].strip() if answer_tag else text.strip()
        numbers = re.findall(r"\d+", target)
        letters = re.findall(r"[a-zA-Z]+", target)
        if numbers:
            return numbers[0]
        if letters:
            return letters[0]
        return target

    @staticmethod
    def _parse_relevance_scores(text: str, expected_len: int) -> list[int]:
        if not text:
            return [3] * expected_len
        match = re.search(r"frame relevance:\s*\[([0-9,\s]+)\]", text, flags=re.IGNORECASE)
        if not match:
            return [3] * expected_len
        raw = [x.strip() for x in match.group(1).split(",") if x.strip()]
        scores: list[int] = []
        for token in raw:
            try:
                v = int(token)
                if v < 1:
                    v = 1
                if v > 3:
                    v = 3
                scores.append(v)
            except ValueError:
                continue
        if not scores:
            return [3] * expected_len
        if len(scores) < expected_len:
            scores.extend([scores[-1]] * (expected_len - len(scores)))
        return scores[:expected_len]

    def _read_video_frames(self, video_path: str) -> list[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        native_fps = cap.get(cv2.CAP_PROP_FPS)
        native_fps = native_fps if native_fps and native_fps > 0 else 30.0
        stride = max(1, int(round(native_fps / max(self.config.extraction_fps, 1e-6))))

        frames_rgb: list[np.ndarray] = []
        idx = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if idx % stride == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames_rgb.append(frame_rgb)
                if len(frames_rgb) >= self.config.max_frames_to_read:
                    break
            idx += 1
        cap.release()
        return frames_rgb

    def _embed_frames(self, frames_rgb: list[np.ndarray]) -> torch.Tensor:
        with torch.inference_mode():
            inputs = self.clip_processor(images=frames_rgb, return_tensors="pt", padding=True).to(self.config.device)
            feats = self.clip_model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return feats

    @staticmethod
    def _closest_points_per_cluster(
        x: torch.Tensor, cluster_ids: np.ndarray, cluster_centers: np.ndarray
    ) -> dict[int, list[int]]:
        closest_points_idx_per_cluster = {cluster_id: [] for cluster_id in range(len(cluster_centers))}
        for cluster_id in range(len(cluster_centers)):
            indices_in_cluster = np.where(cluster_ids == cluster_id)[0]
            if indices_in_cluster.size == 0:
                continue
            points_in_cluster = x[torch.tensor(indices_in_cluster, dtype=torch.long, device=x.device)]
            center = torch.tensor(cluster_centers[cluster_id], dtype=points_in_cluster.dtype, device=x.device)
            distances = torch.norm(points_in_cluster - center, dim=1)
            if distances.numel() == 0:
                continue
            closest_idx_in_cluster = int(torch.argmin(distances).item())
            closest_global_idx = int(indices_in_cluster[closest_idx_in_cluster])
            closest_points_idx_per_cluster[cluster_id].append(closest_global_idx)
        return closest_points_idx_per_cluster

    def _encode_frame(self, frame_rgb: np.ndarray) -> str:
        h, w = frame_rgb.shape[:2]
        scale = min(self.config.image_max_width / w, self.config.image_max_height / h)
        if scale < 1.0:
            frame_rgb = cv2.resize(
                frame_rgb,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, int(self.config.image_quality)])
        if not ok:
            raise ValueError("JPEG encoding failed.")
        return base64.b64encode(buf).decode("utf-8")

    def _build_relevance_messages(
        self, question: str, candidate_frames: list[np.ndarray], choices: Optional[list[str]] = None
    ) -> list[dict[str, Any]]:
        user_content: list[dict[str, Any]] = []
        for frame in candidate_frames:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{self._encode_frame(frame)}"},
                }
            )
        choice_text = ""
        if choices:
            lines = [f"{chr(65 + i)}: {opt}" for i, opt in enumerate(choices)]
            choice_text = "\nOptions:\n" + "\n".join(lines)
        user_content.append(
            {
                "type": "text",
                "text": (
                    "You are given ordered representative keyframes sampled from a video.\n"
                    f"Question: {question}\n"
                    f"{choice_text}\n"
                    "Return EXACTLY this format:\n"
                    "prediction: <A/B/C/D/E or short answer>\n"
                    "explanation: <one sentence>\n"
                    "confidence: <1-100>\n"
                    "frame relevance: [r1, r2, ..., rN]\n"
                    "Each relevance ri must be in {1,2,3}, where 3 means highly relevant."
                ),
            }
        )
        return [
            {
                "role": "system",
                "content": (
                    "You are a video understanding model. Score relevance of each provided frame to the question."
                ),
            },
            {"role": "user", "content": user_content},
        ]

    def _build_qa_messages(
        self, question: str, keyframes: list[np.ndarray], choices: Optional[list[str]] = None
    ) -> list[dict[str, Any]]:
        user_content: list[dict[str, Any]] = []
        for frame in keyframes:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{self._encode_frame(frame)}"},
                }
            )
        choice_text = ""
        if choices:
            lines = [f"{chr(65 + i)}: {opt}" for i, opt in enumerate(choices)]
            choice_text = "\nOptions:\n" + "\n".join(lines)
            answer_hint = "Return only one option letter (A-E)."
        else:
            answer_hint = "Return a concise final answer."
        user_content.append(
            {
                "type": "text",
                "text": (
                    "Answer the question using only the provided keyframes.\n"
                    f"{answer_hint}\n"
                    f"Question: {question}{choice_text}"
                ),
            }
        )
        return [
            {
                "role": "system",
                "content": (
                    "You are a video question answering assistant. "
                    "Use the provided keyframes as visual evidence."
                ),
            },
            {"role": "user", "content": user_content},
        ]

    def _chat_once(self, messages: list[dict[str, Any]], model_name: str) -> str:
        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        try:
            resp = self.client.chat.completions.create(
                **kwargs,
                max_completion_tokens=self.config.max_output_tokens,
            )
        except TypeError:
            resp = self.client.chat.completions.create(
                **kwargs,
                max_tokens=self.config.max_output_tokens,
            )
        return resp.choices[0].message.content or ""

    @staticmethod
    def _cosine_distance(points: torch.Tensor, centroid: torch.Tensor) -> torch.Tensor:
        points_normalized = F.normalize(points, dim=1)
        centroid_normalized = F.normalize(centroid.unsqueeze(0), dim=1)
        return 1 - torch.mm(points_normalized, centroid_normalized.T).squeeze()

    def _hierarchical_clustering_with_external_primary(
        self,
        video_features: np.ndarray,
        cluster_ids: list[int],
        relevance_scores: list[int],
        num_subclusters: int,
        num_subsubclusters: int,
    ) -> dict[int, Any]:
        clusters: dict[int, Any] = {i: {} for i in range(0, max(cluster_ids) + 1)}
        for cluster_id in sorted(set(cluster_ids)):
            primary_indices = [i for i, x in enumerate(cluster_ids) if x == cluster_id]
            score = relevance_scores[cluster_id] if cluster_id < len(relevance_scores) else 3
            if len(primary_indices) < 2:
                clusters[cluster_id] = primary_indices
                continue
            sub_features = video_features[primary_indices]
            if score == 1:
                clusters[cluster_id] = primary_indices
                continue

            # Match baseline depth_expansion behavior: ward linkage + maxclust split.
            sub_k = min(num_subclusters, len(primary_indices))
            linked_sub = linkage(sub_features, method="ward")
            sub_cluster_labels = fcluster(linked_sub, sub_k, criterion="maxclust") - 1

            if score == 2:
                clusters[cluster_id] = {
                    i: [primary_indices[j] for j in np.where(sub_cluster_labels == i)[0]]
                    for i in range(0, sub_k)
                }
                continue

            for subcluster_id in range(0, sub_k):
                sub_indices = np.where(sub_cluster_labels == subcluster_id)[0]
                if len(sub_indices) < 2:
                    continue
                subsub_features = sub_features[sub_indices]
                subsub_k = min(num_subsubclusters, len(sub_indices))
                linked_subsub = linkage(subsub_features, method="ward")
                subsub_cluster_labels = fcluster(linked_subsub, subsub_k, criterion="maxclust") - 1
                clusters[cluster_id][subcluster_id] = {}
                for subsubcluster_id in range(0, subsub_k):
                    final_indices = sub_indices[np.where(subsub_cluster_labels == subsubcluster_id)[0]]
                    original_indices = [primary_indices[i] for i in final_indices]
                    clusters[cluster_id][subcluster_id][subsubcluster_id] = original_indices
        return clusters

    def _find_closest_points_in_temporal_order_subsub(
        self, x: torch.Tensor, clusters: dict[int, Any], relevance_scores: list[int]
    ) -> list[int]:
        closest_points_indices: list[int] = []
        x_cpu = x.detach().cpu()
        for cluster_id, cluster_data in clusters.items():
            relevance = relevance_scores[cluster_id] if cluster_id < len(relevance_scores) else 3
            if isinstance(cluster_data, list):
                cluster_arr = np.array(cluster_data)
                if cluster_arr.size == 0:
                    continue
                points_in_cluster = x_cpu[torch.tensor(cluster_arr, dtype=torch.long)]
                cluster_centroid = points_in_cluster.mean(dim=0)
                distances = self._cosine_distance(points_in_cluster, cluster_centroid)
                if distances.numel() > 0:
                    closest_idx = int(torch.argmin(distances).item())
                    closest_points_indices.append(int(cluster_arr[closest_idx]))
                continue

            if not isinstance(cluster_data, dict):
                continue

            primary_indices: list[np.ndarray] = []
            for subcluster_data in cluster_data.values():
                if isinstance(subcluster_data, dict):
                    for sub_data in subcluster_data.values():
                        arr = np.array(sub_data)
                        if arr.size > 0:
                            primary_indices.append(arr)
                elif isinstance(subcluster_data, list) and len(subcluster_data) > 0:
                    primary_indices.append(np.array(subcluster_data))

            if primary_indices:
                primary_concat = np.concatenate(primary_indices)
                primary_points = x_cpu[torch.tensor(primary_concat, dtype=torch.long)]
                primary_centroid = primary_points.mean(dim=0)
                primary_distances = self._cosine_distance(primary_points, primary_centroid)
                if primary_distances.numel() > 0:
                    closest_primary_idx = int(torch.argmin(primary_distances).item())
                    closest_points_indices.append(int(primary_concat[closest_primary_idx]))

            if relevance == 1:
                continue

            for subclusters in cluster_data.values():
                if isinstance(subclusters, dict):
                    for indices in subclusters.values():
                        if len(indices) == 0:
                            continue
                        indices_tensor = torch.tensor(indices, dtype=torch.long)
                        points = x_cpu[indices_tensor]
                        centroid = points.mean(dim=0)
                        distances = self._cosine_distance(points, centroid)
                        if distances.numel() > 0:
                            closest_idx = int(torch.argmin(distances).item())
                            closest_points_indices.append(int(indices[closest_idx]))
                elif isinstance(subclusters, list):
                    arr = np.array(subclusters)
                    if arr.size == 0:
                        continue
                    points = x_cpu[torch.tensor(arr, dtype=torch.long)]
                    centroid = points.mean(dim=0)
                    distances = self._cosine_distance(points, centroid)
                    if distances.numel() > 0:
                        closest_idx = int(torch.argmin(distances).item())
                        closest_points_indices.append(int(arr[closest_idx]))

        return sorted(set(closest_points_indices))

    def _adaptive_breadth_expansion(
        self, question: str, all_frames: list[np.ndarray], frame_embeddings: torch.Tensor, choices: Optional[list[str]]
    ) -> tuple[list[int], list[int], list[int], str]:
        tree_node = [0]
        cluster_num = min(self.config.init_cluster_num, len(all_frames))
        max_cluster_num = min(self.config.max_cluster_num, len(all_frames))
        adaptive_rate = max(1, self.config.default_adpative_rate)
        iter_threshold = self.config.iter_threshold

        cluster_ids_list: list[int] = [0 for _ in all_frames]
        frame_relevance: list[int] = [3]
        relevance_response = ""
        x_cpu = frame_embeddings.detach().cpu()
        emb_np = x_cpu.numpy()

        while True:
            if cluster_num <= 0:
                break
            kmeans = KMeans(n_clusters=cluster_num, random_state=self.config.seed, n_init="auto")
            cluster_ids = kmeans.fit_predict(emb_np)
            closest_points = self._closest_points_per_cluster(x_cpu, cluster_ids, kmeans.cluster_centers_)
            tree_node = sorted([v for sub in closest_points.values() for v in sub])
            if not tree_node:
                break

            candidate_frames = [all_frames[i] for i in tree_node]
            relevance_messages = self._build_relevance_messages(question, candidate_frames, choices=choices)
            relevance_response = self._chat_once(relevance_messages, model_name=self.relevance_model_name)
            frame_relevance = self._parse_relevance_scores(relevance_response, expected_len=len(tree_node))
            high_relevance_frame_num = frame_relevance.count(3)

            cluster_ids_list = [int(v) for v in cluster_ids.tolist()]
            if high_relevance_frame_num < iter_threshold:
                if cluster_num < max_cluster_num:
                    cluster_num = min(cluster_num * adaptive_rate, max_cluster_num)
                else:
                    break
            else:
                break

        return tree_node, cluster_ids_list, frame_relevance, relevance_response

    def answer_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        question = entry.get("question", "")
        gt_answer = entry.get("answer", "")
        video_path = entry.get("video") or entry.get("video_path")
        choices = entry.get("choices") or entry.get("options")
        if not question:
            raise ValueError("Entry must include a non-empty 'question'.")
        if not video_path:
            raise ValueError("Entry must include 'video' or 'video_path'.")

        all_frames = self._read_video_frames(video_path)
        if not all_frames:
            raise ValueError(f"No frames could be decoded from video: {video_path}")

        frame_embeddings = self._embed_frames(all_frames)
        width_tree_node, cluster_ids_list, relevance_scores, relevance_raw = self._adaptive_breadth_expansion(
            question=question,
            all_frames=all_frames,
            frame_embeddings=frame_embeddings,
            choices=choices,
        )

        # Depth expansion: strict parity with depth_expansion.py
        depth_clusters = self._hierarchical_clustering_with_external_primary(
            video_features=frame_embeddings.detach().cpu().numpy(),
            cluster_ids=cluster_ids_list,
            relevance_scores=relevance_scores,
            num_subclusters=self.config.depth_num_subclusters,
            num_subsubclusters=self.config.depth_num_subsubclusters,
        )
        keyframe_indices = self._find_closest_points_in_temporal_order_subsub(
            x=frame_embeddings,
            clusters=depth_clusters,
            relevance_scores=relevance_scores,
        )

        # Keep API calls bounded while preserving temporal coverage.
        if len(keyframe_indices) > self.config.max_keyframes:
            sampled = np.linspace(0, len(keyframe_indices) - 1, self.config.max_keyframes, dtype=int)
            keyframe_indices = [keyframe_indices[i] for i in sampled]
        keyframes = [all_frames[i] for i in keyframe_indices]
        if len(keyframes) > self.config.max_images_per_call:
            sampled = np.linspace(0, len(keyframes) - 1, self.config.max_images_per_call, dtype=int)
            keyframes = [keyframes[i] for i in sampled]
            keyframe_indices = [keyframe_indices[i] for i in sampled]

        messages = self._build_qa_messages(question, keyframes, choices=choices)

        answers: list[str] = []
        parsed_answers: list[str] = []
        for _ in range(self.config.n_passes):
            raw = self._chat_once(messages, model_name=self.qa_model_name)
            answers.append(raw)
            parsed_answers.append(self._parse_answer(raw))

        return {
            "question": question,
            "ground_truth_answer": gt_answer,
            "video_path": video_path,
            "choices": choices,
            "width_tree_node": width_tree_node,
            "cluster_ids_x": cluster_ids_list,
            "frame_relevance": relevance_scores,
            "relevance_response_raw": relevance_raw,
            "keyframe_indices": keyframe_indices,
            "num_keyframes": len(keyframe_indices),
            "n_passes": self.config.n_passes,
            "answers": answers,
            "parsed_answers": parsed_answers,
        }
