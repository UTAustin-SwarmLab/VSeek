"""VSeek agent that uses UniversalVTG temporal grounding for retrieval.

Drop-in replacement for VSeekAgent that calls the VTG server's /ground
endpoint instead of /search. The temporal grounding model localizes
relevant moments given a text query rather than doing embedding similarity.
"""

import requests

from vseek.agent.video_agent import VSeekAgent


class VSeekVTGAgent(VSeekAgent):
    """VSeekAgent variant using UniversalVTG temporal grounding."""

    def __init__(self, config):
        super().__init__(config)
        self.dataset_name = getattr(config.dataset, "name", "lvb")

    def search_video(self, search_query: str, video_id: str | None = None, topk: int | None = None) -> list[int]:
        """Run temporal grounding instead of embedding similarity search."""
        params = {
            "query": search_query,
            "dataset_name": self.dataset_name,
        }
        if topk is not None:
            params["topk"] = topk
        if video_id:
            params["video_id"] = video_id
        try:
            resp = requests.get(
                f"{self.retriever_server_url}/ground",
                params=params,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            return [int(i) for i in data.get("window_indices", [])]
        except Exception:
            return []

    def search_subtitle(self, search_query: str, video_id: str | None = None, topk: int | None = None) -> list[int]:
        """Subtitle retrieval via the VTG server's BGE-backed /search_subtitle.

        Unlike visual grounding, this matches the query against per-video subtitle
        text embeddings and returns the window indices where the matched lines are
        spoken. Returns [] when the video has no subtitle index (e.g. mlvu/lvbench).
        """
        params = {
            "query": search_query,
            "dataset_name": self.dataset_name,
        }
        if topk is not None:
            params["topk"] = topk
        if video_id:
            params["video_id"] = video_id
        try:
            resp = requests.get(
                f"{self.retriever_server_url}/search_subtitle",
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return [int(i) for i in data.get("subtitle_indices", [])]
        except Exception:
            return []
