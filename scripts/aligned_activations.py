"""Aligned sparse TopK shard access and row hashing."""

from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

from pathlib import Path

import numpy as np

import torch

from scripts.matching_io import read_json

GEMINI_MODEL = "gemini_m131072_k128"

NEMOTRON_MODEL = "nemotron_m131072_k128"

TOP_K = 128

AGREEMENT_STRONG_THRESHOLD = 0.1

AGREEMENT_BOTTOM_THRESHOLDS = (0.01, 0.03, 0.05, 0.1)

class FixedTopKShard:
    def __init__(
        self,
        cache_root: Path,
        relative_shard: str,
        model: str,
        width: int,
    ) -> None:
        model_dir = cache_root / "shards" / relative_shard / model
        complete = read_json(model_dir.parent / "complete.json")
        output = complete.get("outputs", {}).get(model)
        if not isinstance(output, dict):
            raise ValueError(f"Missing completion metadata for {relative_shard}/{model}")
        self.rows, self.width = (int(value) for value in output["shape"])
        if self.width != width or int(output["top_k"]) != TOP_K:
            raise ValueError(f"Unexpected shape or Top-K for {relative_shard}/{model}")
        self.data = np.load(
            _paper_location(model_dir / "data.npy"), mmap_mode="r", allow_pickle=False
        ).reshape(self.rows, TOP_K)
        self.indices = np.load(
            _paper_location(model_dir / "indices.npy"), mmap_mode="r", allow_pickle=False
        ).reshape(self.rows, TOP_K)

    def batch(
        self, start: int, stop: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.from_numpy(
            np.asarray(self.indices[start:stop], dtype=np.int64).copy()
        ).to(device, non_blocking=True)
        values = torch.from_numpy(
            np.asarray(self.data[start:stop], dtype=np.float32).copy()
        ).to(device, non_blocking=True)
        return indices, values

def row_hash(
    global_rows: np.ndarray, sketch_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(global_rows, dtype=np.uint64) + np.uint64(
        0x9E3779B97F4A7C15
    )
    values = (values ^ (values >> np.uint64(30))) * np.uint64(
        0xBF58476D1CE4E5B9
    )
    values = (values ^ (values >> np.uint64(27))) * np.uint64(
        0x94D049BB133111EB
    )
    values ^= values >> np.uint64(31)
    buckets = (values % np.uint64(sketch_dim)).astype(np.int64)
    signs = np.where(values >> np.uint64(63), 1.0, -1.0).astype(np.float32)
    return buckets, signs
