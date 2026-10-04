"""Corpus text resolution, checkpoint loading and sparse TopK validation."""

from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import json

import sys

from collections import OrderedDict

from pathlib import Path

from typing import Any

import numpy as np

import pyarrow.parquet as pq

import torch

from datasets import DatasetDict, load_from_disk

sys.path.insert(0, str(_paper_path(__file__).resolve().parents[1]))

GENERAL_CORPUS_DATASETS = (
    "emotion",
    "fever",
    "gooaq",
    "hotpotqa",
    "msmarco",
    "natural-questions",
    "nfcorpus",
    "paq_10m",
    "scifact",
    "squad",
    "trivia-qa",
    "WebInstructSub",
    "miracl-corpus-text",
    "miracl-corpus-title",
)

CACHE_SCHEMA_VERSION = 1

CACHE_FILENAME = "relu_pre_topk.npy"

CACHE_METADATA_FILENAME = "complete.json"

CACHE_ACTIVATION_KIND = "selected_relu_pre_topk"

CACHE_ACTIVATION_FORMULA = "relu((embedding - b_dec) @ W_enc[:, feature])"

def load_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(_paper_location(path), map_location="cpu", mmap=True, weights_only=False)
    except TypeError:
        payload = torch.load(_paper_location(path), map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload.get("state_dict", payload))
    if "W_enc" not in state or "b_dec" not in state:
        raise KeyError(f"Checkpoint {path} does not contain W_enc and b_dec")
    return state

def discover_shards(embeddings_dir: Path, datasets: tuple[str, ...]) -> list[Path]:
    shards: list[Path] = []
    for dataset in datasets:
        dataset_dir = embeddings_dir / dataset
        if not dataset_dir.is_dir():
            raise FileNotFoundError(dataset_dir)
        shards.extend(path.parent for path in dataset_dir.rglob("shard_meta.json"))
    shards.sort(key=lambda path: str(path.relative_to(embeddings_dir)))
    if not shards:
        raise RuntimeError("No corpus embedding shards were found")
    return shards

class TextResolver:
    def __init__(self, embeddings_dir: Path, datasets_dir: Path):
        self.embeddings_dir = embeddings_dir
        self.datasets_dir = datasets_dir
        self.metadata_cache: dict[str, np.ndarray] = {}
        self.dataset_cache: dict[str, Any] = {}

    def resolve(self, item: dict[str, Any]) -> dict[str, Any]:
        relative = _paper_path(item["shard"])
        parts = relative.parts
        dataset_name = parts[0]
        config_part = next((part for part in parts if part.startswith("config=")), None)
        split_part = next(part for part in parts if part.startswith("split="))
        split_name = split_part.split("=", 1)[1]
        input_path = self.datasets_dir / dataset_name
        if config_part is not None:
            input_path = input_path / config_part
        cache_key = str(input_path)
        if cache_key not in self.dataset_cache:
            self.dataset_cache[cache_key] = load_from_disk(str(input_path))
        loaded = self.dataset_cache[cache_key]
        dataset = loaded[split_name] if isinstance(loaded, DatasetDict) else loaded

        shard_dir = self.embeddings_dir / relative
        meta_key = str(shard_dir / "metadata.parquet")
        if meta_key not in self.metadata_cache:
            table = pq.read_table(meta_key, columns=["row_idx"])
            self.metadata_cache[meta_key] = table.column("row_idx").to_numpy()
        source_row = int(self.metadata_cache[meta_key][int(item["shard_row"])])
        shard_meta = json.loads((shard_dir / "shard_meta.json").read_text())
        text_column = str(shard_meta["text_column"])
        text = dataset[source_row][text_column]
        if text is None or str(text).strip() == "":
            text = "[EMPTY]"
        text = str(text)[:10000].strip()
        return {
            **item,
            "dataset": dataset_name,
            "config": config_part.split("=", 1)[1] if config_part else None,
            "split": split_name,
            "text_column": text_column,
            "source_row": source_row,
            "text": text,
        }

class SparseTopKValidator:
    """Validate raw-score candidates against the full per-row TopK encoder."""

    def __init__(
        self,
        models_dir: Path,
        embeddings_dir: Path,
        device: torch.device,
        batch_size: int,
    ):
        self.models_dir = models_dir
        self.embeddings_dir = embeddings_dir
        self.device = device
        self.batch_size = batch_size
        self.model_cache: dict[int, tuple[torch.Tensor, torch.Tensor, int]] = {}
        self.embedding_cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def _model(self, width: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        if width not in self.model_cache:
            if width == 4096:
                path, top_k = self.models_dir / "gemini_m4096_k32", 32
            elif width == 16384:
                path, top_k = self.models_dir / "gemini_m16384_k64", 64
            elif width == 131072:
                path, top_k = self.models_dir / "gemini_m131072_k128", 128
            else:
                raise ValueError(f"Unsupported SAE width {width}")
            state = load_state(path)
            weights = state["W_enc"].detach().to(self.device, torch.float32)
            bias = state["b_dec"].detach().to(self.device, torch.float32)
            self.model_cache[width] = (weights, bias, top_k)
        return self.model_cache[width]

    def _embedding(self, item: dict[str, Any]) -> np.ndarray:
        shard = str(item["shard"])
        if shard not in self.embedding_cache:
            path = self.embeddings_dir / shard / "embeddings.npy"
            self.embedding_cache[shard] = np.load(_paper_location(path), mmap_mode="r")
            if len(self.embedding_cache) > 32:
                self.embedding_cache.popitem(last=False)
        else:
            self.embedding_cache.move_to_end(shard)
        return np.asarray(
            self.embedding_cache[shard][int(item["shard_row"])], dtype=np.float32
        )

    @torch.inference_mode()
    def validate(
        self, candidates: list[dict[str, Any]], feature: dict[str, Any]
    ) -> list[dict[str, Any]]:
        weights, bias, top_k = self._model(int(feature["width"]))
        feature_id = int(feature["feature_id"])
        results: list[dict[str, Any]] = []
        for start in range(0, len(candidates), self.batch_size):
            batch_items = candidates[start : start + self.batch_size]
            embeddings = np.stack([self._embedding(item) for item in batch_items])
            x = torch.from_numpy(embeddings).to(self.device, torch.float32)
            pre_activations = torch.relu((x - bias) @ weights)
            selected_values = pre_activations[:, feature_id]
            selected_indices = torch.topk(
                pre_activations, k=top_k, dim=1, sorted=False
            ).indices
            active = (selected_indices == feature_id).any(dim=1)
            encoder_rank = (
                (pre_activations > selected_values.unsqueeze(1)).sum(dim=1) + 1
            )
            for item, is_active, value, rank in zip(
                batch_items,
                active.cpu().tolist(),
                selected_values.cpu().tolist(),
                encoder_rank.cpu().tolist(),
            ):
                scan_value = float(item["activation"])
                if abs(scan_value - float(value)) > 2e-4:
                    raise ValueError(
                        f"Candidate activation changed from {scan_value} to {value}"
                    )
                results.append(
                    {
                        **item,
                        "activation": float(value) if is_active else 0.0,
                        "pre_activation": float(value),
                        "encoder_rank": int(rank),
                        "survives_topk": bool(is_active),
                        "sae_top_k": top_k,
                    }
                )
        return results
