"""Read and validate cached sparse activation shard payloads."""

from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import json

import re

import sys

from pathlib import Path

from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

sys.path.insert(0, str(_paper_path(__file__).resolve().parents[1]))

if TYPE_CHECKING:
    from scripts.corpus_io import SparseTopKValidator

SCHEMA_VERSION = 1

CANDIDATE_ARTIFACT_KIND = "feature_family_top_candidates"

REPORT_ARTIFACT_KIND = "feature_family_top_examples"

ACTIVATION_FORMULA = "TopK_k(relu((embedding - b_dec) @ W_enc))"

DEFAULT_CACHE_ROOT = _paper_path(
    "/resources/activation_cache_dir/"
    "gemini_all_corpus_post_topk"
)

MODEL_SPECS = {
    "gemini_m4096_k32": {"width": 4096, "top_k": 32},
    "gemini_m131072_k128": {"width": 131072, "top_k": 128},
}

WIDTH_TO_MODEL = {int(spec["width"]): name for name, spec in MODEL_SPECS.items()}

FAMILY_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")

RESERVED_FAMILY_FIELDS = frozenset(
    ("target_feature_key", "atom_feature_keys", "feature_keys")
)

def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload

def _component_metadata(
    output: Mapping[str, Any], component: str, label: str
) -> Mapping[str, Any]:
    components = output.get("components")
    if not isinstance(components, Mapping):
        raise ValueError(f"{label} has no components mapping")
    metadata = components.get(component)
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{label} has invalid {component} metadata")
    return metadata

def _load_shard_marker(
    cache_root: Path,
    manifest: Mapping[str, Any],
    record: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    relative_shard = str(record["relative_shard"])
    shard_dir = cache_root / "shards" / relative_shard
    marker = read_json(shard_dir / "complete.json", "shard completion marker")
    source = marker.get("source")
    marker_relative = (
        source.get("relative_shard") if isinstance(source, Mapping) else None
    )
    if (
        marker.get("complete") is not True
        or marker.get("spec_id") != manifest.get("spec_id")
        or marker_relative != relative_shard
        or not isinstance(source, Mapping)
        or int(source.get("rows", -1)) != int(record["rows"])
    ):
        raise ValueError(f"Invalid completion marker for {relative_shard}")
    outputs = marker.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError(f"Completion marker has no outputs: {relative_shard}")
    if outputs != record.get("outputs"):
        raise ValueError(f"Manifest and completion outputs disagree: {relative_shard}")
    return shard_dir, marker

def _open_fixed_topk_payload(
    shard_dir: Path,
    record: Mapping[str, Any],
    marker: Mapping[str, Any],
    model_name: str,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
    relative_shard = str(record["relative_shard"])
    rows = int(record["rows"])
    width = int(MODEL_SPECS[model_name]["width"])
    top_k = int(MODEL_SPECS[model_name]["top_k"])
    output = marker["outputs"].get(model_name)
    if not isinstance(output, Mapping):
        raise ValueError(f"Missing output {relative_shard}/{model_name}")
    if output.get("shape") != [rows, width] or output.get("top_k") != top_k:
        raise ValueError(f"Wrong shape or top_k for {relative_shard}/{model_name}")

    model_dir = shard_dir / model_name
    expected = {
        "indptr": ((rows + 1,), np.dtype(np.int32)),
        "indices": ((rows * top_k,), np.dtype(np.int32)),
        "data": ((rows * top_k,), np.dtype(np.float32)),
    }
    arrays: dict[str, np.ndarray] = {}
    for component, (shape, dtype) in expected.items():
        path = model_dir / f"{component}.npy"
        try:
            array = np.load(_paper_location(path), mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot load {path}: {exc}") from exc
        metadata = _component_metadata(
            output, component, f"{relative_shard}/{model_name}"
        )
        if array.shape != shape or array.dtype != dtype:
            raise ValueError(
                f"Wrong {component} array for {relative_shard}/{model_name}: "
                f"shape={array.shape}, dtype={array.dtype}"
            )
        if metadata.get("shape") != list(shape) or metadata.get("dtype") != dtype.name:
            raise ValueError(
                f"Wrong {component} metadata for {relative_shard}/{model_name}"
            )
        declared_file = metadata.get("file")
        if declared_file is not None and _paper_path(str(declared_file)).name != path.name:
            raise ValueError(
                f"Wrong {component} filename for {relative_shard}/{model_name}"
            )
        if (
            "size_bytes" in metadata
            and int(metadata["size_bytes"]) != path.stat().st_size
        ):
            raise ValueError(
                f"Wrong {component} size for {relative_shard}/{model_name}"
            )
        arrays[component] = array

    expected_indptr = np.arange(rows + 1, dtype=np.int32) * top_k
    if not np.array_equal(arrays["indptr"], expected_indptr):
        raise ValueError(f"Non-fixed indptr for {relative_shard}/{model_name}")
    return arrays["indices"], arrays["data"], output

CSV_FIELDS = [
    "family_id",
    "semantic_label",
    "role",
    "interpretation",
    "atom_order",
    "coefficient",
    "feature_key",
    "model_name",
    "width",
    "feature_id",
    "rank",
    "activation",
    "cache_activation",
    "recomputed_activation",
    "activation_delta",
    "pre_activation",
    "encoder_rank",
    "survives_topk",
    "sae_top_k",
    "dataset",
    "config",
    "split",
    "text_column",
    "source_row",
    "shard",
    "shard_row",
    "global_row",
    "full_text_characters",
    "text_truncated",
    "normalized_text_sha256",
    "text",
]
