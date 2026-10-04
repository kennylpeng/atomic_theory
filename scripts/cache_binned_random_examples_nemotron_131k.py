#!/usr/bin/env python3
"""Nemotron 131K configuration for audited binned random activation samples."""

from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import json
import sys
from pathlib import Path

sys.path.insert(0, str(_paper_path(__file__).resolve().parents[1]))
from scripts import cache_payload as feature_reader  # noqa: E402
from scripts import sparse_activation_cache as sparse_reader  # noqa: E402

MODEL_NAME = "nemotron_m131072_k128"
SOURCE_ROOT = _paper_path(
    "/resources/activation_cache_dir/"
    "nemotron_m131072_post_topk_view_v1"
)
OUTPUT_ROOT = _paper_path(
    "/resources/activation_cache_dir/"
    "nemotron_m131072_binned_random_examples_v1"
)
MODEL_SPEC = {"width": 131072, "top_k": 128}
MODEL_SALT = 0x4E454D4F131072

feature_reader.MODEL_SPECS = {MODEL_NAME: MODEL_SPEC}
feature_reader.WIDTH_TO_MODEL = {131072: MODEL_NAME}
sparse_reader.SUPPORTED_MODELS = frozenset((MODEL_NAME,))
from scripts import cache_binned_random_examples as core  # noqa: E402


def load_target_shard_marker(cache_root, manifest, record):
    """Validate the target-model view against a full multi-model marker."""

    relative = str(record["relative_shard"])
    shard_dir = cache_root / "shards" / relative
    marker = feature_reader.read_json(
        shard_dir / "complete.json", "shard completion marker"
    )
    source = marker.get("source")
    marker_output = marker.get("outputs", {}).get(MODEL_NAME)
    record_output = record.get("outputs", {}).get(MODEL_NAME)
    if (
        marker.get("complete") is not True
        or marker.get("spec_id") != manifest.get("spec_id")
        or not isinstance(source, dict)
        or source.get("relative_shard") != relative
        or int(source.get("rows", -1)) != int(record["rows"])
        or marker_output != record_output
    ):
        raise ValueError(f"Invalid target-model completion marker: {relative}")
    return shard_dir, marker


core._load_shard_marker = load_target_shard_marker


def source_positive_slots() -> int:
    path = SOURCE_ROOT / "manifest.json"
    if not path.is_file():
        return 0
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return sum(
        int(shard["outputs"][MODEL_NAME]["positive_nnz"])
        for shard in manifest["shards"]
    )


core.ARTIFACT_KIND = "nemotron_post_topk_binned_random_examples"
core.DEFAULT_CACHE_ROOT = SOURCE_ROOT
core.DEFAULT_OUTPUT_ROOT = OUTPUT_ROOT
core.MODEL_SPECS = {
    MODEL_NAME: {
        **MODEL_SPEC,
        "salt": MODEL_SALT,
        "expected_positive_slots": source_positive_slots(),
    }
}


if __name__ == "__main__":
    core.main()
