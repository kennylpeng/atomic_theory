"""Shared feature identities and paired activation scoring for example tables."""

from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import bisect

import json

from pathlib import Path

from typing import Any, Mapping

import numpy as np

from scripts.aligned_activations import GEMINI_MODEL, NEMOTRON_MODEL, FixedTopKShard

ROOT = _paper_path(__file__).resolve().parents[1]

MATCH_ROOT = ROOT / (
    "full_experiments/results/cross_model_activation_matches/"
    "gemini_nemotron_m131072"
)

PAIR_CSV = MATCH_ROOT / "relaxed_jaccard_full_distribution/per_pair.csv"

MATCH_SUMMARY = MATCH_ROOT / "relaxed_jaccard_full_distribution/summary.json"

OUTPUT_ROOT = MATCH_ROOT / "random_reciprocal_pair_latex_examples"

GEMINI_CACHE = _paper_path(
    "/resources/activation_cache_dir/"
    "gemini_all_corpus_post_topk"
)

NEMOTRON_CACHE = _paper_path(
    "/resources/activation_cache_dir/"
    "nemotron_all_experiment_saes_post_topk"
)

GEMINI_NUMERIC = _paper_path(
    "/resources/activation_cache_dir/"
    "gemini_binned_random_examples_v1/final"
)

NEMOTRON_NUMERIC = _paper_path(
    "/resources/activation_cache_dir/"
    "nemotron_m131072_binned_random_examples_v1/final"
)

GEMINI_TEXT = GEMINI_NUMERIC.parent / "text_catalog/final"

NEMOTRON_TEXT = NEMOTRON_NUMERIC.parent / "text_catalog/final"

HIGH_BIN_INDICES = (6, 5, 4, 3, 2)

PAIR_SEED = 20260902

WIDTH = 131072

def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value

def load_sample_arrays(root: Path, model: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest = read_json(root / "manifest.json")
    complete = read_json(root / "COMPLETE.json")
    if (
        manifest.get("complete") is not True
        or complete.get("complete") is not True
        or complete.get("sampling_spec_sha256")
        != manifest.get("sampling_spec_sha256")
    ):
        raise ValueError(f"Incomplete or inconsistent numeric cache: {root}")
    result = manifest.get("model_results", {}).get(model)
    if not isinstance(result, Mapping):
        raise ValueError(f"Missing {model} in {root}")
    arrays = {
        name: np.load(_paper_location(root / model / f"{name}.npy"), mmap_mode="r", allow_pickle=False)
        for name in ("sample_sizes", "sample_rows", "sample_activations", "sample_priorities")
    }
    expected_prefix = (WIDTH, len(manifest["bin_labels"]))
    if arrays["sample_sizes"].shape != expected_prefix:
        raise ValueError(f"Wrong sample-size shape in {root}")
    for name in ("sample_rows", "sample_activations", "sample_priorities"):
        if arrays[name].shape[:2] != expected_prefix:
            raise ValueError(f"Wrong {name} shape in {root}")
    return manifest, arrays

def attach_paired_activations(pairs: list[dict[str, Any]]) -> None:
    """Attach (Gemini, Nemotron) activations on each example's shared row."""
    plans = {
        "gemini": read_json(GEMINI_CACHE / "plan.json"),
        "nemotron": read_json(NEMOTRON_CACHE / "plan.json"),
    }
    records = {model: plan["source_shards"] for model, plan in plans.items()}
    starts = {
        model: [int(record["global_row_start"]) for record in model_records]
        for model, model_records in records.items()
    }
    by_name = {
        model: {record["relative_shard"]: record for record in model_records}
        for model, model_records in records.items()
    }
    requests: dict[str, list[tuple[dict[str, Any], int, int, str, int]]] = {}
    for pair in pairs:
        gemini_feature = int(pair["gemini_feature"])
        nemotron_feature = int(pair["nemotron_feature"])
        for source_model, examples_key, target_model in (
            ("gemini", "gemini_examples", "nemotron"),
            ("nemotron", "nemotron_examples", "gemini"),
        ):
            for example in pair[examples_key]:
                global_row = int(example["global_row"])
                record_index = bisect.bisect_right(starts[source_model], global_row) - 1
                if record_index < 0:
                    raise ValueError(f"Unmapped {source_model} row {global_row}")
                source_record = records[source_model][record_index]
                if not (int(source_record["global_row_start"]) <= global_row < int(source_record["global_row_stop"])):
                    raise ValueError(f"Unmapped {source_model} row {global_row}")
                relative_shard = source_record["relative_shard"]
                target_record = by_name[target_model].get(relative_shard)
                if target_record is None:
                    raise ValueError(f"Selected row has no {target_model} shard: {relative_shard}")
                local_row = global_row - int(source_record["global_row_start"])
                if local_row >= int(target_record["rows"]):
                    raise ValueError(f"Local-row mismatch for {relative_shard}")
                requests.setdefault(relative_shard, []).append(
                    (example, gemini_feature, nemotron_feature, source_model, local_row)
                )

    def activation_at(shard: FixedTopKShard, row: int, feature: int) -> float:
        indices = np.asarray(shard.indices[row], dtype=np.int64)
        positions = np.flatnonzero(indices == feature)
        if positions.size == 0:
            return 0.0
        if positions.size != 1:
            raise ValueError(f"Duplicate feature {feature} in TopK row {row}")
        return float(shard.data[row, int(positions[0])])

    for relative_shard, shard_requests in requests.items():
        gemini = FixedTopKShard(GEMINI_CACHE, relative_shard, GEMINI_MODEL, WIDTH)
        nemotron = FixedTopKShard(NEMOTRON_CACHE, relative_shard, NEMOTRON_MODEL, WIDTH)
        for example, gemini_feature, nemotron_feature, source_model, local_row in shard_requests:
            gemini_activation = activation_at(gemini, local_row, gemini_feature)
            nemotron_activation = activation_at(nemotron, local_row, nemotron_feature)
            source_activation = gemini_activation if source_model == "gemini" else nemotron_activation
            if not np.isclose(source_activation, float(example["activation"]), rtol=0.0, atol=5e-5):
                raise ValueError(
                    f"Source activation mismatch in {relative_shard} row {local_row}: "
                    f"cache={source_activation}, sample={example['activation']}"
                )
            example["gemini_activation"] = gemini_activation
            example["nemotron_activation"] = nemotron_activation
