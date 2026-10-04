"""Read frozen activation samples and their resolved texts."""

from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import bisect

from pathlib import Path

from typing import Any, Mapping

import numpy as np

import pyarrow.parquet as pq

from scripts.cache_binned_random_examples import (
    BIN_LABELS,
    BIN_UPPER_BOUNDS,
    MODEL_SPECS,
    read_json,
    sha256_file,
    validate_and_open_arrays,
)

DEFAULT_FAMILY_SPEC = _paper_path(
    "full_experiments/results/top_feature_examples/"
    "gemini_10_additional_families/family_spec.json"
)

DEFAULT_REPORT_ROOT = _paper_path(
    "full_experiments/results/top_feature_examples/"
    "gemini_10_additional_families/reports"
)

DEFAULT_NUMERIC_CACHE = _paper_path(
    "/resources/activation_cache_dir/"
    "gemini_binned_random_examples_v1/final"
)

DEFAULT_TEXT_CATALOG = DEFAULT_NUMERIC_CACHE.parent / "text_catalog" / "final"

OUTPUT_FILENAME = "binned_random_samples_readable.txt"

def load_numeric_samples(
    numeric_cache: Path, families: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[tuple[str, int], list[tuple[float, int]]]]:
    manifest_path = numeric_cache / "manifest.json"
    manifest = read_json(manifest_path, "numeric cache manifest")
    complete = read_json(numeric_cache / "COMPLETE.json", "numeric completion marker")
    if manifest.get("complete") is not True:
        raise ValueError("Numeric sample cache is incomplete")
    if complete.get("sampling_spec_sha256") != manifest.get("sampling_spec_sha256"):
        raise ValueError("Numeric completion marker disagrees with its manifest")
    if manifest.get("bin_labels") != list(BIN_LABELS):
        raise ValueError("Numeric cache bin labels do not match the renderer")

    requested: dict[str, set[int]] = {name: set() for name in MODEL_SPECS}
    for family in families:
        requested["gemini_m4096_k32"].add(int(family["target_feature"]))
        requested["gemini_m131072_k128"].update(
            int(value) for value in family["atom_features"]
        )

    result: dict[tuple[str, int], list[tuple[float, int]]] = {}
    sample_count = int(manifest["sample_count_per_feature_bin"])
    for model_name, feature_ids in requested.items():
        spec = MODEL_SPECS[model_name]
        model_result = manifest.get("model_results", {}).get(model_name)
        if not isinstance(model_result, Mapping):
            raise ValueError(f"Numeric cache is missing {model_name}")
        arrays = validate_and_open_arrays(
            numeric_cache / model_name,
            model_result["arrays"],
            int(spec["width"]),
            sample_count,
            verify_hash=True,
        )
        sizes = arrays["sample_sizes"]
        rows = arrays["sample_rows"]
        activations = arrays["sample_activations"]
        for feature in sorted(feature_ids):
            examples: list[tuple[float, int]] = []
            for bin_index in range(len(BIN_LABELS)):
                size = int(sizes[feature, bin_index])
                for slot in range(size):
                    row = int(rows[feature, bin_index, slot])
                    activation = float(activations[feature, bin_index, slot])
                    assigned = int(
                        np.searchsorted(
                            BIN_UPPER_BOUNDS, np.float32(activation), side="left"
                        )
                    )
                    if row < 0 or not np.isfinite(activation) or activation <= 0:
                        raise ValueError(f"Invalid sample for {model_name}/{feature}")
                    if assigned != bin_index:
                        raise ValueError(
                            f"Sample in wrong bin for {model_name}/{feature}: {activation}"
                        )
                    examples.append((activation, row))
            if len({row for _, row in examples}) != len(examples):
                raise ValueError(f"Duplicate sampled row for {model_name}/{feature}")
            examples.sort(key=lambda item: (-item[0], item[1]))
            result[(model_name, feature)] = examples
    return manifest, result

def _row_group_range(parquet: pq.ParquetFile, row_group: int) -> tuple[int, int]:
    column_index = parquet.schema_arrow.get_field_index("global_row")
    metadata = parquet.metadata.row_group(row_group).column(column_index)
    statistics = metadata.statistics
    if statistics is None or not statistics.has_min_max:
        return -(1 << 63), (1 << 63) - 1
    return int(statistics.min), int(statistics.max)

def load_texts(
    text_catalog: Path,
    numeric_manifest_path: Path,
    requested_rows: set[int],
) -> tuple[dict[str, Any], dict[int, str]]:
    manifest = read_json(text_catalog / "manifest.json", "text catalog manifest")
    complete = read_json(text_catalog / "COMPLETE.json", "text completion marker")
    if manifest.get("complete") is not True or complete.get("complete") is not True:
        raise ValueError("Text catalog is incomplete")
    if complete.get("catalog_spec_sha256") != manifest.get("catalog_spec_sha256"):
        raise ValueError("Text completion marker disagrees with its manifest")
    if manifest.get("numeric_manifest_sha256") != sha256_file(numeric_manifest_path):
        raise ValueError("Text catalog is bound to a different numeric manifest")

    sorted_rows = sorted(requested_rows)
    found: dict[int, str] = {}
    for part in manifest.get("parts", []):
        if not isinstance(part, Mapping):
            raise ValueError("Invalid text-catalog part metadata")
        lower = int(part["global_row_min"])
        upper = int(part["global_row_max"])
        left = bisect.bisect_left(sorted_rows, lower)
        right = bisect.bisect_right(sorted_rows, upper)
        part_targets = sorted_rows[left:right]
        if not part_targets:
            continue
        path = text_catalog / str(part["file"])
        if not path.is_file() or path.stat().st_size != int(part["size_bytes"]):
            raise ValueError(f"Missing or wrong-sized text-catalog part: {path}")
        parquet = pq.ParquetFile(path)
        for row_group in range(parquet.metadata.num_row_groups):
            group_lower, group_upper = _row_group_range(parquet, row_group)
            group_left = bisect.bisect_left(part_targets, group_lower)
            group_right = bisect.bisect_right(part_targets, group_upper)
            group_targets = part_targets[group_left:group_right]
            if not group_targets:
                continue
            table = parquet.read_row_group(
                row_group, columns=["global_row", "full_text"]
            )
            rows = table.column("global_row").to_numpy(zero_copy_only=False)
            texts = table.column("full_text").to_pylist()
            for target in group_targets:
                position = int(np.searchsorted(rows, target))
                if position >= len(rows) or int(rows[position]) != target:
                    raise ValueError(f"Missing global row {target} in declared row group")
                if target in found:
                    raise ValueError(f"Duplicate global row in text catalog: {target}")
                found[target] = str(texts[position])
    missing = requested_rows - found.keys()
    if missing:
        preview = sorted(missing)[:10]
        raise ValueError(f"Text catalog is missing {len(missing)} rows, e.g. {preview}")
    return manifest, found
