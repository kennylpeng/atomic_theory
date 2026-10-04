#!/usr/bin/env python3
"""Cross-score sampled Gemini feature-3290 texts against its largest atoms.

This is a focused, read-only analysis helper.  It reads the published binned
sample cache and full post-TopK sparse activation cache, then prints one JSON
record per sampled target text.  No model inference or heuristic text labeling
is involved: every reported cross-activation is the cached float32 value.
"""
from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import argparse
import bisect
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(_paper_path(__file__).resolve().parents[1]))
from scripts.cache_binned_random_examples import BIN_UPPER_BOUNDS
from scripts.sample_io import load_numeric_samples, load_texts


DEFAULT_RESULT_DIR = _paper_path(
    "full_experiments/results/top_feature_examples/"
    "gemini_target_3290_expanded_17atom_sse_0p05"
)
DEFAULT_NUMERIC_CACHE = _paper_path(
    "/resources/activation_cache_dir/"
    "gemini_binned_random_examples_v1/final"
)
DEFAULT_TEXT_CATALOG = DEFAULT_NUMERIC_CACHE.parent / "text_catalog" / "final"
DEFAULT_SPARSE_CACHE = _paper_path(
    "/resources/activation_cache_dir/"
    "gemini_all_corpus_post_topk"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--numeric-cache", type=Path, default=DEFAULT_NUMERIC_CACHE)
    parser.add_argument("--text-catalog", type=Path, default=DEFAULT_TEXT_CATALOG)
    parser.add_argument("--sparse-cache", type=Path, default=DEFAULT_SPARSE_CACHE)
    parser.add_argument(
        "--top-definition",
        choices=("omp", "magnitude"),
        default="omp",
        help="Use the first five OMP atoms (default) or five largest |coefficient|.",
    )
    return parser.parse_args()


def load_family(
    result_dir: Path, top_definition: str
) -> tuple[dict[str, Any], list[int], dict[int, float]]:
    document = json.loads((result_dir / "family_spec.json").read_text(encoding="utf-8"))
    family = document["families"][0]
    atoms = [int(value) for value in family["atom_features"]]
    coefficients = [float(value) for value in family["coefficients"]]
    if top_definition == "omp":
        selected = atoms[:5]
    else:
        # Magnitude matches circle area in the molecule plot.
        selected = [
            atom
            for atom, _ in sorted(
                zip(atoms, coefficients), key=lambda item: -abs(item[1])
            )[:5]
        ]
    return family, selected, dict(zip(atoms, coefficients))


def cross_activations(
    sparse_cache: Path, rows: list[int], atom_ids: list[int]
) -> dict[int, dict[int, float]]:
    manifest = json.loads((sparse_cache / "manifest.json").read_text(encoding="utf-8"))
    shard_records = list(manifest["shards"])
    starts = [int(record["global_row_start"]) for record in shard_records]
    by_shard: dict[int, list[int]] = {}
    for row in rows:
        shard_index = bisect.bisect_right(starts, row) - 1
        if shard_index < 0 or row >= int(shard_records[shard_index]["global_row_stop"]):
            raise ValueError(f"Global row {row} is outside the sparse cache")
        by_shard.setdefault(shard_index, []).append(row)

    atom_set = set(atom_ids)
    result: dict[int, dict[int, float]] = {}
    for shard_index, selected_rows in by_shard.items():
        record = shard_records[shard_index]
        shard_dir = sparse_cache / "shards" / str(record["relative_shard"])
        model_dir = shard_dir / "gemini_m131072_k128"
        indices = np.load(_paper_location(model_dir / "indices.npy"), mmap_mode="r", allow_pickle=False)
        values = np.load(_paper_location(model_dir / "data.npy"), mmap_mode="r", allow_pickle=False)
        top_k = 128
        start = int(record["global_row_start"])
        for row in selected_rows:
            local_row = row - start
            begin, end = local_row * top_k, (local_row + 1) * top_k
            row_indices = np.asarray(indices[begin:end])
            row_values = np.asarray(values[begin:end])
            active = {
                int(feature): float(value)
                for feature, value in zip(row_indices, row_values)
                if int(feature) in atom_set and float(value) > 0
            }
            result[row] = {atom: active.get(atom, 0.0) for atom in atom_ids}
    return result


def conditional_histogram_percentile(
    value: float, counts: np.ndarray, sampled_maximum: float
) -> float:
    """Piecewise-linear CDF proxy conditional on a positive activation.

    Counts per bin are exact.  Interpolation within a bin is only a transparent
    tie-breaker because the cache intentionally stores samples, not full CDFs.
    """

    if value <= 0:
        return 0.0
    total = float(np.sum(counts))
    bin_index = int(np.searchsorted(BIN_UPPER_BOUNDS, np.float32(value), side="left"))
    below = float(np.sum(counts[:bin_index]))
    lower = 0.0 if bin_index == 0 else float(BIN_UPPER_BOUNDS[bin_index - 1])
    if bin_index < len(BIN_UPPER_BOUNDS):
        upper = float(BIN_UPPER_BOUNDS[bin_index])
    else:
        upper = max(sampled_maximum, lower + 1e-12)
    fraction = min(1.0, max(0.0, (value - lower) / (upper - lower)))
    return (below + fraction * float(counts[bin_index])) / total


def compact(value: str, limit: int = 240) -> str:
    normalized = " ".join(str(value).split())
    return normalized if len(normalized) <= limit else normalized[: limit - 1].rstrip() + "…"


def main() -> None:
    args = parse_args()
    family, atom_ids, coefficient_by_atom = load_family(
        args.result_dir, args.top_definition
    )
    numeric_manifest, samples = load_numeric_samples(args.numeric_cache, [family])
    target_key = ("gemini_m4096_k32", int(family["target_feature"]))
    target_entries = samples[target_key]
    target_rows = [row for _, row in target_entries]
    _, texts = load_texts(
        args.text_catalog, args.numeric_cache / "manifest.json", set(target_rows)
    )
    activations = cross_activations(args.sparse_cache, target_rows, atom_ids)

    model_dir = args.numeric_cache / "gemini_m131072_k128"
    bin_counts = np.load(_paper_location(model_dir / "bin_counts.npy"), mmap_mode="r", allow_pickle=False)
    sample_maxima = {
        atom: max(value for value, _ in samples[("gemini_m131072_k128", atom)])
        for atom in atom_ids
    }

    header = {
        "target_feature": int(family["target_feature"]),
        "top_atom_definition": (
            "first five atoms on the greedy OMP path"
            if args.top_definition == "omp"
            else "five largest absolute decoder-factorization coefficients"
        ),
        "atoms": [
            {
                "feature": atom,
                "coefficient": coefficient_by_atom[atom],
                "sample_maximum": sample_maxima[atom],
                "positive_count": int(np.sum(bin_counts[atom])),
            }
            for atom in atom_ids
        ],
        "winner_definitions": {
            "raw": "argmax_j a_j(x)",
            "histogram_percentile": (
                "argmax_j approximate conditional-positive CDF_j(a_j(x)); exact bin "
                "counts with piecewise-linear within-bin tie breaking"
            ),
            "molecule_magnitude": "argmax_j |c_j| a_j(x)",
        },
        "corpus_rows": int(numeric_manifest["corpus"]["rows"]),
    }
    print(json.dumps({"metadata": header}, ensure_ascii=False))
    for target_activation, row in target_entries:
        values = activations[row]
        percentile_scores = {
            atom: conditional_histogram_percentile(
                values[atom], np.asarray(bin_counts[atom]), sample_maxima[atom]
            )
            for atom in atom_ids
        }
        contribution_scores = {
            atom: abs(coefficient_by_atom[atom]) * values[atom] for atom in atom_ids
        }
        raw_winner = max(atom_ids, key=lambda atom: (values[atom], -atom))
        percentile_winner = max(
            atom_ids, key=lambda atom: (percentile_scores[atom], -atom)
        )
        contribution_winner = max(
            atom_ids, key=lambda atom: (contribution_scores[atom], -atom)
        )
        record = {
            "global_row": row,
            "target_activation": target_activation,
            "atom_activations": values,
            "histogram_percentiles": percentile_scores,
            "molecule_magnitudes": contribution_scores,
            "raw_winner": raw_winner if values[raw_winner] > 0 else None,
            "histogram_percentile_winner": (
                percentile_winner if percentile_scores[percentile_winner] > 0 else None
            ),
            "molecule_magnitude_winner": (
                contribution_winner if contribution_scores[contribution_winner] > 0 else None
            ),
            "text": compact(texts[row]),
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"Non-finite activation for row {row}")
        print(json.dumps(record, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
