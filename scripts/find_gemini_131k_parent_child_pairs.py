#!/usr/bin/env python
"""Find directional feature-subsumption pairs from a co-occurrence matrix."""

from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 1
METRIC_COLUMNS = (
    "parent_feature",
    "child_feature",
    "parent_support",
    "child_support",
    "cooccurrence_count",
    "child_to_parent_containment",
    "child_to_parent_wilson_lower_95",
    "reverse_containment",
    "lift",
    "directional_subsumption_score",
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def wilson_lower(successes: np.ndarray, trials: np.ndarray, z: float = 1.96) -> np.ndarray:
    successes = np.asarray(successes, dtype=np.float64)
    trials = np.asarray(trials, dtype=np.float64)
    proportion = successes / trials
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    center = proportion + z_squared / (2.0 * trials)
    margin = z * np.sqrt(
        proportion * (1.0 - proportion) / trials
        + z_squared / (4.0 * trials * trials)
    )
    return (center - margin) / denominator


def minimum_successes_for_wilson(
    supports: np.ndarray, threshold: float, minimum_support: int
) -> np.ndarray:
    """Return the least overlap whose Wilson lower bound reaches threshold."""

    supports = np.asarray(supports, dtype=np.int64)
    low = np.zeros_like(supports)
    high = supports.copy()
    live = supports >= minimum_support
    for _ in range(32):
        unresolved = live & (low < high)
        if not np.any(unresolved):
            break
        middle = (low + high) // 2
        indices = np.flatnonzero(unresolved)
        accepted = wilson_lower(middle[indices], supports[indices]) >= threshold
        high[indices[accepted]] = middle[indices[accepted]]
        low[indices[~accepted]] = middle[indices[~accepted]] + 1
    result = low
    result[~live] = np.iinfo(np.int64).max
    return result


def load_supports(matrix_manifest: dict[str, Any]) -> np.ndarray:
    width = int(matrix_manifest["width"])
    supports = np.empty(width, dtype=np.uint32)
    for block in matrix_manifest["blocks"]:
        start, stop = int(block["row_start"]), int(block["row_stop"])
        matrix = np.load(_paper_location(block["path"]), mmap_mode="r", allow_pickle=False)
        local = np.arange(stop - start, dtype=np.int64)
        supports[start:stop] = matrix[local, local + start]
    return supports


def prepare(args: argparse.Namespace) -> None:
    matrix_root = args.matrix_root.resolve()
    output_root = args.output_root.resolve()
    matrix_manifest = read_json(matrix_root / "manifest.json")
    if matrix_manifest.get("complete") is not True:
        raise ValueError(f"Matrix artifact is incomplete: {matrix_root}")
    if float(matrix_manifest["activation_threshold"]) != args.activation_threshold:
        raise ValueError(
            "Matrix activation threshold differs from --activation-threshold: "
            f"{matrix_manifest['activation_threshold']}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "partials").mkdir(exist_ok=True)
    supports = load_supports(matrix_manifest)
    support_path = output_root / "supports.npy"
    np.save(_paper_location(support_path), supports, allow_pickle=False)
    wilson_required = minimum_successes_for_wilson(
        supports, args.r_lower_threshold, args.minimum_child_support
    )
    wilson_required = np.maximum(wilson_required, args.minimum_overlap)
    wilson_path = output_root / "wilson_required_overlap.npy"
    np.save(_paper_location(wilson_path), wilson_required, allow_pickle=False)
    plan = {
        "schema_version": SCHEMA_VERSION,
        "complete": False,
        "matrix_root": str(matrix_root),
        "matrix_manifest": str((matrix_root / "manifest.json").resolve()),
        "matrix_plan_fingerprint": matrix_manifest["plan_fingerprint"],
        "activation_threshold": args.activation_threshold,
        "rows": int(matrix_manifest["source_rows"]),
        "width": int(matrix_manifest["width"]),
        "num_blocks": int(matrix_manifest["num_blocks"]),
        "supports": str(support_path),
        "wilson_required_overlap": str(wilson_path),
        "metric": (
            "max(0, (wilson_lower_95(P(parent|child)) - P(parent)) / "
            "(1 - P(parent))) * (1 - P(child|parent))"
        ),
        "filters": {
            "minimum_child_support": args.minimum_child_support,
            "minimum_overlap": args.minimum_overlap,
            "r_lower_threshold": args.r_lower_threshold,
            "maximum_reverse_containment": args.maximum_reverse_containment,
            "minimum_lift": args.minimum_lift,
            "strict_parent_support_greater_than_child": True,
        },
        "blocks": matrix_manifest["blocks"],
    }
    plan_path = output_root / "plan.json"
    if plan_path.exists():
        existing = read_json(plan_path)
        if existing != plan:
            raise ValueError(f"Existing plan differs from requested plan: {plan_path}")
        print(f"Plan already exists and matches: {plan_path}")
        return
    atomic_json(plan_path, plan)
    print(
        f"Prepared parent-child scan over {plan['num_blocks']} blocks at {output_root}"
    )


def partial_path(output_root: Path, block_id: int) -> Path:
    return output_root / "partials" / f"candidates_block_{block_id:03d}.npz"


def partial_marker_path(output_root: Path, block_id: int) -> Path:
    return output_root / "partials" / f"candidates_block_{block_id:03d}.json"


def empty_result() -> dict[str, np.ndarray]:
    return {
        "parent_feature": np.empty(0, dtype=np.int32),
        "child_feature": np.empty(0, dtype=np.int32),
        "parent_support": np.empty(0, dtype=np.uint32),
        "child_support": np.empty(0, dtype=np.uint32),
        "cooccurrence_count": np.empty(0, dtype=np.uint32),
        "child_to_parent_containment": np.empty(0, dtype=np.float32),
        "child_to_parent_wilson_lower_95": np.empty(0, dtype=np.float32),
        "reverse_containment": np.empty(0, dtype=np.float32),
        "lift": np.empty(0, dtype=np.float32),
        "directional_subsumption_score": np.empty(0, dtype=np.float32),
    }


def scan_parent_row(
    parent: int,
    overlaps: np.ndarray,
    supports: np.ndarray,
    supports_int64: np.ndarray,
    wilson_required: np.ndarray,
    rows: int,
    maximum_reverse_containment: float,
    minimum_lift: float,
) -> dict[str, np.ndarray]:
    parent_support = int(supports[parent])
    if parent_support == 0:
        return empty_result()
    child_supports = supports_int64
    lift_required = np.ceil(
        minimum_lift * parent_support * child_supports.astype(np.float64) / rows
    ).astype(np.int64)
    required = np.maximum(wilson_required, lift_required)
    mask = overlaps.astype(np.int64, copy=False) >= required
    mask &= child_supports < parent_support
    mask &= overlaps.astype(np.float64, copy=False) <= (
        maximum_reverse_containment * parent_support
    )
    children = np.flatnonzero(mask)
    if children.size == 0:
        return empty_result()
    overlap = overlaps[children].astype(np.uint32, copy=False)
    child_support = supports[children]
    containment = overlap.astype(np.float64) / child_support
    lower = wilson_lower(overlap, child_support)
    reverse = overlap.astype(np.float64) / parent_support
    lift = containment * rows / parent_support
    baseline = parent_support / rows
    excess = np.maximum(0.0, (lower - baseline) / (1.0 - baseline))
    score = excess * (1.0 - reverse)
    count = children.size
    return {
        "parent_feature": np.full(count, parent, dtype=np.int32),
        "child_feature": children.astype(np.int32),
        "parent_support": np.full(count, parent_support, dtype=np.uint32),
        "child_support": child_support.astype(np.uint32, copy=False),
        "cooccurrence_count": overlap,
        "child_to_parent_containment": containment.astype(np.float32),
        "child_to_parent_wilson_lower_95": lower.astype(np.float32),
        "reverse_containment": reverse.astype(np.float32),
        "lift": lift.astype(np.float32),
        "directional_subsumption_score": score.astype(np.float32),
    }


def concatenate_results(results: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not results:
        return empty_result()
    return {name: np.concatenate([item[name] for item in results]) for name in METRIC_COLUMNS}


def compute(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    plan = read_json(output_root / "plan.json")
    block_id = args.block_id
    if block_id < 0 or block_id >= int(plan["num_blocks"]):
        raise ValueError(f"Invalid block_id: {block_id}")
    output = partial_path(output_root, block_id)
    marker_path = partial_marker_path(output_root, block_id)
    if output.is_file() and marker_path.is_file():
        marker = read_json(marker_path)
        if marker.get("complete") is True:
            print(f"Block {block_id} already complete: {output}")
            return
    block = plan["blocks"][block_id]
    matrix = np.load(_paper_location(block["path"]), mmap_mode="r", allow_pickle=False)
    supports = np.load(_paper_location(plan["supports"]), mmap_mode="r", allow_pickle=False)
    supports_int64 = np.asarray(supports, dtype=np.int64)
    wilson_required = np.load(
        _paper_location(plan["wilson_required_overlap"]), mmap_mode="r", allow_pickle=False
    )
    filters = plan["filters"]
    row_start, row_stop = int(block["row_start"]), int(block["row_stop"])
    results: list[dict[str, np.ndarray]] = []
    started = time.monotonic()
    for local_parent, parent in enumerate(range(row_start, row_stop)):
        result = scan_parent_row(
            parent,
            np.asarray(matrix[local_parent]),
            supports,
            supports_int64,
            wilson_required,
            int(plan["rows"]),
            float(filters["maximum_reverse_containment"]),
            float(filters["minimum_lift"]),
        )
        if result["parent_feature"].size:
            results.append(result)
        completed = local_parent + 1
        if completed % args.progress_every == 0 or parent + 1 == row_stop:
            count = sum(item["parent_feature"].size for item in results)
            print(
                f"Block {block_id}: {completed}/{row_stop - row_start} parent rows, "
                f"{count:,} candidates, {(time.monotonic() - started) / 60:.1f} min",
                flush=True,
            )
    combined = concatenate_results(results)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(_paper_location(handle), **combined)
    os.replace(temporary, output)
    atomic_json(
        marker_path,
        {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "block_id": block_id,
            "row_start": row_start,
            "row_stop": row_stop,
            "candidates": int(combined["parent_feature"].size),
            "elapsed_seconds": time.monotonic() - started,
            "output": str(output),
        },
    )
    print(f"Wrote {combined['parent_feature'].size:,} candidates to {output}")


def write_csv(path: Path, arrays: dict[str, np.ndarray], order: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(METRIC_COLUMNS)
        for index in order:
            writer.writerow([arrays[name][index].item() for name in METRIC_COLUMNS])
    os.replace(temporary, path)


def direct_edge_mask(parent: np.ndarray, child: np.ndarray, width: int) -> np.ndarray:
    """Remove P->C if a candidate M has both P->M and M->C."""

    count = parent.size
    direct = np.ones(count, dtype=bool)
    edge_keys = set((parent.astype(np.int64) * width + child).tolist())
    child_order = np.argsort(child, kind="stable")
    sorted_child = child[child_order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_child)) + 1, count]
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        group_indices = child_order[start:stop]
        group_parents = parent[group_indices]
        for index, candidate_parent in zip(group_indices, group_parents):
            for intermediate in group_parents:
                if intermediate == candidate_parent:
                    continue
                if int(candidate_parent) * width + int(intermediate) in edge_keys:
                    direct[index] = False
                    break
    return direct


def reduce(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    plan = read_json(output_root / "plan.json")
    markers = [
        read_json(partial_marker_path(output_root, block_id))
        for block_id in range(int(plan["num_blocks"]))
    ]
    if not all(marker.get("complete") is True for marker in markers):
        raise ValueError("At least one partial block is incomplete")
    partials = [
        np.load(_paper_location(partial_path(output_root, block_id)), allow_pickle=False)
        for block_id in range(int(plan["num_blocks"]))
    ]
    arrays = {
        name: np.concatenate([partial[name] for partial in partials])
        for name in METRIC_COLUMNS
    }
    candidate_count = arrays["parent_feature"].size
    rank_order = np.argsort(
        arrays["directional_subsumption_score"], kind="stable"
    )[::-1]
    candidate_npz = output_root / "all_qualifying_ancestor_pairs.npz"
    np.savez_compressed(_paper_location(candidate_npz), **arrays)
    candidate_csv = output_root / "all_qualifying_ancestor_pairs.csv"
    write_csv(candidate_csv, arrays, rank_order)

    direct = direct_edge_mask(
        arrays["parent_feature"], arrays["child_feature"], int(plan["width"])
    )
    direct_arrays = {name: values[direct] for name, values in arrays.items()}
    direct_order = np.argsort(
        direct_arrays["directional_subsumption_score"], kind="stable"
    )[::-1]
    direct_npz = output_root / "immediate_parent_child_pairs.npz"
    np.savez_compressed(_paper_location(direct_npz), **direct_arrays)
    direct_csv = output_root / "immediate_parent_child_pairs.csv"
    write_csv(direct_csv, direct_arrays, direct_order)

    summary = {
        **plan,
        "complete": True,
        "qualifying_ancestor_pairs": int(candidate_count),
        "immediate_parent_child_pairs": int(np.count_nonzero(direct)),
        "transitive_pairs_removed": int(np.count_nonzero(~direct)),
        "transitive_reduction": (
            "remove P->C when a qualifying intermediate M has both P->M and M->C"
        ),
        "outputs": {
            "all_qualifying_ancestor_pairs_npz": str(candidate_npz),
            "all_qualifying_ancestor_pairs_csv": str(candidate_csv),
            "immediate_parent_child_pairs_npz": str(direct_npz),
            "immediate_parent_child_pairs_csv": str(direct_csv),
        },
    }
    atomic_json(output_root / "summary.json", summary)
    atomic_json(
        output_root / "COMPLETE.json",
        {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "summary": str((output_root / "summary.json").resolve()),
            "qualifying_ancestor_pairs": int(candidate_count),
            "immediate_parent_child_pairs": int(np.count_nonzero(direct)),
        },
    )
    print(
        f"Wrote {candidate_count:,} qualifying ancestor pairs and "
        f"{np.count_nonzero(direct):,} immediate parent-child pairs"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--matrix-root", type=Path, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    prepare_parser.add_argument("--activation-threshold", type=float, default=0.05)
    prepare_parser.add_argument("--minimum-child-support", type=int, default=100)
    prepare_parser.add_argument("--minimum-overlap", type=int, default=100)
    prepare_parser.add_argument("--r-lower-threshold", type=float, default=0.7)
    prepare_parser.add_argument("--maximum-reverse-containment", type=float, default=0.5)
    prepare_parser.add_argument("--minimum-lift", type=float, default=5.0)
    prepare_parser.set_defaults(func=prepare)

    compute_parser = commands.add_parser("compute")
    compute_parser.add_argument("--output-root", type=Path, required=True)
    compute_parser.add_argument("--block-id", type=int, required=True)
    compute_parser.add_argument("--progress-every", type=int, default=256)
    compute_parser.set_defaults(func=compute)

    reduce_parser = commands.add_parser("reduce")
    reduce_parser.add_argument("--output-root", type=Path, required=True)
    reduce_parser.set_defaults(func=reduce)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    arguments.func(arguments)
