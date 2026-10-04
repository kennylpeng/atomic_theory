#!/usr/bin/env python
"""Compute an exact block-sharded Gemini 131K feature co-occurrence matrix.

The input cache stores fixed-TopK post-activation rows.  For a threshold ``t``,
this script defines a feature as present when its cached activation is strictly
greater than ``t`` and computes

    C[i, j] = number of corpus rows where i and j are both present.

The output is row-blocked dense uint32 rather than one monolithic file.  This
keeps individual jobs and files manageable while preserving every matrix entry.
"""

from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


MODEL_NAME = "gemini_m131072_k128"
SCHEMA_VERSION = 1


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


def cache_model(plan: dict[str, Any], model_name: str) -> dict[str, Any]:
    matches = [model for model in plan["models"] if model.get("name") == model_name]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one model named {model_name!r}")
    return matches[0]


def output_block_path(output_root: Path, block_id: int, start: int, stop: int) -> Path:
    return output_root / "blocks" / f"block_{block_id:03d}_{start:06d}_{stop:06d}.npy"


def output_marker_path(output_root: Path, block_id: int) -> Path:
    return output_root / "block_metadata" / f"block_{block_id:03d}.json"


def plan_fingerprint(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def make_output_plan(
    cache_root: Path,
    output_root: Path,
    block_size: int,
    threshold: float,
) -> dict[str, Any]:
    source_plan = read_json(cache_root / "plan.json")
    source_complete = read_json(cache_root / "COMPLETE.json")
    if source_complete.get("complete") is not True:
        raise ValueError(f"Source cache is not complete: {cache_root}")
    model = cache_model(source_plan, MODEL_NAME)
    width = int(model["width"])
    top_k = int(model["top_k"])
    if width != 131072 or top_k != 128:
        raise ValueError(f"Unexpected model dimensions: width={width}, top_k={top_k}")
    if block_size <= 0 or block_size > width:
        raise ValueError("block_size must be in [1, width]")
    if threshold < 0:
        raise ValueError("threshold must be nonnegative")
    blocks = []
    for block_id, start in enumerate(range(0, width, block_size)):
        stop = min(start + block_size, width)
        blocks.append(
            {
                "block_id": block_id,
                "row_start": start,
                "row_stop": stop,
                "shape": [stop - start, width],
                "path": str(output_block_path(output_root, block_id, start, stop)),
            }
        )
    result = {
        "schema_version": SCHEMA_VERSION,
        "complete": False,
        "definition": (
            "C[i,j] is the number of source-cache rows for which cached post-TopK "
            "activations i and j are both strictly greater than activation_threshold"
        ),
        "model": MODEL_NAME,
        "width": width,
        "top_k": top_k,
        "activation_threshold": threshold,
        "comparison": "strictly_greater_than",
        "matrix_dtype": "uint32",
        "matrix_shape": [width, width],
        "storage": "dense row blocks in NumPy .npy format",
        "block_size": block_size,
        "num_blocks": len(blocks),
        "source_cache_root": str(cache_root.resolve()),
        "source_cache_spec_id": source_complete["spec_id"],
        "source_rows": int(source_complete["rows"]),
        "source_shards": int(source_complete["shards"]),
        "blocks": blocks,
    }
    result["plan_fingerprint"] = plan_fingerprint(result)
    return result


def prepare(args: argparse.Namespace) -> None:
    cache_root = args.cache_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "blocks").mkdir(exist_ok=True)
    (output_root / "block_metadata").mkdir(exist_ok=True)
    planned = make_output_plan(cache_root, output_root, args.block_size, args.threshold)
    plan_path = output_root / "plan.json"
    if plan_path.exists():
        existing = read_json(plan_path)
        if existing != planned:
            raise ValueError(
                f"Existing plan differs from requested computation: {plan_path}"
            )
        print(f"Plan already exists and matches: {plan_path}")
        return
    atomic_json(plan_path, planned)
    print(
        f"Prepared {planned['num_blocks']} blocks for a "
        f"{planned['matrix_shape'][0]}x{planned['matrix_shape'][1]} matrix at "
        f"{output_root}"
    )


class FixedTopKShard:
    def __init__(self, cache_root: Path, relative_shard: str, width: int, top_k: int):
        model_dir = cache_root / "shards" / relative_shard / MODEL_NAME
        complete = read_json(model_dir.parent / "complete.json")
        output = complete.get("outputs", {}).get(MODEL_NAME)
        if not isinstance(output, dict):
            raise ValueError(f"Missing completion metadata: {model_dir}")
        rows, found_width = (int(item) for item in output["shape"])
        if found_width != width or int(output["top_k"]) != top_k:
            raise ValueError(f"Unexpected model shape or TopK in {model_dir}")
        self.rows = rows
        self.indices = np.load(
            _paper_location(model_dir / "indices.npy"), mmap_mode="r", allow_pickle=False
        ).reshape(rows, top_k)
        self.values = np.load(
            _paper_location(model_dir / "data.npy"), mmap_mode="r", allow_pickle=False
        ).reshape(rows, top_k)


def accumulate_batch(
    counts: torch.Tensor,
    feature_ids: torch.Tensor,
    values: torch.Tensor,
    row_start: int,
    row_stop: int,
    threshold: float,
) -> int:
    """Accumulate one batch and return the number of pair increments."""

    width = counts.shape[1]
    active = values > threshold
    source_mask = active & (feature_ids >= row_start) & (feature_ids < row_stop)
    source_rows, source_slots = torch.nonzero(source_mask, as_tuple=True)
    if source_rows.numel() == 0:
        return 0
    source_local = feature_ids[source_rows, source_slots] - row_start
    target_ids = feature_ids[source_rows]
    target_active = active[source_rows]
    locations = source_local[:, None] * width + target_ids
    locations = locations[target_active]
    counts.view(-1).index_add_(
        0,
        locations,
        torch.ones(locations.numel(), dtype=counts.dtype, device=counts.device),
    )
    return locations.numel()


def save_tensor_npy(path: Path, tensor: torch.Tensor, row_chunk: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.npy")
    target = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.uint32, shape=tuple(tensor.shape)
    )
    for start in range(0, tensor.shape[0], row_chunk):
        stop = min(start + row_chunk, tensor.shape[0])
        target[start:stop] = tensor[start:stop].cpu().numpy()
    target.flush()
    del target
    os.replace(temporary, path)


@torch.no_grad()
def compute(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for block computation")
    output_root = args.output_root.resolve()
    plan = read_json(output_root / "plan.json")
    if args.block_id < 0 or args.block_id >= int(plan["num_blocks"]):
        raise ValueError(f"block_id must be in [0, {int(plan['num_blocks']) - 1}]")
    block = plan["blocks"][args.block_id]
    start, stop = int(block["row_start"]), int(block["row_stop"])
    block_path = _paper_path(block["path"])
    marker_path = output_marker_path(output_root, args.block_id)
    if marker_path.exists():
        marker = read_json(marker_path)
        if marker.get("complete") is True and block_path.is_file():
            print(f"Block {args.block_id} is already complete: {block_path}")
            return

    device = torch.device(args.device)
    width = int(plan["width"])
    top_k = int(plan["top_k"])
    counts = torch.zeros((stop - start, width), dtype=torch.int32, device=device)
    cache_root = _paper_path(plan["source_cache_root"])
    source_plan = read_json(cache_root / "plan.json")
    shards = source_plan["source_shards"]
    rows_done = 0
    increments = 0
    started = time.monotonic()
    for shard_number, shard_record in enumerate(shards, start=1):
        relative_shard = shard_record["relative_shard"]
        shard = FixedTopKShard(cache_root, relative_shard, width, top_k)
        for batch_start in range(0, shard.rows, args.batch_rows):
            batch_stop = min(batch_start + args.batch_rows, shard.rows)
            feature_ids = torch.from_numpy(
                np.asarray(shard.indices[batch_start:batch_stop], dtype=np.int64).copy()
            ).to(device, non_blocking=True)
            values = torch.from_numpy(
                np.asarray(shard.values[batch_start:batch_stop], dtype=np.float32).copy()
            ).to(device, non_blocking=True)
            increments += accumulate_batch(
                counts,
                feature_ids,
                values,
                start,
                stop,
                float(plan["activation_threshold"]),
            )
        rows_done += shard.rows
        if shard_number % args.progress_every == 0 or shard_number == len(shards):
            torch.cuda.synchronize(device)
            print(
                f"Block {args.block_id}: {shard_number}/{len(shards)} shards, "
                f"{rows_done:,} rows, {increments:,} pair increments, "
                f"{(time.monotonic() - started) / 60:.1f} min",
                flush=True,
            )

    torch.cuda.synchronize(device)
    observed_sum = int(counts.sum(dtype=torch.int64).item())
    if observed_sum != increments:
        raise RuntimeError(
            f"Count sum {observed_sum} differs from increments {increments}"
        )
    maximum = int(counts.max().item())
    save_tensor_npy(block_path, counts, args.save_row_chunk)
    marker = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "plan_fingerprint": plan["plan_fingerprint"],
        "block_id": args.block_id,
        "row_start": start,
        "row_stop": stop,
        "shape": [stop - start, width],
        "dtype": "uint32",
        "path": str(block_path),
        "source_rows": rows_done,
        "source_shards": len(shards),
        "pair_increments": increments,
        "matrix_sum": observed_sum,
        "maximum_count": maximum,
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json(marker_path, marker)
    print(f"Wrote block {args.block_id}: {block_path}")


def load_valid_marker(output_root: Path, plan: dict[str, Any], block_id: int) -> dict[str, Any]:
    marker_path = output_marker_path(output_root, block_id)
    marker = read_json(marker_path)
    if marker.get("complete") is not True:
        raise ValueError(f"Incomplete block marker: {marker_path}")
    if marker.get("plan_fingerprint") != plan["plan_fingerprint"]:
        raise ValueError(f"Block marker has wrong plan fingerprint: {marker_path}")
    block = plan["blocks"][block_id]
    path = _paper_path(block["path"])
    matrix = np.load(_paper_location(path), mmap_mode="r", allow_pickle=False)
    expected_shape = tuple(int(item) for item in block["shape"])
    if matrix.shape != expected_shape or matrix.dtype != np.uint32:
        raise ValueError(f"Invalid matrix block {path}: {matrix.shape}, {matrix.dtype}")
    return marker


def finalize(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    plan = read_json(output_root / "plan.json")
    markers = [
        load_valid_marker(output_root, plan, block_id)
        for block_id in range(int(plan["num_blocks"]))
    ]
    result = {
        **plan,
        "complete": True,
        "total_pair_increments": sum(int(item["pair_increments"]) for item in markers),
        "maximum_count": max(int(item["maximum_count"]) for item in markers),
        "total_size_bytes": sum(
            _paper_path(block["path"]).stat().st_size for block in plan["blocks"]
        ),
        "block_metadata": str((output_root / "block_metadata").resolve()),
    }
    atomic_json(output_root / "manifest.json", result)
    atomic_json(
        output_root / "COMPLETE.json",
        {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "manifest": str((output_root / "manifest.json").resolve()),
            "model": plan["model"],
            "shape": plan["matrix_shape"],
            "dtype": plan["matrix_dtype"],
            "activation_threshold": plan["activation_threshold"],
            "source_rows": plan["source_rows"],
            "source_shards": plan["source_shards"],
            "total_size_bytes": result["total_size_bytes"],
        },
    )
    print(f"Finalized exact co-occurrence artifact: {output_root}")


def inspect_feature(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    plan = read_json(output_root / "manifest.json")
    width = int(plan["width"])
    if args.feature < 0 or args.feature >= width:
        raise ValueError(f"feature must be in [0, {width - 1}]")
    block_id = args.feature // int(plan["block_size"])
    block = plan["blocks"][block_id]
    matrix = np.load(_paper_location(block["path"]), mmap_mode="r", allow_pickle=False)
    row = np.asarray(matrix[args.feature - int(block["row_start"])])
    ranking = row.copy()
    ranking[args.feature] = 0
    k = min(args.top_k, width - 1)
    candidates = np.argpartition(ranking, -k)[-k:]
    candidates = candidates[np.argsort(ranking[candidates])[::-1]]
    print(f"feature,support,{args.feature},{int(row[args.feature])}")
    print("rank,neighbor,cooccurrence_count")
    for rank, neighbor in enumerate(candidates, start=1):
        print(f"{rank},{int(neighbor)},{int(row[neighbor])}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--cache-root", type=Path, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    prepare_parser.add_argument("--block-size", type=int, default=8192)
    prepare_parser.add_argument("--threshold", type=float, default=0.0)
    prepare_parser.set_defaults(func=prepare)

    compute_parser = commands.add_parser("compute")
    compute_parser.add_argument("--output-root", type=Path, required=True)
    compute_parser.add_argument("--block-id", type=int, required=True)
    compute_parser.add_argument("--batch-rows", type=int, default=8192)
    compute_parser.add_argument("--progress-every", type=int, default=5)
    compute_parser.add_argument("--save-row-chunk", type=int, default=64)
    compute_parser.add_argument("--device", default="cuda")
    compute_parser.set_defaults(func=compute)

    finalize_parser = commands.add_parser("finalize")
    finalize_parser.add_argument("--output-root", type=Path, required=True)
    finalize_parser.set_defaults(func=finalize)

    inspect_parser = commands.add_parser("inspect-feature")
    inspect_parser.add_argument("--output-root", type=Path, required=True)
    inspect_parser.add_argument("--feature", type=int, required=True)
    inspect_parser.add_argument("--top-k", type=int, default=20)
    inspect_parser.set_defaults(func=inspect_feature)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    arguments.func(arguments)
