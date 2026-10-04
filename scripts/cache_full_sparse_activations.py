#!/usr/bin/env python3
"""Cache complete Gemini SAE TopK activations in shard-aligned CSR matrices."""

from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import argparse
import csv
import hashlib
import heapq
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(_paper_path(__file__).resolve().parents[1]))
from project_paths import get_path
from scripts.corpus_io import (
    GENERAL_CORPUS_DATASETS,
    SparseTopKValidator,
    TextResolver,
    discover_shards,
    load_state,
)


SCHEMA_VERSION = 1
ACTIVATION_FORMULA = "TopK_k(relu((embedding - b_dec) @ W_enc))"
EXPECTED_INPUT_DIM = 3072
EXPECTED_SHARDS = 929
EXPECTED_ROWS = 89_827_558
DEFAULT_CACHE_ROOT = _paper_path(
    "/resources/activation_cache_dir/"
    "gemini_all_corpus_post_topk"
)
SAE_SPECS = (
    {"name": "gemini_m4096_k32", "width": 4096, "top_k": 32},
    {"name": "gemini_m131072_k128", "width": 131072, "top_k": 128},
)
TRACKED_FEATURES = (
    {"key": "gemini_m4096_feature_1667", "role": "target",
     "width": 4096, "feature_id": 1667},
    {"key": "gemini_m131072_feature_66514", "role": "atom_1",
     "width": 131072, "feature_id": 66514},
    {"key": "gemini_m131072_feature_120924", "role": "atom_2",
     "width": 131072, "feature_id": 120924},
    {"key": "gemini_m131072_feature_74537", "role": "atom_3",
     "width": 131072, "feature_id": 74537},
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp.{os.environ.get('SLURM_JOB_ID', 'local')}.{os.getpid()}"
    )
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def update_heap(
    heap: list[tuple[float, str, int]],
    activation: float,
    shard: str,
    row: int,
    limit: int,
) -> None:
    item = (activation, shard, row)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def checkpoint_record(models_dir: Path, spec: dict[str, Any]) -> dict[str, Any]:
    path = (models_dir / spec["name"]).resolve()
    stat = path.stat()
    state = load_state(path)
    weights, bias = state["W_enc"], state["b_dec"]
    expected = (int(bias.numel()), int(spec["width"]))
    if tuple(weights.shape) != expected:
        raise ValueError(f"{path}: W_enc {tuple(weights.shape)} != {expected}")
    return {
        **spec,
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
        "input_dim": int(weights.shape[0]),
        "weight_shape": list(weights.shape),
        "bias_shape": list(bias.shape),
    }


def source_record(embeddings_dir: Path, shard: Path) -> dict[str, Any]:
    relative = str(shard.relative_to(embeddings_dir))
    embedding_path = shard / "embeddings.npy"
    metadata_path = shard / "metadata.parquet"
    meta_path = shard / "shard_meta.json"
    embeddings = np.load(_paper_location(embedding_path), mmap_mode="r")
    if embeddings.ndim != 2 or embeddings.shape[1] != EXPECTED_INPUT_DIM:
        raise ValueError(f"Unexpected shape {embeddings.shape}: {relative}")
    if embeddings.dtype != np.float16:
        raise ValueError(f"Unexpected dtype {embeddings.dtype}: {relative}")
    shard_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    parquet_rows = int(pq.ParquetFile(metadata_path).metadata.num_rows)
    if int(shard_meta["rows"]) != len(embeddings) or parquet_rows != len(embeddings):
        raise ValueError(f"Row metadata mismatch: {relative}")
    embedding_stat, metadata_stat = embedding_path.stat(), metadata_path.stat()
    return {
        "relative_shard": relative,
        "rows": int(embeddings.shape[0]),
        "dim": int(embeddings.shape[1]),
        "dtype": str(embeddings.dtype),
        "embeddings_size_bytes": embedding_stat.st_size,
        "embeddings_mtime_ns": embedding_stat.st_mtime_ns,
        "metadata_rows": parquet_rows,
        "metadata_size_bytes": metadata_stat.st_size,
        "metadata_mtime_ns": metadata_stat.st_mtime_ns,
        "shard_meta_sha256": sha256_file(meta_path),
        "text_column": str(shard_meta["text_column"]),
    }


def balanced_assignments(records: list[dict[str, Any]], tasks: int) -> list[int]:
    loads = [0] * tasks
    assignments = [-1] * len(records)
    order = sorted(
        range(len(records)),
        key=lambda i: (-int(records[i]["rows"]), records[i]["relative_shard"]),
    )
    for index in order:
        task = min(range(tasks), key=lambda i: (loads[i], i))
        assignments[index] = task
        loads[task] += int(records[index]["rows"])
    return assignments


def prepare(args: argparse.Namespace) -> None:
    started = time.monotonic()
    datasets = tuple(part for part in args.datasets.split(",") if part)
    if datasets != GENERAL_CORPUS_DATASETS:
        raise ValueError("Use the exact ordered 14-dataset Gemini training corpus")
    shards = discover_shards(args.embeddings_dir, datasets)
    sources = [source_record(args.embeddings_dir, shard) for shard in shards]
    assignments = balanced_assignments(sources, args.num_tasks)
    global_start = 0
    for source, task in zip(sources, assignments):
        source["global_row_start"] = global_start
        source["global_row_stop"] = global_start + int(source["rows"])
        source["task_id"] = task
        global_start = int(source["global_row_stop"])
    relative_paths = [source["relative_shard"] for source in sources]
    if len(relative_paths) != len(set(relative_paths)):
        raise RuntimeError("Duplicate corpus shard paths")
    if len(sources) != EXPECTED_SHARDS or global_start != EXPECTED_ROWS:
        raise RuntimeError(
            f"Corpus coverage changed: got {len(sources)} shards / "
            f"{global_start:,} rows; expected {EXPECTED_SHARDS} / "
            f"{EXPECTED_ROWS:,}"
        )
    models = [checkpoint_record(args.models_dir, spec) for spec in SAE_SPECS]
    identity = {
        "schema_version": SCHEMA_VERSION,
        "activation_formula": ACTIVATION_FORMULA,
        "matrix_format": "csr",
        "compute_dtype": "float32",
        "value_dtype": "float32",
        "index_dtype": "int32",
        "indptr_dtype": "int32",
        "explicit_zero_slots": True,
        "explicit_zero_semantics": (
            "indices paired with zero data are structural padding and non-semantic"
        ),
        "values_descending_within_row": True,
        "models": models,
        "embeddings_dir": str(args.embeddings_dir.resolve()),
        "datasets": list(datasets),
        "num_tasks": args.num_tasks,
        "tracked_features": list(TRACKED_FEATURES),
        "tracked_top_per_task": args.tracked_top_per_task,
        "source_shards": sources,
    }
    spec_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    plan = {
        "complete": True,
        "spec_id": spec_id,
        "cache_root": str(args.cache_root.resolve()),
        "expected_shards": len(sources),
        "expected_rows": global_start,
        "created_at": utc_now(),
        **identity,
    }
    plan_path = args.cache_root / "plan.json"
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        comparable = lambda item: {k: v for k, v in item.items() if k != "created_at"}
        if comparable(existing) != comparable(plan):
            raise RuntimeError(f"Incompatible existing plan: {plan_path}")
        print(f"Compatible plan already exists: {plan_path}")
    else:
        atomic_write_json(plan_path, plan)
        print(f"Wrote {plan_path}")
    loads = [
        sum(int(s["rows"]) for s in sources if int(s["task_id"]) == task)
        for task in range(args.num_tasks)
    ]
    print(json.dumps({
        "spec_id": spec_id,
        "shards": len(sources),
        "rows": global_start,
        "task_rows_min": min(loads),
        "task_rows_max": max(loads),
        "elapsed_seconds": time.monotonic() - started,
    }, indent=2))


def load_plan(cache_root: Path) -> dict[str, Any]:
    plan = json.loads((cache_root / "plan.json").read_text(encoding="utf-8"))
    if plan.get("complete") is not True or plan.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("Invalid cache plan")
    if _paper_path(plan["cache_root"]).resolve() != cache_root.resolve():
        raise RuntimeError("Cache root does not match plan")
    return plan


def expected_npy(path: Path, shape: tuple[int, ...], dtype: np.dtype) -> bool:
    try:
        array = np.load(_paper_location(path), mmap_mode="r")
    except (OSError, ValueError):
        return False
    return array.shape == shape and array.dtype == dtype


def completed_shard(
    cache_root: Path, plan: dict[str, Any], source: dict[str, Any]
) -> tuple[Path, dict[str, Any]] | None:
    shard_dir = cache_root / "shards" / source["relative_shard"]
    marker = shard_dir / "complete.json"
    if not marker.is_file():
        return None
    try:
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        if (
            metadata.get("complete") is not True
            or metadata.get("spec_id") != plan["spec_id"]
            or metadata.get("source") != source
        ):
            return None
        rows = int(source["rows"])
        for model in plan["models"]:
            name, width, top_k = (
                model["name"], int(model["width"]), int(model["top_k"])
            )
            output, model_dir = metadata["outputs"][name], shard_dir / name
            if output["shape"] != [rows, width] or output["top_k"] != top_k:
                return None
            if not expected_npy(
                model_dir / "indptr.npy", (rows + 1,), np.dtype("int32")
            ):
                return None
            if not expected_npy(
                model_dir / "indices.npy", (rows * top_k,), np.dtype("int32")
            ):
                return None
            if not expected_npy(
                model_dir / "data.npy", (rows * top_k,), np.dtype("float32")
            ):
                return None
            indptr = np.load(_paper_location(model_dir / "indptr.npy"), mmap_mode="r")
            expected_ptr = np.arange(rows + 1, dtype=np.int32) * top_k
            if not np.array_equal(indptr, expected_ptr):
                return None
        return shard_dir, metadata
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def validate_source(embeddings_dir: Path, source: dict[str, Any]) -> np.ndarray:
    path = embeddings_dir / source["relative_shard"] / "embeddings.npy"
    stat = path.stat()
    embeddings = np.load(_paper_location(path), mmap_mode="r")
    shard_dir = path.parent
    metadata_stat = (shard_dir / "metadata.parquet").stat()
    if (
        list(embeddings.shape) != [int(source["rows"]), int(source["dim"])]
        or str(embeddings.dtype) != source["dtype"]
        or stat.st_size != int(source["embeddings_size_bytes"])
        or stat.st_mtime_ns != int(source["embeddings_mtime_ns"])
        or metadata_stat.st_size != int(source["metadata_size_bytes"])
        or metadata_stat.st_mtime_ns != int(source["metadata_mtime_ns"])
        or sha256_file(shard_dir / "shard_meta.json")
        != source["shard_meta_sha256"]
    ):
        raise RuntimeError(f"Source changed: {source['relative_shard']}")
    return embeddings


def validate_checkpoint(record: dict[str, Any]) -> None:
    path = _paper_path(record["path"])
    stat = path.stat()
    if (
        stat.st_size != int(record["size_bytes"])
        or stat.st_mtime_ns != int(record["mtime_ns"])
        or sha256_file(path) != record["sha256"]
    ):
        raise RuntimeError(f"Checkpoint changed after prepare: {path}")


def load_models(
    plan: dict[str, Any], device: torch.device
) -> dict[str, tuple[torch.Tensor, torch.Tensor, int, int]]:
    models = {}
    for record in plan["models"]:
        validate_checkpoint(record)
        state = load_state(_paper_path(record["path"]))
        weights = state["W_enc"].detach().to(device=device, dtype=torch.float32)
        bias = state["b_dec"].detach().to(device=device, dtype=torch.float32)
        models[record["name"]] = (
            weights, bias, int(record["width"]), int(record["top_k"])
        )
    return models


@torch.inference_mode()
def compute_batch(
    embeddings: np.ndarray,
    start: int,
    stop: int,
    models: dict[str, tuple[torch.Tensor, torch.Tensor, int, int]],
    model_records: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    x = torch.from_numpy(np.asarray(embeddings[start:stop]).copy()).to(
        device=device, dtype=torch.float32
    )
    outputs = {}
    for record in model_records:
        name = record["name"]
        weights, bias, _, top_k = models[name]
        pre = torch.relu((x - bias) @ weights)
        values, indices = torch.topk(
            pre, k=top_k, dim=1, largest=True, sorted=True
        )
        outputs[name] = (
            indices.cpu().numpy().astype(np.int32, copy=False),
            values.cpu().numpy().astype(np.float32, copy=False),
        )
    return outputs


def flush_and_fsync(array: np.memmap, path: Path) -> None:
    array.flush()
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def tracked_items(
    heaps: dict[str, list[tuple[float, str, int]]]
) -> dict[str, list[dict[str, Any]]]:
    return {
        key: [
            {"activation": activation, "shard": shard, "shard_row": row}
            for activation, shard, row in sorted(heap, reverse=True)
        ]
        for key, heap in heaps.items()
    }


def add_tracked_items(
    heaps: dict[str, list[tuple[float, str, int]]],
    items_by_key: dict[str, list[dict[str, Any]]],
    limit: int,
) -> None:
    for key, items in items_by_key.items():
        for item in items:
            update_heap(
                heaps[key],
                float(item["activation"]),
                str(item["shard"]),
                int(item["shard_row"]),
                limit,
            )


@torch.inference_mode()
def compute_one_shard(
    cache_root: Path,
    plan: dict[str, Any],
    source: dict[str, Any],
    embeddings: np.ndarray,
    models: dict[str, tuple[torch.Tensor, torch.Tensor, int, int]],
    device: torch.device,
    initial_batch_size: int,
) -> dict[str, Any]:
    final_dir = cache_root / "shards" / source["relative_shard"]
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = final_dir.with_name(
        f".{final_dir.name}.tmp.{os.environ.get('SLURM_JOB_ID', 'local')}."
        f"{os.environ.get('SLURM_ARRAY_TASK_ID', '0')}.{uuid.uuid4().hex}"
    )
    if temporary_dir.exists():
        raise RuntimeError(f"Temporary path exists: {temporary_dir}")
    temporary_dir.mkdir(parents=True)

    rows = int(source["rows"])
    arrays: dict[str, dict[str, Any]] = {}
    stats: dict[str, dict[str, Any]] = {}
    digests: dict[str, dict[str, Any]] = {}
    local_heaps = {feature["key"]: [] for feature in plan["tracked_features"]}
    heap_limit = int(plan["tracked_top_per_task"])
    try:
        for record in plan["models"]:
            name, top_k = record["name"], int(record["top_k"])
            model_dir = temporary_dir / name
            model_dir.mkdir()
            paths = {
                component: model_dir / f"{component}.npy"
                for component in ("indptr", "indices", "data")
            }
            indptr = np.lib.format.open_memmap(
                paths["indptr"], mode="w+", dtype=np.int32, shape=(rows + 1,)
            )
            indptr[:] = np.arange(rows + 1, dtype=np.int32) * top_k
            arrays[name] = {
                "indptr": indptr,
                "indices": np.lib.format.open_memmap(
                    paths["indices"], mode="w+", dtype=np.int32,
                    shape=(rows * top_k,)
                ),
                "data": np.lib.format.open_memmap(
                    paths["data"], mode="w+", dtype=np.float32,
                    shape=(rows * top_k,)
                ),
                "paths": paths,
            }
            stats[name] = {
                "positive_nnz": 0,
                "explicit_zero_slots": 0,
                "minimum": float("inf"),
                "maximum": float("-inf"),
            }
            digests[name] = {
                "indices": hashlib.sha256(),
                "data": hashlib.sha256(),
            }

        batch_size, start = initial_batch_size, 0
        while start < rows:
            stop = min(start + batch_size, rows)
            try:
                batch_outputs = compute_batch(
                    embeddings, start, stop, models, plan["models"], device
                )
            except torch.cuda.OutOfMemoryError:
                if batch_size <= 32:
                    raise
                batch_size = max(32, batch_size // 2)
                torch.cuda.empty_cache()
                print(
                    f"OOM in {source['relative_shard']}; retrying batch "
                    f"{start} at size {batch_size}",
                    flush=True,
                )
                continue

            for record in plan["models"]:
                name = record["name"]
                width, top_k = int(record["width"]), int(record["top_k"])
                indices_np, values_np = batch_outputs[name]
                if (
                    not np.isfinite(values_np).all()
                    or (values_np < 0).any()
                    or (indices_np < 0).any()
                    or (indices_np >= width).any()
                ):
                    raise ValueError(f"Invalid TopK output for {name}")
                if values_np.shape[1] > 1 and (
                    values_np[:, 1:] > values_np[:, :-1]
                ).any():
                    raise ValueError(f"TopK values not descending for {name}")
                flat_indices = indices_np.reshape(-1)
                flat_values = values_np.reshape(-1)
                destination = slice(start * top_k, stop * top_k)
                arrays[name]["indices"][destination] = flat_indices
                arrays[name]["data"][destination] = flat_values
                digests[name]["indices"].update(flat_indices.tobytes(order="C"))
                digests[name]["data"].update(flat_values.tobytes(order="C"))
                positive = int(np.count_nonzero(flat_values > 0))
                stats[name]["positive_nnz"] += positive
                stats[name]["explicit_zero_slots"] += int(
                    flat_values.size - positive
                )
                stats[name]["minimum"] = min(
                    stats[name]["minimum"], float(flat_values.min())
                )
                stats[name]["maximum"] = max(
                    stats[name]["maximum"], float(flat_values.max())
                )
                for feature in plan["tracked_features"]:
                    if int(feature["width"]) != width:
                        continue
                    matching_rows, positions = np.nonzero(
                        indices_np == int(feature["feature_id"])
                    )
                    for row, position in zip(matching_rows, positions):
                        activation = float(values_np[row, position])
                        if activation > 0:
                            update_heap(
                                local_heaps[feature["key"]],
                                activation,
                                source["relative_shard"],
                                start + int(row),
                                heap_limit,
                            )
            start = stop

        outputs = {}
        for record in plan["models"]:
            name = record["name"]
            width, top_k = int(record["width"]), int(record["top_k"])
            for component in ("indptr", "indices", "data"):
                flush_and_fsync(
                    arrays[name][component], arrays[name]["paths"][component]
                )
            outputs[name] = {
                "shape": [rows, width],
                "top_k": top_k,
                "stored_slots": rows * top_k,
                **stats[name],
                "components": {
                    "indptr": {
                        "file": f"{name}/indptr.npy",
                        "dtype": "int32",
                        "shape": [rows + 1],
                        "size_bytes": arrays[name]["paths"]["indptr"].stat().st_size,
                    },
                    "indices": {
                        "file": f"{name}/indices.npy",
                        "dtype": "int32",
                        "shape": [rows * top_k],
                        "size_bytes": arrays[name]["paths"]["indices"].stat().st_size,
                        "payload_sha256": digests[name]["indices"].hexdigest(),
                    },
                    "data": {
                        "file": f"{name}/data.npy",
                        "dtype": "float32",
                        "shape": [rows * top_k],
                        "size_bytes": arrays[name]["paths"]["data"].stat().st_size,
                        "payload_sha256": digests[name]["data"].hexdigest(),
                    },
                },
            }
        for model_arrays in arrays.values():
            for component in ("indptr", "indices", "data"):
                del model_arrays[component]

        metadata = {
            "complete": True,
            "schema_version": SCHEMA_VERSION,
            "spec_id": plan["spec_id"],
            "activation_formula": ACTIVATION_FORMULA,
            "matrix_format": "csr",
            "source": source,
            "outputs": outputs,
            "tracked_top": tracked_items(local_heaps),
            "producer": {
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                "device": str(device),
                "initial_batch_size": initial_batch_size,
                "final_batch_size": batch_size,
                "torch_version": torch.__version__,
                "numpy_version": np.__version__,
            },
            "completed_at": utc_now(),
        }
        atomic_write_json(temporary_dir / "complete.json", metadata)
        if final_dir.exists():
            raise RuntimeError(f"Output appeared concurrently: {final_dir}")
        os.replace(temporary_dir, final_dir)
        return metadata
    except Exception:
        print(f"Incomplete temporary output retained at {temporary_dir}", flush=True)
        raise


def compute(args: argparse.Namespace) -> None:
    if not 0 <= args.task_id < args.num_tasks:
        raise ValueError("task-id must be in [0, num-tasks)")
    if args.device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    plan = load_plan(args.cache_root)
    if int(plan["num_tasks"]) != args.num_tasks:
        raise RuntimeError("--num-tasks disagrees with plan")
    torch.set_float32_matmul_precision("highest")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
    sources = [
        source for source in plan["source_shards"]
        if int(source["task_id"]) == args.task_id
    ]
    task_heaps = {feature["key"]: [] for feature in plan["tracked_features"]}
    limit = int(plan["tracked_top_per_task"])
    models = load_models(plan, args.device)
    rows = computed = reused = 0
    started = time.monotonic()
    for number, source in enumerate(sources, start=1):
        embeddings = validate_source(args.embeddings_dir, source)
        completed = completed_shard(args.cache_root, plan, source)
        if completed is None:
            invalid_dir = args.cache_root / "shards" / source["relative_shard"]
            if invalid_dir.exists():
                raise RuntimeError(
                    f"Existing cache shard is invalid: {invalid_dir}. "
                    "Move it aside before retrying."
                )
            metadata = compute_one_shard(
                args.cache_root, plan, source, embeddings, models,
                args.device, args.batch_size
            )
            computed += 1
        else:
            _, metadata = completed
            reused += 1
        add_tracked_items(task_heaps, metadata["tracked_top"], limit)
        rows += int(source["rows"])
        if number % args.progress_every == 0 or number == len(sources):
            print(
                f"Task {args.task_id}: {number}/{len(sources)} shards, "
                f"{rows:,} rows ({reused} reused)",
                flush=True,
            )
    summary = {
        "complete": True,
        "spec_id": plan["spec_id"],
        "task_id": args.task_id,
        "num_tasks": args.num_tasks,
        "assigned_shards": [source["relative_shard"] for source in sources],
        "rows": rows,
        "shards": len(sources),
        "computed_shards": computed,
        "reused_shards": reused,
        "tracked_top_per_task": limit,
        "tracked_top": tracked_items(task_heaps),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "elapsed_seconds": time.monotonic() - started,
        "completed_at": utc_now(),
    }
    path = args.cache_root / "task_summaries" / f"task_{args.task_id:03d}.json"
    atomic_write_json(path, summary)
    print(f"Wrote {path}", flush=True)


def structural_audit(
    cache_root: Path, plan: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, float]]:
    records = []
    candidate_limit = (
        int(plan["tracked_top_per_task"]) * int(plan["num_tasks"])
    )
    heaps = {feature["key"]: [] for feature in plan["tracked_features"]}
    for source in plan["source_shards"]:
        completed = completed_shard(cache_root, plan, source)
        if completed is None:
            raise RuntimeError(f"Missing/invalid shard: {source['relative_shard']}")
        shard_dir, metadata = completed
        for model in plan["models"]:
            name = model["name"]
            width, top_k = int(model["width"]), int(model["top_k"])
            output = metadata["outputs"][name]
            indices = np.load(_paper_location(shard_dir / name / "indices.npy"), mmap_mode="r")
            data = np.load(_paper_location(shard_dir / name / "data.npy"), mmap_mode="r")
            indices_digest, data_digest = hashlib.sha256(), hashlib.sha256()
            rows = int(source["rows"])
            for row_start in range(0, rows, 8192):
                row_stop = min(row_start + 8192, rows)
                begin, end = row_start * top_k, row_stop * top_k
                indices_chunk = np.asarray(indices[begin:end])
                data_chunk = np.asarray(data[begin:end])
                indices_digest.update(indices_chunk.tobytes(order="C"))
                data_digest.update(data_chunk.tobytes(order="C"))
                if (
                    int(indices_chunk.min()) < 0
                    or int(indices_chunk.max()) >= width
                    or not np.isfinite(data_chunk).all()
                    or (data_chunk < 0).any()
                ):
                    raise RuntimeError(
                        f"Invalid CSR payload: {source['relative_shard']} / {name}"
                    )
                indices_2d = indices_chunk.reshape(row_stop - row_start, top_k)
                values_2d = data_chunk.reshape(row_stop - row_start, top_k)
                if (values_2d[:, 1:] > values_2d[:, :-1]).any():
                    raise RuntimeError(
                        f"Non-descending values: "
                        f"{source['relative_shard']} / {name}"
                    )
                sorted_indices = np.sort(indices_2d, axis=1)
                if (sorted_indices[:, 1:] == sorted_indices[:, :-1]).any():
                    raise RuntimeError(
                        f"Duplicate feature in CSR row: "
                        f"{source['relative_shard']} / {name}"
                    )
                for feature in plan["tracked_features"]:
                    if int(feature["width"]) != width:
                        continue
                    matching_rows, positions = np.nonzero(
                        indices_2d == int(feature["feature_id"])
                    )
                    for row, position in zip(matching_rows, positions):
                        activation = float(values_2d[row, position])
                        if activation > 0:
                            update_heap(
                                heaps[feature["key"]],
                                activation,
                                source["relative_shard"],
                                row_start + int(row),
                                candidate_limit,
                            )
            if (
                indices_digest.hexdigest()
                != output["components"]["indices"]["payload_sha256"]
                or data_digest.hexdigest()
                != output["components"]["data"]["payload_sha256"]
            ):
                raise RuntimeError(
                    f"Payload checksum mismatch: "
                    f"{source['relative_shard']} / {name}"
                )
        records.append({
            **source,
            "cache_shard_dir": str(shard_dir.resolve()),
            "complete_metadata": str((shard_dir / "complete.json").resolve()),
            "outputs": metadata["outputs"],
        })
        if len(records) % 25 == 0 or len(records) == EXPECTED_SHARDS:
            print(
                f"Audit: {len(records)}/{EXPECTED_SHARDS} shards structurally "
                "validated",
                flush=True,
            )
    if len(records) != EXPECTED_SHARDS:
        raise RuntimeError("Audited shard count mismatch")
    if sum(int(record["rows"]) for record in records) != EXPECTED_ROWS:
        raise RuntimeError("Audited row count mismatch")
    candidates = tracked_items(heaps)
    omitted_bounds = {
        key: (float(heap[0][0]) if len(heap) == candidate_limit else 0.0)
        for key, heap in heaps.items()
    }
    return records, candidates, omitted_bounds


@torch.inference_mode()
def recompute_samples(
    args: argparse.Namespace,
    plan: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    if args.validation_samples <= 0:
        return {"samples": 0}
    rng = np.random.default_rng(args.validation_seed)
    random_rows = rng.choice(
            EXPECTED_ROWS,
            size=min(args.validation_samples, EXPECTED_ROWS),
            replace=False,
    )
    boundary_rows = [
        row
        for record in records
        for row in (
            int(record["global_row_start"]),
            int(record["global_row_stop"]) - 1,
        )
    ]
    global_rows = np.asarray(
        sorted(set(int(row) for row in random_rows) | set(boundary_rows)),
        dtype=np.int64,
    )
    stops = np.asarray(
        [record["global_row_stop"] for record in records], dtype=np.int64
    )
    sample_items = []
    for global_row in global_rows:
        shard_index = int(np.searchsorted(stops, global_row, side="right"))
        source = records[shard_index]
        local_row = int(global_row - int(source["global_row_start"]))
        embedding = np.asarray(
            np.load(
                _paper_location(args.embeddings_dir
                / source["relative_shard"]
                / "embeddings.npy"),
                mmap_mode="r",
            )[local_row],
            dtype=np.float32,
        )
        sample_items.append((source, local_row, embedding))

    maximum_error = 0.0
    models = load_models(plan, args.device)
    for model_record in plan["models"]:
        name = model_record["name"]
        weights, bias, _, top_k = models[name]
        for start in range(0, len(sample_items), args.validation_batch_size):
            batch_items = sample_items[start : start + args.validation_batch_size]
            x = torch.from_numpy(np.stack([item[2] for item in batch_items])).to(
                args.device, torch.float32
            )
            values, indices = torch.topk(
                torch.relu((x - bias) @ weights),
                k=top_k,
                dim=1,
                largest=True,
                sorted=True,
            )
            values_np = values.cpu().numpy()
            indices_np = indices.cpu().numpy().astype(np.int32, copy=False)
            for batch_row, (source, local_row, _) in enumerate(batch_items):
                model_dir = (
                    args.cache_root / "shards" / source["relative_shard"] / name
                )
                begin, end = local_row * top_k, (local_row + 1) * top_k
                cached_indices = np.load(
                    _paper_location(model_dir / "indices.npy"), mmap_mode="r"
                )[begin:end]
                cached_values = np.load(
                    _paper_location(model_dir / "data.npy"), mmap_mode="r"
                )[begin:end]
                cached_positive = cached_values > 0
                recomputed_positive = values_np[batch_row] > 0
                cached_support = np.sort(cached_indices[cached_positive])
                recomputed_support = np.sort(
                    indices_np[batch_row][recomputed_positive]
                )
                if not np.array_equal(cached_support, recomputed_support):
                    raise RuntimeError(
                        f"Support mismatch: {source['relative_shard']} "
                        f"row {local_row} / {name}"
                    )
                cached_pairs = sorted(
                    zip(cached_indices[cached_positive], cached_values[cached_positive])
                )
                recomputed_pairs = sorted(
                    zip(
                        indices_np[batch_row][recomputed_positive],
                        values_np[batch_row][recomputed_positive],
                    )
                )
                error = max(
                    (abs(float(a[1]) - float(b[1])) for a, b in zip(cached_pairs, recomputed_pairs)),
                    default=0.0,
                )
                maximum_error = max(maximum_error, error)
                if error > args.validation_atol:
                    raise RuntimeError(
                        f"Activation mismatch {error}: "
                        f"{source['relative_shard']} row {local_row} / {name}"
                    )
    return {
        "samples": len(sample_items),
        "seed": args.validation_seed,
        "absolute_tolerance": args.validation_atol,
        "maximum_absolute_error": maximum_error,
    }


def load_task_summaries(
    cache_root: Path, plan: dict[str, Any]
) -> list[dict[str, Any]]:
    summaries = []
    for task_id in range(int(plan["num_tasks"])):
        path = cache_root / "task_summaries" / f"task_{task_id:03d}.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        expected = sorted(
            source["relative_shard"]
            for source in plan["source_shards"]
            if int(source["task_id"]) == task_id
        )
        if (
            summary.get("complete") is not True
            or summary.get("spec_id") != plan["spec_id"]
            or int(summary["task_id"]) != task_id
            or sorted(summary["assigned_shards"]) != expected
        ):
            raise RuntimeError(f"Invalid task summary: {path}")
        summaries.append(summary)
    return summaries


def atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def resolve_top_examples(
    args: argparse.Namespace,
    plan: dict[str, Any],
    candidates_by_feature: dict[str, list[dict[str, Any]]],
    omitted_bounds: dict[str, float],
) -> dict[str, Any]:
    if not plan["tracked_features"]:
        report = {
            "cache_root": str(args.cache_root.resolve()),
            "cache_spec_id": plan["spec_id"],
            "activation_definition": ACTIVATION_FORMULA,
            "deduplication": None,
            "top_n": 0,
            "features": [],
        }
        atomic_write_json(args.report_dir / "top_examples.json", report)
        return report

    resolver = TextResolver(args.embeddings_dir, args.datasets_dir)
    model_parents = {str(_paper_path(model["path"]).parent) for model in plan["models"]}
    if len(model_parents) != 1:
        raise RuntimeError("Planned checkpoints do not share one models directory")
    planned_models_dir = _paper_path(next(iter(model_parents)))
    if args.models_dir.resolve() != planned_models_dir.resolve():
        raise RuntimeError(
            f"--models-dir {args.models_dir} differs from plan {planned_models_dir}"
        )
    validator = SparseTopKValidator(
        planned_models_dir, args.embeddings_dir,
        args.device, args.validation_batch_size
    )
    output_features, flat_rows = [], []
    for feature in plan["tracked_features"]:
        candidates = list(candidates_by_feature[feature["key"]])
        candidates.sort(key=lambda item: float(item["activation"]), reverse=True)
        omitted_bound = float(omitted_bounds[feature["key"]])
        examples, seen, candidates_checked = [], set(), 0
        for start in range(0, len(candidates), args.validation_batch_size):
            batch = candidates[start : start + args.validation_batch_size]
            for candidate in validator.validate(batch, feature):
                candidates_checked += 1
                if not candidate["survives_topk"]:
                    raise RuntimeError("Cached candidate failed full TopK validation")
                resolved = resolver.resolve(candidate)
                dedupe_key = " ".join(resolved["text"].split())
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                resolved["rank"] = len(examples) + 1
                examples.append(resolved)
                flat_rows.append({
                    "feature_key": feature["key"],
                    "role": feature["role"],
                    "width": feature["width"],
                    "feature_id": feature["feature_id"],
                    **resolved,
                })
                if len(examples) == args.top_n:
                    break
            if len(examples) == args.top_n:
                break
        if len(examples) < args.top_n:
            raise RuntimeError(f"Too few unique examples for {feature['key']}")
        if float(examples[-1]["activation"]) <= omitted_bound:
            raise RuntimeError(f"Candidate pool too shallow for {feature['key']}")
        output_features.append({
            **feature,
            "candidates_checked": candidates_checked,
            "omitted_activation_upper_bound": omitted_bound,
            "exact_top_n_certified": True,
            "top_examples": examples,
        })
    report = {
        "cache_root": str(args.cache_root.resolve()),
        "cache_spec_id": plan["spec_id"],
        "activation_definition": ACTIVATION_FORMULA,
        "deduplication": (
            "exact text after whitespace normalization, independently per feature"
        ),
        "top_n": args.top_n,
        "features": output_features,
    }
    atomic_write_json(args.report_dir / "top_examples.json", report)
    atomic_write_csv(args.report_dir / "top_examples.csv", flat_rows)
    return report


def audit(args: argparse.Namespace) -> None:
    if args.device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_float32_matmul_precision("highest")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
    plan = load_plan(args.cache_root)
    records, candidates, omitted_bounds = structural_audit(
        args.cache_root, plan
    )
    load_task_summaries(args.cache_root, plan)
    sample_validation = recompute_samples(args, plan, records)
    top_report = resolve_top_examples(
        args, plan, candidates, omitted_bounds
    )
    manifest = {
        "complete": True,
        "schema_version": SCHEMA_VERSION,
        "spec_id": plan["spec_id"],
        "activation_formula": ACTIVATION_FORMULA,
        "matrix_format": "csr",
        "compute_dtype": "float32",
        "value_dtype": "float32",
        "index_dtype": "int32",
        "indptr_dtype": "int32",
        "explicit_zero_slots": True,
        "explicit_zero_semantics": plan["explicit_zero_semantics"],
        "values_descending_within_row": True,
        "expected_shards": EXPECTED_SHARDS,
        "expected_rows": EXPECTED_ROWS,
        "models": plan["models"],
        "embeddings_dir": plan["embeddings_dir"],
        "datasets": plan["datasets"],
        "sample_validation": sample_validation,
        "shards": records,
        "completed_at": utc_now(),
    }
    manifest_path = args.cache_root / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    complete = {
        "complete": True,
        "schema_version": SCHEMA_VERSION,
        "spec_id": plan["spec_id"],
        "manifest": str(manifest_path.resolve()),
        "shards": len(records),
        "rows": sum(int(record["rows"]) for record in records),
        "sample_validation": sample_validation,
        "top_examples_report": str(
            (args.report_dir / "top_examples.json").resolve()
        ),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "completed_at": utc_now(),
    }
    atomic_write_json(args.cache_root / "COMPLETE.json", complete)
    atomic_write_json(args.report_dir / "metadata.json", {
        **complete,
        "cache_root": str(args.cache_root.resolve()),
        "activation_definition": ACTIVATION_FORMULA,
        "top_n": top_report["top_n"],
    })
    print(
        f"Published {len(records)} shards / {complete['rows']:,} rows "
        f"at {args.cache_root}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    prepare_parser.add_argument(
        "--models-dir", type=Path, default=_paper_path(get_path("models_dir"))
    )
    prepare_parser.add_argument(
        "--embeddings-dir", type=Path,
        default=_paper_path(get_path("gemini_embeddings_dir"))
    )
    prepare_parser.add_argument("--datasets", default=",".join(GENERAL_CORPUS_DATASETS))
    prepare_parser.add_argument("--num-tasks", type=int, default=16)
    prepare_parser.add_argument("--tracked-top-per-task", type=int, default=256)
    prepare_parser.set_defaults(func=prepare)

    compute_parser = subparsers.add_parser("compute")
    compute_parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    compute_parser.add_argument(
        "--embeddings-dir", type=Path,
        default=_paper_path(get_path("gemini_embeddings_dir"))
    )
    compute_parser.add_argument("--task-id", type=int, required=True)
    compute_parser.add_argument("--num-tasks", type=int, default=16)
    compute_parser.add_argument("--batch-size", type=int, default=1024)
    compute_parser.add_argument("--progress-every", type=int, default=2)
    compute_parser.add_argument(
        "--device", type=torch.device, default=torch.device("cuda")
    )
    compute_parser.set_defaults(func=compute)

    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    audit_parser.add_argument(
        "--models-dir", type=Path, default=_paper_path(get_path("models_dir"))
    )
    audit_parser.add_argument(
        "--embeddings-dir", type=Path,
        default=_paper_path(get_path("gemini_embeddings_dir"))
    )
    audit_parser.add_argument(
        "--datasets-dir", type=Path, default=_paper_path(get_path("datasets_dir"))
    )
    audit_parser.add_argument("--report-dir", type=Path, required=True)
    audit_parser.add_argument("--validation-samples", type=int, default=64)
    audit_parser.add_argument("--validation-seed", type=int, default=20260813)
    audit_parser.add_argument("--validation-atol", type=float, default=2e-4)
    audit_parser.add_argument("--validation-batch-size", type=int, default=32)
    audit_parser.add_argument("--top-n", type=int, default=20)
    audit_parser.add_argument(
        "--device", type=torch.device, default=torch.device("cuda")
    )
    audit_parser.set_defaults(func=audit)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    parsed.func(parsed)
