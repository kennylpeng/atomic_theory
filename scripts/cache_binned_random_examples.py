#!/usr/bin/env python3
"""Count and deterministically sample binned activations from a sparse SAE cache.

The ``scan`` command is intended for a Slurm array.  Every task scans a
disjoint, contiguous range of cache shards and writes exact counts plus a
mergeable bottom-priority reservoir.  ``merge`` validates all partials,
combines them, audits every retained sample, and atomically publishes a compact
cache.  A sample is a corpus-global row and its exact cached activation; text
can therefore be resolved later without duplicating corpus strings millions of
times.

For each (feature, bin) activation occurrence, a deterministic SplitMix64 hash
of ``(seed, model, feature, global_row)`` is used as its random priority.  The
20 smallest priorities are retained.  This is reproducible, partition
independent, and implements uniform sampling without replacement under the
pseudorandom-priority model.
"""

from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import argparse
import hashlib
import json
import os
import shutil
import socket
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numba
import numpy as np

sys.path.insert(0, str(_paper_path(__file__).resolve().parents[1]))
from scripts.cache_payload import (  # noqa: E402
    _load_shard_marker,
    _open_fixed_topk_payload,
)
from scripts.sparse_activation_cache import load_manifest  # noqa: E402


SCHEMA_VERSION = 1
ARTIFACT_KIND = "gemini_post_topk_binned_random_examples"
DEFAULT_CACHE_ROOT = _paper_path(
    "/resources/activation_cache_dir/"
    "gemini_all_corpus_post_topk"
)
DEFAULT_OUTPUT_ROOT = _paper_path(
    "/resources/activation_cache_dir/"
    "gemini_binned_random_examples_v1"
)
MODEL_SPECS = {
    "gemini_m4096_k32": {
        "width": 4096,
        "top_k": 32,
        "salt": 0x409620260813,
        "expected_positive_slots": 2_874_481_856,
    },
    "gemini_m131072_k128": {
        "width": 131072,
        "top_k": 128,
        "salt": 0x13107220260813,
        "expected_positive_slots": 11_488_330_818,
    },
}
ROW_ID_BITS = 27
BIN_UPPER_BOUNDS = np.asarray(
    [0.05, 0.10, 0.15, 0.20, 0.25, 0.30], dtype=np.float32
)
BIN_LABELS = (
    "(0,0.05]",
    "(0.05,0.1]",
    "(0.1,0.15]",
    "(0.15,0.2]",
    "(0.2,0.25]",
    "(0.25,0.3]",
    "(0.3,inf)",
)
ARRAY_FILES = {
    "bin_counts": (np.dtype(np.uint64), 2),
    "sample_rows": (np.dtype(np.int64), 3),
    "sample_activations": (np.dtype(np.float32), 3),
    "sample_priorities": (np.dtype(np.uint64), 3),
    "sample_sizes": (np.dtype(np.uint8), 2),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {label}: {path}")
    return value


def _write_npy(path: Path, value: np.ndarray) -> None:
    with path.open("wb") as handle:
        np.save(_paper_location(handle), value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def partition_bounds(length: int, task_index: int, task_count: int) -> tuple[int, int]:
    if task_count <= 0 or not 0 <= task_index < task_count:
        raise ValueError("Require task_count > 0 and 0 <= task_index < task_count")
    return length * task_index // task_count, length * (task_index + 1) // task_count


def artifact_identity(
    cache_root: Path,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    sample_count: int,
    seed: int,
    task_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "source_cache_root": str(cache_root.resolve()),
        "source_cache_spec_id": manifest["spec_id"],
        "source_manifest_sha256": manifest_sha256,
        "models": [
            {"name": name, "width": spec["width"], "top_k": spec["top_k"]}
            for name, spec in MODEL_SPECS.items()
        ],
        "bin_upper_bounds_float32": [float(value) for value in BIN_UPPER_BOUNDS],
        "bin_labels": list(BIN_LABELS),
        "interval_convention": "right-closed except final open-ended bin",
        "sample_count_per_feature_bin": sample_count,
        "sampling_seed": seed,
        "sampling_method": (
            "20 lowest SplitMix64(seed xor model_salt xor "
            "((feature_id << 27) | global_row)) priorities per feature/bin"
        ).replace("20", str(sample_count), 1),
        "task_count": task_count,
    }


def identity_sha256(identity: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@numba.njit(cache=True, inline="always")
def _splitmix64(value: np.uint64) -> np.uint64:
    value = value + np.uint64(0x9E3779B97F4A7C15)
    value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return value ^ (value >> np.uint64(31))


@numba.njit(cache=True, inline="always")
def _priority(seed: np.uint64, salt: np.uint64, feature: int, row: int) -> np.uint64:
    # Injective because the published corpus has <2**27 rows and <=2**17 features.
    occurrence = (np.uint64(feature) << np.uint64(ROW_ID_BITS)) | np.uint64(row)
    return _splitmix64(seed ^ salt ^ occurrence)


@numba.njit(cache=True, inline="always")
def _activation_bin(value: np.float32, edges: np.ndarray) -> int:
    for index in range(edges.shape[0]):
        if value <= edges[index]:
            return index
    return edges.shape[0]


@numba.njit(cache=True, inline="always")
def _recompute_maximum(
    cell: int,
    sample_rows: np.ndarray,
    priorities: np.ndarray,
    sample_count: int,
) -> tuple[np.uint64, int]:
    maximum = priorities[cell, 0]
    maximum_row = sample_rows[cell, 0]
    maximum_slot = 0
    for slot in range(1, sample_count):
        candidate = priorities[cell, slot]
        candidate_row = sample_rows[cell, slot]
        if candidate > maximum or (
            candidate == maximum and candidate_row > maximum_row
        ):
            maximum = candidate
            maximum_row = candidate_row
            maximum_slot = slot
    return maximum, maximum_slot


@numba.njit(cache=True)
def scan_payload_chunk(
    indices: np.ndarray,
    activations: np.ndarray,
    top_k: int,
    global_row_start: int,
    seed: np.uint64,
    salt: np.uint64,
    edges: np.ndarray,
    counts: np.ndarray,
    sample_rows: np.ndarray,
    sample_activations: np.ndarray,
    priorities: np.ndarray,
    sizes: np.ndarray,
    maximum_priorities: np.ndarray,
    maximum_slots: np.ndarray,
) -> int:
    """Consume complete fixed-TopK rows; return the number of positive slots."""

    sample_count = sample_rows.shape[1]
    positive = 0
    for position in range(indices.shape[0]):
        activation = activations[position]
        if activation <= np.float32(0.0):
            continue
        positive += 1
        feature = int(indices[position])
        row = global_row_start + position // top_k
        bin_index = _activation_bin(activation, edges)
        cell = feature * (edges.shape[0] + 1) + bin_index
        counts[cell] += np.uint64(1)
        priority = _priority(seed, salt, feature, row)
        size = int(sizes[cell])
        if size < sample_count:
            sample_rows[cell, size] = row
            sample_activations[cell, size] = activation
            priorities[cell, size] = priority
            size += 1
            sizes[cell] = size
            if size == sample_count:
                maximum, maximum_slot = _recompute_maximum(
                    cell, sample_rows, priorities, sample_count
                )
                maximum_priorities[cell] = maximum
                maximum_slots[cell] = maximum_slot
        else:
            maximum = maximum_priorities[cell]
            maximum_slot = int(maximum_slots[cell])
            maximum_row = sample_rows[cell, maximum_slot]
            if priority < maximum or (priority == maximum and row < maximum_row):
                sample_rows[cell, maximum_slot] = row
                sample_activations[cell, maximum_slot] = activation
                priorities[cell, maximum_slot] = priority
                maximum, maximum_slot = _recompute_maximum(
                    cell, sample_rows, priorities, sample_count
                )
                maximum_priorities[cell] = maximum
                maximum_slots[cell] = maximum_slot
    return positive


@numba.njit(cache=True)
def merge_samples_inplace(
    incoming_rows: np.ndarray,
    incoming_activations: np.ndarray,
    incoming_priorities: np.ndarray,
    incoming_sizes: np.ndarray,
    rows: np.ndarray,
    activations: np.ndarray,
    priorities: np.ndarray,
    sizes: np.ndarray,
    maximum_priorities: np.ndarray,
    maximum_slots: np.ndarray,
) -> None:
    sample_count = rows.shape[1]
    for cell in range(rows.shape[0]):
        for incoming_slot in range(int(incoming_sizes[cell])):
            row = incoming_rows[cell, incoming_slot]
            activation = incoming_activations[cell, incoming_slot]
            priority = incoming_priorities[cell, incoming_slot]
            size = int(sizes[cell])
            if size < sample_count:
                rows[cell, size] = row
                activations[cell, size] = activation
                priorities[cell, size] = priority
                size += 1
                sizes[cell] = size
                if size == sample_count:
                    maximum, maximum_slot = _recompute_maximum(
                        cell, rows, priorities, sample_count
                    )
                    maximum_priorities[cell] = maximum
                    maximum_slots[cell] = maximum_slot
            else:
                maximum = maximum_priorities[cell]
                maximum_slot = int(maximum_slots[cell])
                maximum_row = rows[cell, maximum_slot]
                if priority < maximum or (
                    priority == maximum and row < maximum_row
                ):
                    rows[cell, maximum_slot] = row
                    activations[cell, maximum_slot] = activation
                    priorities[cell, maximum_slot] = priority
                    maximum, maximum_slot = _recompute_maximum(
                        cell, rows, priorities, sample_count
                    )
                    maximum_priorities[cell] = maximum
                    maximum_slots[cell] = maximum_slot


@numba.njit(cache=True)
def sort_samples_inplace(
    rows: np.ndarray,
    activations: np.ndarray,
    priorities: np.ndarray,
    sizes: np.ndarray,
) -> None:
    """Sort each valid reservoir prefix by ``(priority, global_row)``."""

    for cell in range(rows.shape[0]):
        size = int(sizes[cell])
        for slot in range(1, size):
            priority = priorities[cell, slot]
            row = rows[cell, slot]
            activation = activations[cell, slot]
            destination = slot
            while destination > 0:
                previous_priority = priorities[cell, destination - 1]
                previous_row = rows[cell, destination - 1]
                if previous_priority < priority or (
                    previous_priority == priority and previous_row <= row
                ):
                    break
                priorities[cell, destination] = previous_priority
                rows[cell, destination] = previous_row
                activations[cell, destination] = activations[cell, destination - 1]
                destination -= 1
            priorities[cell, destination] = priority
            rows[cell, destination] = row
            activations[cell, destination] = activation


@numba.njit(cache=True)
def audit_samples_compiled(
    counts: np.ndarray,
    rows: np.ndarray,
    activations: np.ndarray,
    priorities: np.ndarray,
    sizes: np.ndarray,
    expected_rows: int,
    seed: np.uint64,
    salt: np.uint64,
    edges: np.ndarray,
) -> tuple[int, int, int, int, int, int]:
    """Return ``(error, cell, slot, entries, nonempty, full)``."""

    sample_count = rows.shape[1]
    bins = edges.shape[0] + 1
    valid_count = 0
    nonempty = 0
    full = 0
    for cell in range(rows.shape[0]):
        count = counts[cell]
        expected_size = sample_count if count >= sample_count else int(count)
        size = int(sizes[cell])
        if size != expected_size:
            return 1, cell, -1, valid_count, nonempty, full
        if count > 0:
            nonempty += 1
        if count >= sample_count:
            full += 1
        feature = cell // bins
        bin_index = cell % bins
        for slot in range(sample_count):
            row = rows[cell, slot]
            activation = activations[cell, slot]
            priority = priorities[cell, slot]
            if slot < size:
                if row < 0 or row >= expected_rows:
                    return 2, cell, slot, valid_count, nonempty, full
                if not np.isfinite(activation) or activation <= np.float32(0.0):
                    return 3, cell, slot, valid_count, nonempty, full
                if _activation_bin(activation, edges) != bin_index:
                    return 4, cell, slot, valid_count, nonempty, full
                if priority != _priority(seed, salt, feature, row):
                    return 5, cell, slot, valid_count, nonempty, full
                # Packed occurrence IDs and SplitMix's permutation make valid
                # priorities unique for a feature.  Strict order therefore
                # checks deterministic ordering and duplicate rows together.
                if slot > 0 and priority <= priorities[cell, slot - 1]:
                    return 6, cell, slot, valid_count, nonempty, full
                valid_count += 1
            else:
                if row != -1:
                    return 7, cell, slot, valid_count, nonempty, full
                if not np.isnan(activation):
                    return 8, cell, slot, valid_count, nonempty, full
                if priority != np.iinfo(np.uint64).max:
                    return 9, cell, slot, valid_count, nonempty, full
    return 0, -1, -1, valid_count, nonempty, full


def empty_reservoir(width: int, sample_count: int) -> dict[str, np.ndarray]:
    cells = width * len(BIN_LABELS)
    return {
        "bin_counts": np.zeros(cells, dtype=np.uint64),
        "sample_rows": np.full((cells, sample_count), -1, dtype=np.int64),
        "sample_activations": np.full(
            (cells, sample_count), np.nan, dtype=np.float32
        ),
        "sample_priorities": np.full(
            (cells, sample_count), np.iinfo(np.uint64).max, dtype=np.uint64
        ),
        "sample_sizes": np.zeros(cells, dtype=np.uint8),
        "maximum_priorities": np.full(
            cells, np.iinfo(np.uint64).max, dtype=np.uint64
        ),
        "maximum_slots": np.zeros(cells, dtype=np.uint8),
    }


def _array_metadata(path: Path, array: np.ndarray) -> dict[str, Any]:
    return {
        "file": path.name,
        "shape": list(array.shape),
        "dtype": array.dtype.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_model_arrays(
    directory: Path, reservoir: Mapping[str, np.ndarray], width: int
) -> dict[str, Any]:
    directory.mkdir(parents=True)
    result: dict[str, Any] = {}
    bins = len(BIN_LABELS)
    for name in ARRAY_FILES:
        array = reservoir[name]
        if array.ndim == 1:
            shaped = array.reshape(width, bins)
        else:
            shaped = array.reshape(width, bins, array.shape[-1])
        path = directory / f"{name}.npy"
        _write_npy(path, shaped)
        result[name] = _array_metadata(path, shaped)
    return result


def validate_and_open_arrays(
    directory: Path,
    metadata: Mapping[str, Any],
    width: int,
    sample_count: int,
    *,
    verify_hash: bool = True,
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    expected_shapes = {
        "bin_counts": (width, len(BIN_LABELS)),
        "sample_sizes": (width, len(BIN_LABELS)),
        "sample_rows": (width, len(BIN_LABELS), sample_count),
        "sample_activations": (width, len(BIN_LABELS), sample_count),
        "sample_priorities": (width, len(BIN_LABELS), sample_count),
    }
    for name, (dtype, _) in ARRAY_FILES.items():
        item = metadata.get(name)
        if not isinstance(item, Mapping) or item.get("file") != f"{name}.npy":
            raise ValueError(f"Invalid metadata for {directory}/{name}")
        path = directory / str(item["file"])
        if not path.is_file() or path.stat().st_size != item.get("size_bytes"):
            raise ValueError(f"Missing or wrong-sized array: {path}")
        if verify_hash and sha256_file(path) != item.get("sha256"):
            raise ValueError(f"Checksum mismatch: {path}")
        array = np.load(_paper_location(path), mmap_mode="r", allow_pickle=False)
        if array.shape != expected_shapes[name] or array.dtype != dtype:
            raise ValueError(
                f"Wrong array contract for {path}: {array.shape}, {array.dtype}"
            )
        if item.get("shape") != list(array.shape) or item.get("dtype") != dtype.name:
            raise ValueError(f"Metadata disagrees with array: {path}")
        arrays[name] = array
    return arrays


def _publish_directory(staging: Path, destination: Path) -> None:
    if destination.exists():
        raise RuntimeError(f"Refusing to replace existing artifact: {destination}")
    os.replace(staging, destination)


def _compatible_metadata(path: Path, spec_sha256: str, task_index: int | None) -> bool:
    try:
        metadata = read_json(path, "existing artifact metadata")
    except ValueError:
        return False
    return bool(
        metadata.get("complete") is True
        and metadata.get("artifact_kind") == ARTIFACT_KIND
        and metadata.get("sampling_spec_sha256") == spec_sha256
        and (task_index is None or metadata.get("task_index") == task_index)
    )


def _component_payload_sha(output: Mapping[str, Any], component: str) -> str:
    try:
        value = output["components"][component]["payload_sha256"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Missing declared payload checksum for {component}") from exc
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"Invalid declared payload checksum for {component}")
    return value


def scan(args: argparse.Namespace) -> None:
    started = time.monotonic()
    if args.sample_count <= 0 or args.sample_count > 255:
        raise ValueError("sample-count must be in [1,255]")
    if args.chunk_rows <= 0:
        raise ValueError("chunk-rows must be positive")
    cache_root = args.cache_root.resolve()
    output_root = args.output_root.resolve()
    manifest_path = cache_root / "manifest.json"
    manifest_sha = sha256_file(manifest_path)
    manifest = load_manifest(cache_root)
    if int(manifest["expected_rows"]) >= 1 << ROW_ID_BITS:
        raise ValueError(
            f"Corpus has too many rows for {ROW_ID_BITS}-bit packed row IDs"
        )
    identity = artifact_identity(
        cache_root, manifest, manifest_sha, args.sample_count, args.seed, args.task_count
    )
    spec_sha = identity_sha256(identity)
    destination = (
        output_root / "partials" / f"task-{args.task_index:03d}-of-{args.task_count:03d}"
    )
    if destination.exists():
        metadata_path = destination / "metadata.json"
        if _compatible_metadata(metadata_path, spec_sha, args.task_index):
            metadata = read_json(metadata_path, "existing partial metadata")
            for model_name, spec in MODEL_SPECS.items():
                validate_and_open_arrays(
                    destination / model_name,
                    metadata["model_results"][model_name]["arrays"],
                    spec["width"],
                    args.sample_count,
                )
            print(f"Compatible, checksum-verified partial already exists: {destination}")
            return
        raise RuntimeError(f"Refusing to replace incompatible partial: {destination}")

    records = list(manifest["shards"])
    shard_start, shard_stop = partition_bounds(
        len(records), args.task_index, args.task_count
    )
    selected = records[shard_start:shard_stop]
    partials_dir = output_root / "partials"
    partials_dir.mkdir(parents=True, exist_ok=True)
    staging = partials_dir / f".tmp-{destination.name}-{uuid.uuid4().hex}"
    staging.mkdir()
    model_results: dict[str, Any] = {}
    try:
        for model_name, model_spec in MODEL_SPECS.items():
            width = int(model_spec["width"])
            top_k = int(model_spec["top_k"])
            reservoir = empty_reservoir(width, args.sample_count)
            source_components = []
            payload_bytes = 0
            positive_slots = 0
            for local_index, record in enumerate(selected):
                shard_dir, marker = _load_shard_marker(cache_root, manifest, record)
                indices, data, output = _open_fixed_topk_payload(
                    shard_dir, record, marker, model_name
                )
                indices_digest = hashlib.sha256()
                data_digest = hashlib.sha256()
                rows = int(record["rows"])
                for row_start in range(0, rows, args.chunk_rows):
                    row_stop = min(rows, row_start + args.chunk_rows)
                    begin, end = row_start * top_k, row_stop * top_k
                    index_chunk = np.asarray(indices[begin:end])
                    data_chunk = np.asarray(data[begin:end])
                    indices_digest.update(index_chunk.tobytes(order="C"))
                    data_digest.update(data_chunk.tobytes(order="C"))
                    payload_bytes += index_chunk.nbytes + data_chunk.nbytes
                    if len(index_chunk) and (
                        int(index_chunk.min()) < 0 or int(index_chunk.max()) >= width
                    ):
                        raise ValueError(
                            f"Out-of-range feature in {record['relative_shard']}/{model_name}"
                        )
                    if not np.isfinite(data_chunk).all() or (data_chunk < 0).any():
                        raise ValueError(
                            f"Invalid activation in {record['relative_shard']}/{model_name}"
                        )
                    positive_slots += scan_payload_chunk(
                        index_chunk,
                        data_chunk,
                        top_k,
                        int(record["global_row_start"]) + row_start,
                        np.uint64(args.seed),
                        np.uint64(model_spec["salt"]),
                        BIN_UPPER_BOUNDS,
                        reservoir["bin_counts"],
                        reservoir["sample_rows"],
                        reservoir["sample_activations"],
                        reservoir["sample_priorities"],
                        reservoir["sample_sizes"],
                        reservoir["maximum_priorities"],
                        reservoir["maximum_slots"],
                    )
                actual_indices_sha = indices_digest.hexdigest()
                actual_data_sha = data_digest.hexdigest()
                if actual_indices_sha != _component_payload_sha(output, "indices"):
                    raise ValueError(
                        f"Indices checksum mismatch: {record['relative_shard']}/{model_name}"
                    )
                if actual_data_sha != _component_payload_sha(output, "data"):
                    raise ValueError(
                        f"Data checksum mismatch: {record['relative_shard']}/{model_name}"
                    )
                source_components.append(
                    {
                        "manifest_shard_index": shard_start + local_index,
                        "relative_shard": record["relative_shard"],
                        "rows": rows,
                        "global_row_start": record["global_row_start"],
                        "global_row_stop": record["global_row_stop"],
                        "indices_payload_sha256": actual_indices_sha,
                        "data_payload_sha256": actual_data_sha,
                    }
                )
                completed = local_index + 1
                if completed % args.progress_every == 0 or completed == len(selected):
                    print(
                        f"task {args.task_index} {model_name}: {completed}/{len(selected)} "
                        f"shards, {payload_bytes / 2**30:.2f} GiB validated",
                        flush=True,
                    )
            if int(reservoir["bin_counts"].sum(dtype=np.uint64)) != positive_slots:
                raise RuntimeError(f"Count conservation failed for {model_name}")
            arrays = write_model_arrays(staging / model_name, reservoir, width)
            model_results[model_name] = {
                "width": width,
                "top_k": top_k,
                "positive_slots": positive_slots,
                "bin_totals": [
                    int(value)
                    for value in reservoir["bin_counts"]
                    .reshape(width, len(BIN_LABELS))
                    .sum(axis=0, dtype=np.uint64)
                ],
                "payload_bytes_validated": payload_bytes,
                "source_components": source_components,
                "arrays": arrays,
            }
            del reservoir
        if sha256_file(manifest_path) != manifest_sha:
            raise RuntimeError("Source manifest changed during scan")
        metadata = {
            "complete": True,
            **identity,
            "sampling_spec_sha256": spec_sha,
            "task_index": args.task_index,
            "shard_index_start": shard_start,
            "shard_index_stop": shard_stop,
            "shard_count": len(selected),
            "model_results": model_results,
            "validation": {
                "source_payload_sha256_verified": True,
                "indices_in_bounds": True,
                "activations_finite_nonnegative": True,
                "positive_count_conservation": True,
            },
            "producer": {
                "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
                "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                "hostname": socket.gethostname(),
                "numpy_version": np.__version__,
                "numba_version": numba.__version__,
                "elapsed_seconds": time.monotonic() - started,
            },
            "completed_at": utc_now(),
        }
        _write_json(staging / "metadata.json", metadata)
        _publish_directory(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"Published partial: {destination}", flush=True)


def _validate_partial_identity(
    metadata: Mapping[str, Any],
    identity: Mapping[str, Any],
    spec_sha: str,
    task_index: int,
    shard_bounds: tuple[int, int],
) -> None:
    for key, value in identity.items():
        if metadata.get(key) != value:
            raise ValueError(f"Partial {task_index} differs on identity field {key}")
    if (
        metadata.get("complete") is not True
        or metadata.get("sampling_spec_sha256") != spec_sha
        or metadata.get("task_index") != task_index
        or metadata.get("shard_index_start") != shard_bounds[0]
        or metadata.get("shard_index_stop") != shard_bounds[1]
    ):
        raise ValueError(f"Invalid partition metadata for partial {task_index}")


def _audit_samples(
    model_name: str,
    reservoir: Mapping[str, np.ndarray],
    width: int,
    sample_count: int,
    expected_rows: int,
    seed: int,
    salt: int,
) -> dict[str, Any]:
    result = audit_samples_compiled(
        reservoir["bin_counts"],
        reservoir["sample_rows"],
        reservoir["sample_activations"],
        reservoir["sample_priorities"],
        reservoir["sample_sizes"],
        expected_rows,
        np.uint64(seed),
        np.uint64(salt),
        BIN_UPPER_BOUNDS,
    )
    error, cell, slot, valid_count, nonempty, full = result
    if error:
        feature, bin_index = divmod(cell, len(BIN_LABELS))
        labels = {
            1: "sample size differs from min(count, sample_count)",
            2: "sample row is outside the corpus",
            3: "sample activation is not finite and positive",
            4: "sample activation is in the wrong bin",
            5: "sample priority does not recompute",
            6: "sample order is not strictly increasing or row is duplicated",
            7: "sample row padding is not -1",
            8: "sample activation padding is not NaN",
            9: "sample priority padding is not UINT64_MAX",
        }
        raise ValueError(
            f"Audit failed for {model_name}/feature={feature}/bin={bin_index}/"
            f"slot={slot}: {labels[error]}"
        )
    return {
        "sample_entries": valid_count,
        "nonempty_feature_bins": nonempty,
        "fully_sampled_feature_bins": full,
        "sample_sizes_equal_min_counts": True,
        "sample_rows_unique_within_feature_bin": True,
        "sample_rows_in_corpus_range": True,
        "sample_activations_match_bins": True,
        "sample_priorities_recomputed": True,
        "samples_sorted_by_priority_then_global_row": True,
        "padding_verified": True,
    }


def merge(args: argparse.Namespace) -> None:
    started = time.monotonic()
    cache_root = args.cache_root.resolve()
    output_root = args.output_root.resolve()
    manifest_path = cache_root / "manifest.json"
    manifest_sha = sha256_file(manifest_path)
    manifest = load_manifest(cache_root)
    if int(manifest["expected_rows"]) >= 1 << ROW_ID_BITS:
        raise ValueError(
            f"Corpus has too many rows for {ROW_ID_BITS}-bit packed row IDs"
        )
    identity = artifact_identity(
        cache_root, manifest, manifest_sha, args.sample_count, args.seed, args.task_count
    )
    spec_sha = identity_sha256(identity)
    destination = output_root / "final"
    if destination.exists():
        metadata_path = destination / "manifest.json"
        if _compatible_metadata(metadata_path, spec_sha, None):
            metadata = read_json(metadata_path, "published manifest")
            for model_name, spec in MODEL_SPECS.items():
                validate_and_open_arrays(
                    destination / model_name,
                    metadata["model_results"][model_name]["arrays"],
                    spec["width"],
                    args.sample_count,
                )
            print(f"Compatible, checksum-verified final cache already exists: {destination}")
            return
        raise RuntimeError(f"Refusing to replace incompatible final cache: {destination}")

    partial_metadata: list[dict[str, Any]] = []
    records = list(manifest["shards"])
    for task_index in range(args.task_count):
        partial_dir = (
            output_root
            / "partials"
            / f"task-{task_index:03d}-of-{args.task_count:03d}"
        )
        metadata = read_json(partial_dir / "metadata.json", "partial metadata")
        bounds = partition_bounds(len(records), task_index, args.task_count)
        _validate_partial_identity(metadata, identity, spec_sha, task_index, bounds)
        partial_metadata.append(metadata)

    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".tmp-final-{uuid.uuid4().hex}"
    staging.mkdir()
    model_results: dict[str, Any] = {}
    try:
        for model_name, model_spec in MODEL_SPECS.items():
            width = int(model_spec["width"])
            reservoir = empty_reservoir(width, args.sample_count)
            expected_source_index = 0
            positive_slots = 0
            payload_bytes = 0
            for task_index, metadata in enumerate(partial_metadata):
                partial_dir = (
                    output_root
                    / "partials"
                    / f"task-{task_index:03d}-of-{args.task_count:03d}"
                )
                result = metadata["model_results"].get(model_name)
                if not isinstance(result, Mapping):
                    raise ValueError(f"Partial {task_index} is missing {model_name}")
                arrays = validate_and_open_arrays(
                    partial_dir / model_name,
                    result["arrays"],
                    width,
                    args.sample_count,
                )
                sources = result.get("source_components")
                if not isinstance(sources, list):
                    raise ValueError(f"Missing source attestations in partial {task_index}")
                for source in sources:
                    record = records[expected_source_index]
                    output = record["outputs"][model_name]
                    if (
                        source.get("manifest_shard_index") != expected_source_index
                        or source.get("relative_shard") != record["relative_shard"]
                        or source.get("global_row_start") != record["global_row_start"]
                        or source.get("global_row_stop") != record["global_row_stop"]
                        or source.get("indices_payload_sha256")
                        != _component_payload_sha(output, "indices")
                        or source.get("data_payload_sha256")
                        != _component_payload_sha(output, "data")
                    ):
                        raise ValueError(
                            f"Bad source attestation at shard {expected_source_index}/{model_name}"
                        )
                    expected_source_index += 1
                incoming_counts = np.asarray(arrays["bin_counts"]).reshape(-1)
                maximum = np.iinfo(np.uint64).max
                if np.any(reservoir["bin_counts"] > maximum - incoming_counts):
                    raise OverflowError(f"Count overflow while merging {model_name}")
                reservoir["bin_counts"] += incoming_counts
                merge_samples_inplace(
                    np.asarray(arrays["sample_rows"]).reshape(-1, args.sample_count),
                    np.asarray(arrays["sample_activations"]).reshape(
                        -1, args.sample_count
                    ),
                    np.asarray(arrays["sample_priorities"]).reshape(
                        -1, args.sample_count
                    ),
                    np.asarray(arrays["sample_sizes"]).reshape(-1),
                    reservoir["sample_rows"],
                    reservoir["sample_activations"],
                    reservoir["sample_priorities"],
                    reservoir["sample_sizes"],
                    reservoir["maximum_priorities"],
                    reservoir["maximum_slots"],
                )
                positive_slots += int(result["positive_slots"])
                payload_bytes += int(result["payload_bytes_validated"])
            if expected_source_index != len(records):
                raise ValueError(
                    f"Source coverage for {model_name} is {expected_source_index}/{len(records)}"
                )
            total_count = int(reservoir["bin_counts"].sum(dtype=np.uint64))
            if total_count != positive_slots:
                raise ValueError(f"Merged count conservation failed for {model_name}")
            expected_positive = int(model_spec["expected_positive_slots"])
            if manifest["spec_id"] == (
                "8ea0dbadf8b45bd6180a37d15eb00a00202381131c6a4b82926a3624fa8ff155"
            ) and total_count != expected_positive:
                raise ValueError(
                    f"Published-cache positive total for {model_name} is "
                    f"{total_count}, expected {expected_positive}"
                )
            sort_samples_inplace(
                reservoir["sample_rows"],
                reservoir["sample_activations"],
                reservoir["sample_priorities"],
                reservoir["sample_sizes"],
            )
            audit = _audit_samples(
                model_name,
                reservoir,
                width,
                args.sample_count,
                int(manifest["expected_rows"]),
                args.seed,
                int(model_spec["salt"]),
            )
            arrays = write_model_arrays(staging / model_name, reservoir, width)
            bin_counts = reservoir["bin_counts"].reshape(width, len(BIN_LABELS))
            model_results[model_name] = {
                "width": width,
                "top_k": model_spec["top_k"],
                "positive_slots": positive_slots,
                "expected_positive_slots_for_source_cache": expected_positive,
                "bin_totals": [
                    int(value) for value in bin_counts.sum(axis=0, dtype=np.uint64)
                ],
                "payload_bytes_validated": payload_bytes,
                "source_shards_verified": len(records),
                "arrays": arrays,
                "audit": audit,
            }
            del reservoir
            print(f"Merged and audited {model_name}", flush=True)
        if sha256_file(manifest_path) != manifest_sha:
            raise RuntimeError("Source manifest changed during merge")
        metadata = {
            "complete": True,
            **identity,
            "sampling_spec_sha256": spec_sha,
            "corpus": {
                "rows": manifest["expected_rows"],
                "shards": manifest["expected_shards"],
                "datasets": manifest.get("datasets"),
            },
            "model_results": model_results,
            "data_contract": {
                "bin_counts": "exact positive post-TopK activation occurrences",
                "sample_rows": "corpus-global row; -1 pads unused sample slots",
                "sample_activations": "exact cached float32 activation; NaN padding",
                "sample_priorities": "deterministic uint64 priority; UINT64_MAX padding",
                "sample_sizes": "valid prefix length on the sample axis",
                "sample_axis_order": (
                    "valid prefix sorted lexicographically by "
                    "(sample_priority, global_row), not activation"
                ),
            },
            "validation": {
                "all_array_sha256_recorded": True,
                "all_partial_array_sha256_verified": True,
                "all_source_payload_sha256_verified_by_scan": True,
                "all_source_attestations_bound_to_manifest": True,
                "all_shards_exactly_once": True,
                "count_conservation": True,
                "sampling_audited": True,
            },
            "producer": {
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "hostname": socket.gethostname(),
                "numpy_version": np.__version__,
                "numba_version": numba.__version__,
                "elapsed_seconds": time.monotonic() - started,
            },
            "completed_at": utc_now(),
        }
        _write_json(staging / "manifest.json", metadata)
        _write_json(
            staging / "COMPLETE.json",
            {
                "complete": True,
                "schema_version": SCHEMA_VERSION,
                "artifact_kind": ARTIFACT_KIND,
                "manifest": "manifest.json",
                "sampling_spec_sha256": spec_sha,
                "source_cache_spec_id": manifest["spec_id"],
                "completed_at": metadata["completed_at"],
            },
        )
        _publish_directory(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"Published final binned-example cache: {destination}", flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    for command in ("scan", "merge"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
        sub.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
        sub.add_argument("--sample-count", type=int, default=20)
        sub.add_argument("--seed", type=int, default=20260813)
        sub.add_argument("--task-count", type=int, default=8)
        if command == "scan":
            sub.add_argument("--task-index", type=int, required=True)
            sub.add_argument("--chunk-rows", type=int, default=8192)
            sub.add_argument("--progress-every", type=int, default=10)
            sub.set_defaults(func=scan)
        else:
            sub.set_defaults(func=merge)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.seed < 0 or args.seed > np.iinfo(np.uint64).max:
        raise ValueError("seed must fit in uint64")
    if args.task_count <= 0:
        raise ValueError("task-count must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
