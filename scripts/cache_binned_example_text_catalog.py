#!/usr/bin/env python3
"""Resolve the deduplicated text catalog for binned random-example samples.

The numeric sampler deliberately stores corpus-global row ids rather than
duplicating text for every (model, feature, bin) sample.  This companion cache
resolves the union of those ids exactly once.  ``resolve`` is Slurm-array safe:
each task owns a contiguous range of source-cache shards and atomically writes
one sorted Parquet part.  ``merge`` verifies every part and exact coverage of
the numeric sample arrays before publishing an immutable catalog.  ``audit``
repeats the checks on an already-published catalog.
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
from typing import Any, Iterable, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(_paper_path(__file__).resolve().parents[1]))
from project_paths import get_path  # noqa: E402
from scripts.cache_binned_random_examples import (  # noqa: E402
    ARTIFACT_KIND as NUMERIC_ARTIFACT_KIND,
    MODEL_SPECS,
    read_json,
    sha256_file,
)
from scripts.sparse_activation_cache import load_manifest  # noqa: E402


SCHEMA_VERSION = 1
ARTIFACT_KIND = "gemini_binned_random_example_text_catalog"
DEFAULT_NUMERIC_CACHE = _paper_path(
    "/resources/activation_cache_dir/"
    "gemini_binned_random_examples_v1/final"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_NUMERIC_CACHE.parent / "text_catalog"
CATALOG_SCHEMA = pa.schema(
    [
        ("global_row", pa.int64()),
        ("shard", pa.string()),
        ("shard_row", pa.int64()),
        ("dataset", pa.string()),
        ("config", pa.string()),
        ("split", pa.string()),
        ("source_row", pa.int64()),
        ("text_column", pa.string()),
        ("full_text", pa.large_string()),
    ]
)


def load_from_disk(path: str) -> Any:
    """Import datasets lazily so structural/audit commands stay lightweight."""
    from datasets import load_from_disk as datasets_load_from_disk

    return datasets_load_from_disk(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def partition_bounds(length: int, task_index: int, task_count: int) -> tuple[int, int]:
    if task_count <= 0 or not 0 <= task_index < task_count:
        raise ValueError("Require task_count > 0 and 0 <= task_index < task_count")
    return length * task_index // task_count, length * (task_index + 1) // task_count


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _identity_sha(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _numeric_contract(numeric_cache: Path) -> tuple[dict[str, Any], str, Path, dict[str, Any], str]:
    numeric_manifest_path = numeric_cache / "manifest.json"
    numeric_sha = sha256_file(numeric_manifest_path)
    numeric = read_json(numeric_manifest_path, "numeric sample manifest")
    if numeric.get("complete") is not True or numeric.get("artifact_kind") != NUMERIC_ARTIFACT_KIND:
        raise ValueError(f"Not a complete binned-example cache: {numeric_cache}")
    complete = read_json(numeric_cache / "COMPLETE.json", "numeric completion marker")
    if complete.get("sampling_spec_sha256") != numeric.get("sampling_spec_sha256"):
        raise ValueError("Numeric completion marker disagrees with manifest")
    source_root = _paper_path(str(numeric["source_cache_root"])).resolve()
    source_manifest_path = source_root / "manifest.json"
    source_sha = sha256_file(source_manifest_path)
    if source_sha != numeric.get("source_manifest_sha256"):
        raise ValueError("Numeric cache source manifest has changed")
    source = load_manifest(source_root)
    if source.get("spec_id") != numeric.get("source_cache_spec_id"):
        raise ValueError("Numeric and source cache spec ids disagree")
    for name, spec in MODEL_SPECS.items():
        result = numeric.get("model_results", {}).get(name)
        if not isinstance(result, Mapping) or result.get("width") != spec["width"]:
            raise ValueError(f"Numeric cache is missing model contract {name}")
    return numeric, numeric_sha, source_root, source, source_sha


def catalog_identity(
    numeric_cache: Path,
    numeric: Mapping[str, Any],
    numeric_sha: str,
    source: Mapping[str, Any],
    source_sha: str,
    datasets_dir: Path,
    task_count: int,
    compression: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "numeric_cache": str(numeric_cache.resolve()),
        "numeric_manifest_sha256": numeric_sha,
        "sampling_spec_sha256": numeric["sampling_spec_sha256"],
        "source_cache_root": str(_paper_path(str(numeric["source_cache_root"])).resolve()),
        "source_cache_spec_id": source["spec_id"],
        "source_manifest_sha256": source_sha,
        "datasets_dir": str(datasets_dir.resolve()),
        "task_count": task_count,
        "parquet_compression": compression,
        "text_semantics": "full str(value).strip(); null/blank values become [EMPTY]",
        "deduplication_key": "corpus-global row across all models/features/bins",
    }


def _sample_rows_array(
    numeric_cache: Path,
    numeric: Mapping[str, Any],
    model_name: str,
    *,
    verify_hash: bool,
) -> np.ndarray:
    meta = numeric["model_results"][model_name]["arrays"]["sample_rows"]
    path = numeric_cache / model_name / str(meta["file"])
    if not path.is_file() or path.stat().st_size != meta.get("size_bytes"):
        raise ValueError(f"Missing or wrong-sized sample rows: {path}")
    if verify_hash and sha256_file(path) != meta.get("sha256"):
        raise ValueError(f"Checksum mismatch: {path}")
    array = np.load(_paper_location(path), mmap_mode="r", allow_pickle=False)
    expected = (
        int(MODEL_SPECS[model_name]["width"]),
        len(numeric["bin_labels"]),
        int(numeric["sample_count_per_feature_bin"]),
    )
    if array.shape != expected or array.dtype != np.int64:
        raise ValueError(f"Wrong sample-row array contract for {path}: {array.shape}/{array.dtype}")
    return array


def unique_sampled_rows(
    numeric_cache: Path,
    numeric: Mapping[str, Any],
    row_start: int,
    row_stop: int,
    *,
    verify_hash: bool = True,
    chunk_entries: int = 2_000_000,
) -> tuple[np.ndarray, int]:
    """Return sorted unique sampled rows in [row_start,row_stop), plus entry count."""
    if not 0 <= row_start <= row_stop <= int(numeric["corpus"]["rows"]):
        raise ValueError("Invalid corpus-global row interval")
    pieces: list[np.ndarray] = []
    matched_entries = 0
    for model_name in MODEL_SPECS:
        flat = _sample_rows_array(
            numeric_cache, numeric, model_name, verify_hash=verify_hash
        ).reshape(-1)
        for begin in range(0, flat.size, chunk_entries):
            chunk = np.asarray(flat[begin : begin + chunk_entries])
            selected = chunk[(chunk >= row_start) & (chunk < row_stop)]
            matched_entries += int(selected.size)
            if selected.size:
                pieces.append(np.unique(selected))
    if not pieces:
        return np.empty(0, dtype=np.int64), matched_entries
    return np.unique(np.concatenate(pieces)).astype(np.int64, copy=False), matched_entries


def int64_payload_sha(rows: np.ndarray) -> str:
    value = np.asarray(rows, dtype="<i8")
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _parse_location(relative_shard: str) -> tuple[str, str | None, str]:
    parts = _paper_path(relative_shard).parts
    if not parts:
        raise ValueError(f"Invalid relative shard: {relative_shard}")
    config_part = next((part for part in parts if part.startswith("config=")), None)
    split_parts = [part for part in parts if part.startswith("split=")]
    if len(split_parts) != 1:
        raise ValueError(f"Expected one split component: {relative_shard}")
    return (
        parts[0],
        config_part.split("=", 1)[1] if config_part else None,
        split_parts[0].split("=", 1)[1],
    )


class BulkTextResolver:
    def __init__(self, embeddings_dir: Path, datasets_dir: Path):
        self.embeddings_dir = embeddings_dir
        self.datasets_dir = datasets_dir
        self.dataset_cache: dict[str, Any] = {}

    def _dataset(self, dataset_name: str, config: str | None, split: str) -> Any:
        path = self.datasets_dir / dataset_name
        if config is not None:
            path = path / f"config={config}"
        key = str(path)
        if key not in self.dataset_cache:
            self.dataset_cache[key] = load_from_disk(key)
        loaded = self.dataset_cache[key]
        return loaded[split] if isinstance(loaded, Mapping) else loaded

    def resolve_batches(
        self,
        record: Mapping[str, Any],
        global_rows: np.ndarray,
        batch_rows: int,
    ) -> Iterable[pa.Table]:
        relative = str(record["relative_shard"])
        dataset_name, config, split = _parse_location(relative)
        shard_dir = self.embeddings_dir / relative
        metadata_path = shard_dir / "metadata.parquet"
        stat = metadata_path.stat()
        if stat.st_size != int(record["metadata_size_bytes"]):
            raise ValueError(f"Metadata size changed: {metadata_path}")
        expected_mtime = record.get("metadata_mtime_ns")
        if expected_mtime is not None and stat.st_mtime_ns != int(expected_mtime):
            raise ValueError(f"Metadata mtime changed: {metadata_path}")
        shard_meta_path = shard_dir / "shard_meta.json"
        if sha256_file(shard_meta_path) != record.get("shard_meta_sha256"):
            raise ValueError(f"Shard metadata checksum mismatch: {shard_meta_path}")
        shard_meta = read_json(shard_meta_path, "embedding shard metadata")
        text_column = str(shard_meta["text_column"])
        if text_column != str(record["text_column"]):
            raise ValueError(f"Text column disagreement: {relative}")
        row_map = pq.read_table(metadata_path, columns=["row_idx"])["row_idx"]
        if len(row_map) != int(record["rows"]):
            raise ValueError(f"Metadata row count disagreement: {relative}")
        dataset = self._dataset(dataset_name, config, split)
        local_all = global_rows - int(record["global_row_start"])
        if len(local_all) and (int(local_all.min()) < 0 or int(local_all.max()) >= len(row_map)):
            raise ValueError(f"Rows outside source shard: {relative}")
        for begin in range(0, len(global_rows), batch_rows):
            stop = min(len(global_rows), begin + batch_rows)
            global_batch = global_rows[begin:stop]
            local = local_all[begin:stop].astype(np.int64, copy=False)
            source_rows = row_map.take(pa.array(local)).to_numpy().astype(np.int64, copy=False)
            values = dataset[source_rows.tolist()][text_column]
            texts = [
                "[EMPTY]" if value is None or not str(value).strip() else str(value).strip()
                for value in values
            ]
            count = len(global_batch)
            yield pa.Table.from_arrays(
                [
                    pa.array(global_batch, type=pa.int64()),
                    pa.array([relative] * count, type=pa.string()),
                    pa.array(local, type=pa.int64()),
                    pa.array([dataset_name] * count, type=pa.string()),
                    pa.array([config] * count, type=pa.string()),
                    pa.array([split] * count, type=pa.string()),
                    pa.array(source_rows, type=pa.int64()),
                    pa.array([text_column] * count, type=pa.string()),
                    pa.array(texts, type=pa.large_string()),
                ],
                schema=CATALOG_SCHEMA,
            )


def _task_row_interval(records: list[Mapping[str, Any]], start: int, stop: int) -> tuple[int, int]:
    if start == stop:
        boundary = int(records[start]["global_row_start"]) if start < len(records) else int(records[-1]["global_row_stop"])
        return boundary, boundary
    return int(records[start]["global_row_start"]), int(records[stop - 1]["global_row_stop"])


def _validate_records(records: list[Mapping[str, Any]], expected_rows: int) -> None:
    cursor = 0
    for index, record in enumerate(records):
        start, stop = int(record["global_row_start"]), int(record["global_row_stop"])
        if start != cursor or stop - start != int(record["rows"]):
            raise ValueError(f"Non-contiguous source manifest at shard {index}")
        cursor = stop
    if cursor != expected_rows:
        raise ValueError(f"Source manifest covers {cursor}, expected {expected_rows}")


def resolve(args: argparse.Namespace) -> None:
    started = time.monotonic()
    numeric_cache, output_root = args.numeric_cache.resolve(), args.output_root.resolve()
    numeric, numeric_sha, source_root, source, source_sha = _numeric_contract(numeric_cache)
    records = list(source["shards"])
    _validate_records(records, int(source["expected_rows"]))
    identity = catalog_identity(
        numeric_cache,
        numeric,
        numeric_sha,
        source,
        source_sha,
        args.datasets_dir,
        args.task_count,
        args.compression,
    )
    spec_sha = _identity_sha(identity)
    start, stop = partition_bounds(len(records), args.task_index, args.task_count)
    row_start, row_stop = _task_row_interval(records, start, stop)
    destination = output_root / "partials" / f"task-{args.task_index:03d}-of-{args.task_count:03d}"
    if destination.exists():
        metadata = read_json(destination / "metadata.json", "existing catalog partial")
        if metadata.get("complete") is True and metadata.get("catalog_spec_sha256") == spec_sha and metadata.get("task_index") == args.task_index:
            part = destination / str(metadata["part"]["file"])
            if part.stat().st_size == metadata["part"]["size_bytes"] and sha256_file(part) == metadata["part"]["sha256"]:
                print(f"Compatible, checksum-verified text partial exists: {destination}")
                return
        raise RuntimeError(f"Refusing to replace incompatible partial: {destination}")
    rows, sampled_entries = unique_sampled_rows(
        numeric_cache, numeric, row_start, row_stop, verify_hash=True, chunk_entries=args.chunk_entries
    )
    output_root.joinpath("partials").mkdir(parents=True, exist_ok=True)
    staging = output_root / "partials" / f".tmp-{destination.name}-{uuid.uuid4().hex}"
    staging.mkdir()
    part_path = staging / f"part-{args.task_index:03d}.parquet"
    writer = pq.ParquetWriter(part_path, CATALOG_SCHEMA, compression=args.compression)
    resolver = BulkTextResolver(_paper_path(str(source["embeddings_dir"])), args.datasets_dir.resolve())
    shard_results: list[dict[str, Any]] = []
    written = 0
    try:
        for offset, record in enumerate(records[start:stop]):
            low = np.searchsorted(rows, int(record["global_row_start"]), side="left")
            high = np.searchsorted(rows, int(record["global_row_stop"]), side="left")
            shard_rows = rows[low:high]
            if len(shard_rows):
                for table in resolver.resolve_batches(record, shard_rows, args.batch_rows):
                    writer.write_table(table, row_group_size=args.batch_rows)
                    written += len(table)
            shard_results.append(
                {
                    "manifest_shard_index": start + offset,
                    "relative_shard": record["relative_shard"],
                    "resolved_rows": int(len(shard_rows)),
                }
            )
            if (offset + 1) % args.progress_every == 0 or offset + 1 == stop - start:
                print(f"task {args.task_index}: {offset + 1}/{stop - start} shards, {written:,} unique rows", flush=True)
        if written == 0:
            writer.write_table(pa.Table.from_batches([], schema=CATALOG_SCHEMA))
        writer.close()
        writer = None
        if written != len(rows):
            raise RuntimeError(f"Wrote {written} rows, expected {len(rows)}")
        part_meta = {
            "file": part_path.name,
            "rows": written,
            "size_bytes": part_path.stat().st_size,
            "sha256": sha256_file(part_path),
            "global_rows_sha256": int64_payload_sha(rows),
            "global_row_min": int(rows[0]) if len(rows) else None,
            "global_row_max": int(rows[-1]) if len(rows) else None,
        }
        metadata = {
            "complete": True,
            **identity,
            "catalog_spec_sha256": spec_sha,
            "task_index": args.task_index,
            "shard_index_start": start,
            "shard_index_stop": stop,
            "global_row_start": row_start,
            "global_row_stop": row_stop,
            "sample_entries_in_range": sampled_entries,
            "unique_global_rows": len(rows),
            "part": part_meta,
            "shards": shard_results,
            "producer": {
                "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
                "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                "hostname": socket.gethostname(),
                "elapsed_seconds": time.monotonic() - started,
            },
            "completed_at": utc_now(),
        }
        _write_json(staging / "metadata.json", metadata)
        os.replace(staging, destination)
    except BaseException:
        if writer is not None:
            writer.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"Published text partial: {destination}", flush=True)


def _validate_part(path: Path, metadata: Mapping[str, Any]) -> np.ndarray:
    if not path.is_file() or path.stat().st_size != metadata.get("size_bytes"):
        raise ValueError(f"Missing or wrong-sized Parquet part: {path}")
    if sha256_file(path) != metadata.get("sha256"):
        raise ValueError(f"Parquet checksum mismatch: {path}")
    schema = pq.read_schema(path)
    if not schema.equals(CATALOG_SCHEMA, check_metadata=False):
        raise ValueError(f"Wrong Parquet schema: {path}")
    rows = pq.read_table(path, columns=["global_row"])["global_row"].to_numpy()
    rows = np.asarray(rows, dtype=np.int64)
    if len(rows) != metadata.get("rows") or (len(rows) > 1 and np.any(rows[1:] <= rows[:-1])):
        raise ValueError(f"Parquet rows are not strictly increasing: {path}")
    if int64_payload_sha(rows) != metadata.get("global_rows_sha256"):
        raise ValueError(f"Global-row checksum mismatch: {path}")
    return rows


def _load_expected_rows(numeric_cache: Path, numeric: Mapping[str, Any]) -> tuple[np.ndarray, int]:
    return unique_sampled_rows(
        numeric_cache,
        numeric,
        0,
        int(numeric["corpus"]["rows"]),
        verify_hash=True,
    )


def _validate_partial_metadata(
    metadata: Mapping[str, Any], identity: Mapping[str, Any], spec_sha: str, task_index: int, bounds: tuple[int, int]
) -> None:
    for key, value in identity.items():
        if metadata.get(key) != value:
            raise ValueError(f"Partial {task_index} differs on identity field {key}")
    if metadata.get("complete") is not True or metadata.get("catalog_spec_sha256") != spec_sha or metadata.get("task_index") != task_index or metadata.get("shard_index_start") != bounds[0] or metadata.get("shard_index_stop") != bounds[1]:
        raise ValueError(f"Invalid text partial metadata: task {task_index}")


def _audit_catalog_parts(parts: list[tuple[Path, Mapping[str, Any]]], expected: np.ndarray) -> dict[str, Any]:
    digest = hashlib.sha256()
    previous: int | None = None
    observed_count = 0
    for path, metadata in parts:
        rows = _validate_part(path, metadata)
        if len(rows) and previous is not None and int(rows[0]) <= previous:
            raise ValueError("Duplicate or unordered global rows across Parquet parts")
        if len(rows):
            previous = int(rows[-1])
        digest.update(np.asarray(rows, dtype="<i8").tobytes(order="C"))
        observed_count += len(rows)
    expected_sha = int64_payload_sha(expected)
    if observed_count != len(expected) or digest.hexdigest() != expected_sha:
        raise ValueError("Text catalog does not exactly cover the numeric sampled-row union")
    return {
        "unique_global_rows": len(expected),
        "global_rows_sha256": expected_sha,
        "strictly_sorted_unique_across_parts": True,
        "exact_numeric_sample_union_coverage": True,
        "all_parquet_sha256_verified": True,
    }


def merge(args: argparse.Namespace) -> None:
    started = time.monotonic()
    numeric_cache, output_root = args.numeric_cache.resolve(), args.output_root.resolve()
    numeric, numeric_sha, _, source, source_sha = _numeric_contract(numeric_cache)
    records = list(source["shards"])
    _validate_records(records, int(source["expected_rows"]))
    identity = catalog_identity(
        numeric_cache,
        numeric,
        numeric_sha,
        source,
        source_sha,
        args.datasets_dir,
        args.task_count,
        args.compression,
    )
    spec_sha = _identity_sha(identity)
    destination = output_root / "final"
    if destination.exists():
        _audit_published(destination, numeric_cache)
        existing = read_json(destination / "manifest.json", "published text manifest")
        if existing.get("catalog_spec_sha256") != spec_sha:
            raise RuntimeError(f"Existing catalog is incompatible: {destination}")
        print(f"Compatible, fully audited text catalog exists: {destination}")
        return
    partials: list[tuple[Path, dict[str, Any]]] = []
    for task_index in range(args.task_count):
        partial_dir = output_root / "partials" / f"task-{task_index:03d}-of-{args.task_count:03d}"
        metadata = read_json(partial_dir / "metadata.json", "text partial metadata")
        bounds = partition_bounds(len(records), task_index, args.task_count)
        _validate_partial_metadata(metadata, identity, spec_sha, task_index, bounds)
        partials.append((partial_dir / str(metadata["part"]["file"]), metadata))
    expected, sampled_entries = _load_expected_rows(numeric_cache, numeric)
    audit_result = _audit_catalog_parts([(path, meta["part"]) for path, meta in partials], expected)
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".tmp-final-{uuid.uuid4().hex}"
    parts_dir = staging / "parts"
    parts_dir.mkdir(parents=True)
    final_parts: list[dict[str, Any]] = []
    try:
        for task_index, (source_path, metadata) in enumerate(partials):
            target = parts_dir / f"part-{task_index:03d}.parquet"
            try:
                os.link(source_path, target)
            except OSError:
                shutil.copy2(source_path, target)
            item = dict(metadata["part"])
            item["file"] = f"parts/{target.name}"
            if target.stat().st_size != item["size_bytes"] or sha256_file(target) != item["sha256"]:
                raise RuntimeError(f"Published part changed while copying: {source_path}")
            final_parts.append(item)
        manifest = {
            "complete": True,
            **identity,
            "catalog_spec_sha256": spec_sha,
            "columns": [{"name": field.name, "type": str(field.type), "nullable": field.nullable} for field in CATALOG_SCHEMA],
            "parts": final_parts,
            "numeric_sample_entries": sampled_entries,
            "audit": audit_result,
            "producer": {
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "hostname": socket.gethostname(),
                "elapsed_seconds": time.monotonic() - started,
            },
            "completed_at": utc_now(),
        }
        _write_json(staging / "manifest.json", manifest)
        _write_json(staging / "COMPLETE.json", {"complete": True, "artifact_kind": ARTIFACT_KIND, "catalog_spec_sha256": spec_sha, "manifest": "manifest.json", "completed_at": manifest["completed_at"]})
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"Published text catalog: {destination}", flush=True)


def _audit_published(catalog: Path, numeric_cache: Path) -> dict[str, Any]:
    numeric, numeric_sha, _, source, source_sha = _numeric_contract(numeric_cache)
    manifest = read_json(catalog / "manifest.json", "text catalog manifest")
    if manifest.get("complete") is not True or manifest.get("artifact_kind") != ARTIFACT_KIND:
        raise ValueError(f"Not a complete text catalog: {catalog}")
    identity = catalog_identity(
        numeric_cache,
        numeric,
        numeric_sha,
        source,
        source_sha,
        _paper_path(str(manifest["datasets_dir"])),
        int(manifest["task_count"]),
        str(manifest["parquet_compression"]),
    )
    if manifest.get("catalog_spec_sha256") != _identity_sha(identity):
        raise ValueError("Published text catalog identity mismatch")
    marker = read_json(catalog / "COMPLETE.json", "text catalog completion marker")
    if marker.get("catalog_spec_sha256") != manifest["catalog_spec_sha256"]:
        raise ValueError("Text catalog completion marker mismatch")
    if len(manifest.get("parts", [])) != int(manifest["task_count"]):
        raise ValueError("Text catalog part count disagrees with task count")
    expected, sampled_entries = _load_expected_rows(numeric_cache, numeric)
    parts = [(catalog / str(item["file"]), item) for item in manifest["parts"]]
    result = _audit_catalog_parts(parts, expected)
    if manifest.get("numeric_sample_entries") != sampled_entries or manifest.get("audit") != result:
        raise ValueError("Published audit summary disagrees with recomputed audit")
    return result


def audit(args: argparse.Namespace) -> None:
    result = _audit_published(args.catalog.resolve(), args.numeric_cache.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    resolve_parser = commands.add_parser("resolve")
    resolve_parser.add_argument("--numeric-cache", type=Path, default=DEFAULT_NUMERIC_CACHE)
    resolve_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    resolve_parser.add_argument("--datasets-dir", type=Path, default=_paper_path(get_path("datasets_dir")))
    resolve_parser.add_argument("--task-index", type=int, required=True)
    resolve_parser.add_argument("--task-count", type=int, default=32)
    resolve_parser.add_argument("--batch-rows", type=int, default=4096)
    resolve_parser.add_argument("--chunk-entries", type=int, default=2_000_000)
    resolve_parser.add_argument("--progress-every", type=int, default=5)
    resolve_parser.add_argument("--compression", default="zstd")
    resolve_parser.set_defaults(func=resolve)
    merge_parser = commands.add_parser("merge")
    merge_parser.add_argument("--numeric-cache", type=Path, default=DEFAULT_NUMERIC_CACHE)
    merge_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    merge_parser.add_argument("--datasets-dir", type=Path, default=_paper_path(get_path("datasets_dir")))
    merge_parser.add_argument("--task-count", type=int, default=32)
    merge_parser.add_argument("--compression", default="zstd")
    merge_parser.set_defaults(func=merge)
    audit_parser = commands.add_parser("audit")
    audit_parser.add_argument("--numeric-cache", type=Path, default=DEFAULT_NUMERIC_CACHE)
    audit_parser.add_argument("--catalog", type=Path, default=DEFAULT_OUTPUT_ROOT / "final")
    audit_parser.set_defaults(func=audit)
    return result


def main() -> None:
    args = parser().parse_args()
    if hasattr(args, "task_count") and args.task_count <= 0:
        raise ValueError("task-count must be positive")
    if getattr(args, "batch_rows", 1) <= 0 or getattr(args, "chunk_entries", 1) <= 0:
        raise ValueError("batch-rows and chunk-entries must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
