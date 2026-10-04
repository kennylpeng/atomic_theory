#!/usr/bin/env python3
"""Publish a validated one-model manifest view over the Nemotron TopK cache."""

from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np

MODEL_NAME = "nemotron_m131072_k128"
WIDTH = 131072
TOP_K = 128
DEFAULT_SOURCE_ROOT = _paper_path(
    "/resources/activation_cache_dir/"
    "nemotron_all_experiment_saes_post_topk"
)
DEFAULT_VIEW_ROOT = _paper_path(
    "/resources/activation_cache_dir/"
    "nemotron_m131072_post_topk_view_v1"
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_output(
    source_root: Path,
    relative_shard: str,
    rows: int,
    output: Mapping[str, Any],
) -> None:
    if output.get("shape") != [rows, WIDTH] or output.get("top_k") != TOP_K:
        raise ValueError(f"Wrong target shape/top_k: {relative_shard}")
    expected = {
        "indptr": ((rows + 1,), np.dtype(np.int32)),
        "indices": ((rows * TOP_K,), np.dtype(np.int32)),
        "data": ((rows * TOP_K,), np.dtype(np.float32)),
    }
    for component, (shape, dtype) in expected.items():
        metadata = output.get("components", {}).get(component)
        if not isinstance(metadata, Mapping):
            raise ValueError(f"Missing {component} metadata: {relative_shard}")
        path = source_root / "shards" / relative_shard / MODEL_NAME / f"{component}.npy"
        array = np.load(_paper_location(path), mmap_mode="r", allow_pickle=False)
        if array.shape != shape or array.dtype != dtype:
            raise ValueError(
                f"Wrong {component} payload: {relative_shard}: "
                f"{array.shape}/{array.dtype}"
            )
        if metadata.get("shape") != list(shape) or metadata.get("dtype") != dtype.name:
            raise ValueError(f"Wrong {component} contract: {relative_shard}")
        if int(metadata.get("size_bytes", -1)) != path.stat().st_size:
            raise ValueError(f"Wrong {component} byte size: {relative_shard}")
        declared = metadata.get("file")
        if declared is not None and _paper_path(str(declared)).name != path.name:
            raise ValueError(f"Wrong {component} filename: {relative_shard}")
        if component != "indptr":
            checksum = metadata.get("payload_sha256")
            if not isinstance(checksum, str) or len(checksum) != 64:
                raise ValueError(f"Missing {component} checksum: {relative_shard}")


def build(source_root: Path, view_root: Path) -> None:
    source_root = source_root.resolve()
    plan_path = source_root / "plan.json"
    plan = read_json(plan_path)
    if plan.get("complete") is not True:
        raise ValueError(f"Incomplete source plan: {plan_path}")
    model_records = [
        model for model in plan.get("models", []) if model.get("name") == MODEL_NAME
    ]
    if len(model_records) != 1:
        raise ValueError(f"Expected one {MODEL_NAME} record")
    records = []
    positive_slots = 0
    for index, source in enumerate(plan["source_shards"]):
        relative = str(source["relative_shard"])
        rows = int(source["rows"])
        marker_path = source_root / "shards" / relative / "complete.json"
        marker = read_json(marker_path)
        marker_source = marker.get("source")
        if (
            marker.get("complete") is not True
            or marker.get("spec_id") != plan.get("spec_id")
            or not isinstance(marker_source, Mapping)
            or marker_source.get("relative_shard") != relative
            or int(marker_source.get("rows", -1)) != rows
        ):
            raise ValueError(f"Invalid shard marker: {relative}")
        output = marker.get("outputs", {}).get(MODEL_NAME)
        if not isinstance(output, Mapping):
            raise ValueError(f"Missing target output: {relative}")
        validate_output(source_root, relative, rows, output)
        positive_slots += int(output["positive_nnz"])
        records.append(
            {
                **source,
                "cache_shard_dir": str(
                    (view_root / "shards" / relative).resolve()
                ),
                "complete_metadata": str(marker_path.resolve()),
                "outputs": {MODEL_NAME: output},
                "manifest_shard_index": index,
            }
        )
        if (index + 1) % 100 == 0 or index + 1 == len(plan["source_shards"]):
            print(f"Validated {index + 1}/{len(plan['source_shards'])} shards", flush=True)

    rows = sum(int(record["rows"]) for record in records)
    if rows != int(plan["expected_rows"]) or len(records) != int(plan["expected_shards"]):
        raise ValueError("Source plan totals disagree with shard records")
    manifest = {
        "complete": True,
        "schema_version": 1,
        "spec_id": plan["spec_id"],
        "activation_formula": plan["activation_formula"],
        "matrix_format": plan["matrix_format"],
        "compute_dtype": plan["compute_dtype"],
        "value_dtype": plan["value_dtype"],
        "index_dtype": plan["index_dtype"],
        "indptr_dtype": plan["indptr_dtype"],
        "explicit_zero_slots": plan["explicit_zero_slots"],
        "explicit_zero_semantics": plan["explicit_zero_semantics"],
        "values_descending_within_row": plan["values_descending_within_row"],
        "expected_shards": len(records),
        "expected_rows": rows,
        "models": model_records,
        "embeddings_dir": plan["embeddings_dir"],
        "datasets": plan["datasets"],
        "shards": records,
        "view": {
            "kind": "single_model_symlink_view",
            "source_root": str(source_root),
            "source_plan": str(plan_path),
            "source_plan_sha256": sha256_file(plan_path),
            "target_model": MODEL_NAME,
            "target_positive_slots": positive_slots,
        },
    }

    if view_root.exists():
        existing_path = view_root / "manifest.json"
        existing = read_json(existing_path)
        comparable = lambda value: {
            "spec_id": value.get("spec_id"),
            "models": value.get("models"),
            "expected_rows": value.get("expected_rows"),
            "expected_shards": value.get("expected_shards"),
            "view": value.get("view"),
        }
        if comparable(existing) != comparable(manifest):
            raise RuntimeError(f"Incompatible existing view: {view_root}")
        print(f"Compatible view already exists: {view_root}")
        return

    view_root.mkdir(parents=True)
    os.symlink(source_root / "shards", view_root / "shards", target_is_directory=True)
    atomic_json(view_root / "manifest.json", manifest)
    atomic_json(
        view_root / "COMPLETE.json",
        {
            "complete": True,
            "schema_version": 1,
            "spec_id": plan["spec_id"],
            "manifest": str((view_root / "manifest.json").resolve()),
            "shards": len(records),
            "rows": rows,
            "view_source_root": str(source_root),
            "target_model": MODEL_NAME,
        },
    )
    print(
        f"Published view for {len(records)} shards / {rows:,} rows / "
        f"{positive_slots:,} positive slots: {view_root}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--view-root", type=Path, default=DEFAULT_VIEW_ROOT)
    args = parser.parse_args()
    build(args.source_root, args.view_root)


if __name__ == "__main__":
    main()
