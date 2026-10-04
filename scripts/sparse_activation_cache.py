"""Read shard-aligned sparse SAE activation caches.

The cache stores one CSR matrix per source embedding shard and SAE model::

    CACHE_ROOT/
      manifest.json
      COMPLETE.json
      shards/<relative_shard>/
        complete.json
        <model_name>/
          indptr.npy
          indices.npy
          data.npy

Rows in each matrix are local to the source shard.  ``manifest.json`` assigns
contiguous global row offsets so callers can either stream shard matrices or
look up one corpus-wide row with :func:`get_row`.
"""

from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import bisect
import json
from collections.abc import Iterator, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from scipy import sparse


MANIFEST_FILENAME = "manifest.json"
ROOT_COMPLETE_FILENAME = "COMPLETE.json"
SHARD_COMPLETE_FILENAME = "complete.json"
COMPONENT_DTYPES = {
    "indptr": np.dtype(np.int32),
    "indices": np.dtype(np.int32),
    "data": np.dtype(np.float32),
}
MANIFEST_ROW_COUNT_ALIASES = ("rows", "total_rows", "expected_rows")
MANIFEST_SHARD_COUNT_ALIASES = (
    "shard_count",
    "total_shards",
    "expected_shards",
)
COMPLETION_ROW_COUNT_ALIASES = ("rows", "total_rows", "expected_rows")
COMPLETION_SHARD_COUNT_ALIASES = (
    "shards",
    "shard_count",
    "total_shards",
    "expected_shards",
)


class SparseActivationCacheError(ValueError):
    """The cache exists but does not satisfy the sparse-cache contract."""


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SparseActivationCacheError(f"Invalid {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SparseActivationCacheError(f"Expected an object in {label}: {path}")
    return payload


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    # bool is an int subclass, but accepting it in row counts/offsets obscures
    # malformed JSON metadata.
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SparseActivationCacheError(
            f"{label} must be an integer >= {minimum}, got {value!r}"
        )
    return value


def _declared_count(
    payload: Mapping[str, Any],
    aliases: tuple[str, ...],
    label: str,
    *,
    required: bool = True,
) -> int | None:
    """Read a count while accepting schema aliases and rejecting disagreement."""

    declared = {
        alias: _require_int(payload[alias], f"{label}.{alias}")
        for alias in aliases
        if alias in payload
    }
    if not declared:
        if required:
            raise SparseActivationCacheError(
                f"{label} is missing; expected one of {aliases}"
            )
        return None
    values = set(declared.values())
    if len(values) != 1:
        raise SparseActivationCacheError(
            f"Disagreeing aliases for {label}: {declared}"
        )
    return next(iter(values))


def _relative_shard(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise SparseActivationCacheError("relative_shard must be a nonempty string")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise SparseActivationCacheError(f"Unsafe relative_shard: {value!r}")
    return path.as_posix()


def _expected_shard_dir(cache_root: Path, relative_shard: str) -> Path:
    return cache_root / "shards" / _paper_path(*PurePosixPath(relative_shard).parts)


def _metadata_path(cache_root: Path, value: str | Path) -> Path:
    path = _paper_path(value)
    return path if path.is_absolute() else cache_root / path


def _validate_declared_paths(
    cache_root: Path, record: Mapping[str, Any], relative_shard: str
) -> None:
    expected_dir = _expected_shard_dir(cache_root, relative_shard)
    declared_dir = record.get("cache_shard_dir")
    if not isinstance(declared_dir, str) or not declared_dir:
        raise SparseActivationCacheError(
            f"Manifest shard {relative_shard!r} has no cache_shard_dir"
        )
    if _metadata_path(cache_root, declared_dir).resolve() != expected_dir.resolve():
        raise SparseActivationCacheError(
            f"cache_shard_dir for {relative_shard!r} does not match {expected_dir}"
        )

    complete_metadata = record.get("complete_metadata")
    if isinstance(complete_metadata, str):
        expected_complete = expected_dir / SHARD_COMPLETE_FILENAME
        if (
            _metadata_path(cache_root, complete_metadata).resolve()
            != expected_complete.resolve()
        ):
            raise SparseActivationCacheError(
                f"complete_metadata for {relative_shard!r} does not match "
                f"{expected_complete}"
            )
    elif isinstance(complete_metadata, Mapping):
        if complete_metadata.get("complete") is not True:
            raise SparseActivationCacheError(
                f"Embedded complete_metadata for {relative_shard!r} is incomplete"
            )
    else:
        raise SparseActivationCacheError(
            f"Manifest shard {relative_shard!r} has invalid complete_metadata"
        )


def _record_offsets(record: Mapping[str, Any], relative_shard: str) -> tuple[int, int]:
    try:
        start_value = record["global_row_start"]
        stop_value = record["global_row_stop"]
    except KeyError as exc:
        raise SparseActivationCacheError(
            f"Manifest shard {relative_shard!r} is missing global row offsets"
        ) from exc
    start = _require_int(start_value, f"{relative_shard}.global_row_start")
    stop = _require_int(stop_value, f"{relative_shard}.global_row_stop")
    if stop < start:
        raise SparseActivationCacheError(
            f"Global row stop precedes start for {relative_shard!r}"
        )
    return start, stop


def _manifest_model_names(manifest: Mapping[str, Any]) -> frozenset[str]:
    models = manifest.get("models")
    if not isinstance(models, list) or not models:
        raise SparseActivationCacheError("manifest.models must be a nonempty list")
    names = []
    for position, model in enumerate(models):
        name = model if isinstance(model, str) else (
            model.get("name") if isinstance(model, Mapping) else None
        )
        if not isinstance(name, str) or not name or _paper_path(name).name != name:
            raise SparseActivationCacheError(
                f"manifest.models[{position}] has invalid name {name!r}"
            )
        names.append(name)
    if len(names) != len(set(names)):
        raise SparseActivationCacheError("manifest.models contains duplicate names")
    return frozenset(names)


def _validate_manifest(cache_root: Path, manifest: dict[str, Any]) -> None:
    model_names = _manifest_model_names(manifest)
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise SparseActivationCacheError("manifest.shards must be a list")

    expected_start = 0
    seen: set[str] = set()
    for position, record in enumerate(shards):
        if not isinstance(record, dict):
            raise SparseActivationCacheError(
                f"manifest.shards[{position}] must be an object"
            )
        relative_shard = _relative_shard(record.get("relative_shard"))
        if relative_shard in seen:
            raise SparseActivationCacheError(
                f"Duplicate manifest relative_shard: {relative_shard}"
            )
        seen.add(relative_shard)

        rows = _require_int(record.get("rows"), f"{relative_shard}.rows")
        start, stop = _record_offsets(record, relative_shard)
        if start != expected_start:
            raise SparseActivationCacheError(
                f"Global row ranges are not contiguous at {relative_shard!r}: "
                f"expected start {expected_start}, got {start}"
            )
        if stop - start != rows:
            raise SparseActivationCacheError(
                f"Global row range for {relative_shard!r} has {stop - start} rows, "
                f"metadata declares {rows}"
            )
        expected_start = stop

        _validate_declared_paths(cache_root, record, relative_shard)
        outputs = record.get("outputs")
        if not isinstance(outputs, dict):
            raise SparseActivationCacheError(
                f"Manifest shard {relative_shard!r} has no outputs mapping"
            )
        missing = model_names - outputs.keys()
        if missing:
            raise SparseActivationCacheError(
                f"Manifest shard {relative_shard!r} is missing outputs: {sorted(missing)}"
            )

    declared_rows = _declared_count(
        manifest, MANIFEST_ROW_COUNT_ALIASES, "manifest row count"
    )
    if declared_rows != expected_start:
        raise SparseActivationCacheError(
            f"Manifest total rows {declared_rows} does not match shard total "
            f"{expected_start}"
        )

    declared_shards = _declared_count(
        manifest, MANIFEST_SHARD_COUNT_ALIASES, "manifest shard count"
    )
    if declared_shards != len(shards):
        raise SparseActivationCacheError(
            f"Manifest shard count {declared_shards} does not match shard list "
            f"length {len(shards)}"
        )


def _validate_root_completion(
    completion: Mapping[str, Any], manifest: Mapping[str, Any]
) -> None:
    """Cross-check root publication metadata against the manifest."""

    manifest_rows = _declared_count(
        manifest, MANIFEST_ROW_COUNT_ALIASES, "manifest row count"
    )
    completion_rows = _declared_count(
        completion, COMPLETION_ROW_COUNT_ALIASES, "completion row count"
    )
    if completion_rows != manifest_rows:
        raise SparseActivationCacheError(
            f"Completion row count {completion_rows} does not match manifest "
            f"row count {manifest_rows}"
        )

    manifest_shards = _declared_count(
        manifest, MANIFEST_SHARD_COUNT_ALIASES, "manifest shard count"
    )
    completion_shards = _declared_count(
        completion, COMPLETION_SHARD_COUNT_ALIASES, "completion shard count"
    )
    if completion_shards != manifest_shards:
        raise SparseActivationCacheError(
            f"Completion shard count {completion_shards} does not match manifest "
            f"shard count {manifest_shards}"
        )

    manifest_spec_id = manifest.get("spec_id")
    completion_spec_id = completion.get("spec_id")
    if not isinstance(manifest_spec_id, str) or not manifest_spec_id:
        raise SparseActivationCacheError("manifest.spec_id must be a nonempty string")
    if not isinstance(completion_spec_id, str) or not completion_spec_id:
        raise SparseActivationCacheError("completion.spec_id must be a nonempty string")
    if completion_spec_id != manifest_spec_id:
        raise SparseActivationCacheError(
            f"Completion spec_id {completion_spec_id!r} does not match manifest "
            f"spec_id {manifest_spec_id!r}"
        )


def load_manifest(cache_root: str | Path) -> dict[str, Any]:
    """Load and validate the completed cache manifest.

    A root cache is readable only after ``COMPLETE.json`` exists and marks the
    cache complete.  File component validation is deferred until the relevant
    shard is opened.
    """

    root = _paper_path(cache_root)
    completion = _read_json(root / ROOT_COMPLETE_FILENAME, "cache completion marker")
    if completion.get("complete") is not True:
        raise SparseActivationCacheError(
            f"Cache completion marker is not complete: {root / ROOT_COMPLETE_FILENAME}"
        )
    manifest = _read_json(root / MANIFEST_FILENAME, "cache manifest")
    _validate_manifest(root, manifest)
    _validate_root_completion(completion, manifest)
    return manifest


def _manifest_record(
    manifest: Mapping[str, Any], relative_shard: str
) -> dict[str, Any]:
    target = _relative_shard(relative_shard)
    for record in manifest["shards"]:
        if record["relative_shard"] == target:
            return record
    raise KeyError(f"Unknown relative_shard: {target}")


def _component_metadata(
    output: Mapping[str, Any], component: str, relative_shard: str, model_name: str
) -> Mapping[str, Any]:
    components = output.get("components")
    if not isinstance(components, Mapping):
        raise SparseActivationCacheError(
            f"{relative_shard}/{model_name} has no components mapping"
        )
    metadata = components.get(component)
    if not isinstance(metadata, Mapping):
        raise SparseActivationCacheError(
            f"{relative_shard}/{model_name} has invalid {component} metadata"
        )
    return metadata


def _validate_component_metadata(
    metadata: Mapping[str, Any],
    component: str,
    expected_length: int,
    relative_shard: str,
    model_name: str,
) -> None:
    label = f"{relative_shard}/{model_name}/{component}"
    declared_dtype = metadata.get("dtype")
    if declared_dtype != COMPONENT_DTYPES[component].name:
        raise SparseActivationCacheError(
            f"{label} metadata dtype must be {COMPONENT_DTYPES[component].name}, "
            f"got {declared_dtype!r}"
        )
    declared_shape = metadata.get("shape")
    if declared_shape != [expected_length]:
        raise SparseActivationCacheError(
            f"{label} metadata shape must be [{expected_length}], got "
            f"{declared_shape!r}"
        )
    declared_file = metadata.get("file")
    if declared_file is not None and _paper_path(str(declared_file)).name != f"{component}.npy":
        raise SparseActivationCacheError(
            f"{label} metadata names the wrong file: {declared_file!r}"
        )


def _shape_and_top_k(
    output: Mapping[str, Any], relative_shard: str, model_name: str
) -> tuple[tuple[int, int], int]:
    shape = output.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in shape)
        or any(value < 0 for value in shape)
    ):
        raise SparseActivationCacheError(
            f"Invalid shape for {relative_shard}/{model_name}: {shape!r}"
        )
    top_k = _require_int(
        output.get("top_k"), f"{relative_shard}/{model_name}.top_k", minimum=1
    )
    if top_k > shape[1]:
        raise SparseActivationCacheError(
            f"top_k exceeds width for {relative_shard}/{model_name}"
        )
    return (shape[0], shape[1]), top_k


def _load_shard_csr_from_manifest(
    cache_root: Path,
    manifest: Mapping[str, Any],
    relative_shard: str,
    model_name: str,
    mmap_mode: str | None,
) -> sparse.csr_matrix:
    model_names = _manifest_model_names(manifest)
    if model_name not in model_names:
        raise KeyError(
            f"Unsupported model_name {model_name!r}; expected one of "
            f"{sorted(model_names)}"
        )
    record = _manifest_record(manifest, relative_shard)
    relative_shard = record["relative_shard"]
    shard_dir = _expected_shard_dir(cache_root, relative_shard)
    complete = _read_json(
        shard_dir / SHARD_COMPLETE_FILENAME, "shard completion metadata"
    )
    if complete.get("complete") is not True:
        raise SparseActivationCacheError(
            f"Shard is not complete: {relative_shard}"
        )
    source = complete.get("source")
    nested_relative_shard = (
        source.get("relative_shard") if isinstance(source, Mapping) else None
    )
    completed_relative_shard = complete.get(
        "relative_shard", nested_relative_shard
    )
    if completed_relative_shard != relative_shard:
        raise SparseActivationCacheError(
            f"Shard completion metadata identifies {completed_relative_shard!r}, "
            f"expected {relative_shard!r}"
        )
    outputs = complete.get("outputs")
    if not isinstance(outputs, Mapping) or not isinstance(
        outputs.get(model_name), Mapping
    ):
        raise SparseActivationCacheError(
            f"Shard completion metadata has no output for {model_name}: "
            f"{relative_shard}"
        )
    output = outputs[model_name]
    shape, top_k = _shape_and_top_k(output, relative_shard, model_name)
    if shape[0] != record["rows"]:
        raise SparseActivationCacheError(
            f"CSR row count for {relative_shard}/{model_name} is {shape[0]}, "
            f"manifest declares {record['rows']}"
        )

    manifest_output = record["outputs"].get(model_name)
    if not isinstance(manifest_output, Mapping):
        raise SparseActivationCacheError(
            f"Manifest output for {relative_shard}/{model_name} is invalid"
        )
    manifest_shape, manifest_top_k = _shape_and_top_k(
        manifest_output, relative_shard, model_name
    )
    if manifest_shape != shape or manifest_top_k != top_k:
        raise SparseActivationCacheError(
            f"Manifest and shard metadata disagree for {relative_shard}/{model_name}"
        )

    model_dir = shard_dir / model_name
    arrays: dict[str, np.ndarray] = {}
    for component, expected_dtype in COMPONENT_DTYPES.items():
        path = model_dir / f"{component}.npy"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing CSR component for {relative_shard}/{model_name}: {path}"
            )
        try:
            array = np.load(_paper_location(path), mmap_mode=mmap_mode, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise SparseActivationCacheError(
                f"Cannot load CSR component {path}: {exc}"
            ) from exc
        if array.ndim != 1 or array.dtype != expected_dtype:
            raise SparseActivationCacheError(
                f"CSR component {path} must be 1D {expected_dtype.name}, got "
                f"shape={array.shape}, dtype={array.dtype}"
            )
        arrays[component] = array

    indptr = arrays["indptr"]
    indices = arrays["indices"]
    data = arrays["data"]
    if len(indptr) != shape[0] + 1:
        raise SparseActivationCacheError(
            f"indptr length for {relative_shard}/{model_name} must be "
            f"{shape[0] + 1}, got {len(indptr)}"
        )
    if len(indices) != len(data):
        raise SparseActivationCacheError(
            f"indices/data lengths differ for {relative_shard}/{model_name}"
        )

    for component, expected_length in (
        ("indptr", len(indptr)),
        ("indices", len(indices)),
        ("data", len(data)),
    ):
        _validate_component_metadata(
            _component_metadata(output, component, relative_shard, model_name),
            component,
            expected_length,
            relative_shard,
            model_name,
        )

    if int(indptr[0]) != 0 or int(indptr[-1]) != len(indices):
        raise SparseActivationCacheError(
            f"Invalid indptr endpoints for {relative_shard}/{model_name}"
        )
    if np.any(indptr[1:] < indptr[:-1]):
        raise SparseActivationCacheError(
            f"indptr is not monotonic for {relative_shard}/{model_name}"
        )
    row_counts = np.diff(indptr)
    if np.any(row_counts > top_k):
        raise SparseActivationCacheError(
            f"A row exceeds top_k={top_k} for {relative_shard}/{model_name}"
        )
    if len(indices) and (int(indices.min()) < 0 or int(indices.max()) >= shape[1]):
        raise SparseActivationCacheError(
            f"CSR column index is out of bounds for {relative_shard}/{model_name}"
        )
    if not np.isfinite(data).all():
        raise SparseActivationCacheError(
            f"CSR data contains non-finite values for {relative_shard}/{model_name}"
        )

    matrix = sparse.csr_matrix((data, indices, indptr), shape=shape, copy=False)
    if (
        matrix.data.dtype != np.float32
        or matrix.indices.dtype != np.int32
        or matrix.indptr.dtype != np.int32
    ):
        raise SparseActivationCacheError(
            f"SciPy changed CSR component dtypes for {relative_shard}/{model_name}"
        )
    return matrix


def load_shard_csr(
    cache_root: str | Path,
    relative_shard: str,
    model_name: str,
    mmap_mode: str | None = "r",
) -> sparse.csr_matrix:
    """Load one completed shard/model activation matrix as SciPy CSR."""

    root = _paper_path(cache_root)
    manifest = load_manifest(root)
    return _load_shard_csr_from_manifest(
        root, manifest, relative_shard, model_name, mmap_mode
    )


def iter_shards(
    cache_root: str | Path,
    model_name: str,
    mmap_mode: str | None = "r",
) -> Iterator[tuple[dict[str, Any], sparse.csr_matrix]]:
    """Yield ``(manifest_record, csr_matrix)`` in global row order."""

    root = _paper_path(cache_root)
    manifest = load_manifest(root)
    for record in manifest["shards"]:
        yield record, _load_shard_csr_from_manifest(
            root, manifest, record["relative_shard"], model_name, mmap_mode
        )


def get_row(
    cache_root: str | Path,
    global_row: int,
    model_name: str,
    mmap_mode: str | None = "r",
) -> sparse.csr_matrix:
    """Return one corpus-global activation row as a ``(1, width)`` CSR matrix."""

    if isinstance(global_row, bool) or not isinstance(global_row, int):
        raise TypeError("global_row must be an integer")
    if global_row < 0:
        raise IndexError(f"global_row is out of range: {global_row}")

    root = _paper_path(cache_root)
    manifest = load_manifest(root)
    shards = manifest["shards"]
    stops = [record["global_row_stop"] for record in shards]
    position = bisect.bisect_right(stops, global_row)
    if position >= len(shards):
        raise IndexError(f"global_row is out of range: {global_row}")
    record = shards[position]
    start = record["global_row_start"]
    if global_row < start:
        # This should already be excluded by manifest contiguity validation.
        raise SparseActivationCacheError(
            f"No shard covers global_row {global_row}"
        )
    matrix = _load_shard_csr_from_manifest(
        root, manifest, record["relative_shard"], model_name, mmap_mode
    )
    return matrix.getrow(global_row - start)


__all__ = [
    "SparseActivationCacheError",
    "get_row",
    "iter_shards",
    "load_manifest",
    "load_shard_csr",
]
