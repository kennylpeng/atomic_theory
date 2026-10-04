#!/usr/bin/env python3
"""Fit FAISS K-Means on a streaming raw or PCA-projected embedding sample.

The script first builds a stratified random sample from embedding shards,
optionally projects it into the first N PCA dimensions, stores that sample as a
reusable float32 memmap, then trains one or more K-Means models from it.
"""

from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import sys
from pathlib import Path
sys.path.insert(0, str(_paper_path(__file__).resolve().parents[1]))
from project_paths import get_path

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = _paper_path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compute_embedding_full_pca import _discover_shards  # noqa: E402


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _dataset_list_from_pca(pca_dir: Path) -> List[str]:
    metadata_path = pca_dir / "metadata.json"
    shards_path = pca_dir / "shards.json"
    if metadata_path.exists():
        payload = _load_json(metadata_path)
        datasets = payload.get("datasets")
        if datasets:
            return [str(dataset) for dataset in datasets]
    if shards_path.exists():
        payload = _load_json(shards_path)
        datasets = payload.get("datasets")
        if datasets:
            return [str(dataset) for dataset in datasets]
    raise ValueError("Provide --datasets or use a PCA directory with metadata.json/shards.json.")


def _allocate_stratified_counts(rows: np.ndarray, sample_rows: int) -> np.ndarray:
    total_rows = int(rows.sum())
    if sample_rows > total_rows:
        raise ValueError(f"sample_rows={sample_rows:,} exceeds total rows={total_rows:,}")
    expected = rows.astype(np.float64) * (float(sample_rows) / float(total_rows))
    counts = np.floor(expected).astype(np.int64)
    remainder = int(sample_rows - int(counts.sum()))
    if remainder > 0:
        order = np.argsort(expected - counts)[::-1]
        counts[order[:remainder]] += 1
    return counts


def _sample_indices(rows: int, count: int, rng: np.random.Generator) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    return np.sort(rng.choice(rows, size=count, replace=False).astype(np.int64, copy=False))


def _project_batch(x: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    x32 = np.asarray(x, dtype=np.float32)
    return (x32 - mean) @ components.T


def _sample_metadata_matches(path: Path, expected: Dict[str, Any]) -> bool:
    if not path.exists():
        return False
    try:
        current = _load_json(path)
    except Exception:
        return False
    return all(current.get(key) == value for key, value in expected.items())


def build_or_reuse_sample(
    *,
    embeddings_dir: str,
    datasets: List[str],
    pca_dir: Path | None,
    output_dir: Path,
    sample_dir: Path,
    pca_dim: int | None,
    sample_rows: int,
    seed: int,
    batch_rows: int,
    force_resample: bool,
) -> tuple[Path, int, int]:

    records = _discover_shards(
        embeddings_dir=embeddings_dir,
        datasets=datasets,
        split=None,
        text_col=None,
        local_cache_dir=get_path("local_cache_dir"),
        staging_max_workers=16,
        staging_progress_interval=25,
    )
    records.sort(key=lambda record: (record.dataset, record.embeddings_path))
    if not records:
        raise RuntimeError("No shards found for K-Means sampling.")
    dims = sorted({record.dim for record in records})
    if len(dims) != 1:
        raise RuntimeError(f"Expected one embedding dimension, got {dims}")
    input_dim = int(dims[0])
    training_dim = input_dim if pca_dim is None else int(pca_dim)
    input_space = "raw" if pca_dim is None else "pca"
    sample_label = f"raw{input_dim}" if pca_dim is None else f"pca{pca_dim}"
    sample_path = sample_dir / f"sample_{sample_label}_n{sample_rows}_seed{seed}.f32"
    sample_meta_path = sample_dir / "sample_metadata.json"

    rows = np.array([record.rows for record in records], dtype=np.int64)
    total_rows = int(rows.sum())
    expected_meta = {
        "embeddings_dir": embeddings_dir,
        "datasets": datasets,
        "input_space": input_space,
        "input_dim": input_dim,
        "training_dim": training_dim,
        "pca_dir": None if pca_dir is None else str(pca_dir),
        "pca_dim": pca_dim,
        "sample_rows": int(sample_rows),
        "seed": int(seed),
        "total_rows": total_rows,
    }
    if (
        not force_resample
        and sample_path.exists()
        and sample_path.stat().st_size
        == sample_rows * training_dim * np.dtype("float32").itemsize
        and _sample_metadata_matches(sample_meta_path, expected_meta)
    ):
        print(f"Reusing existing sample memmap: {sample_path}", flush=True)
        return sample_path, input_dim, training_dim

    if pca_dim is None:
        mean = components = None
    else:
        if pca_dir is None:
            raise ValueError("--pca-dir is required unless --raw is used")
        mean = np.load(_paper_location(pca_dir / "mean.npy")).astype(np.float32, copy=False)
        components = np.load(_paper_location(pca_dir / "components.npy"), mmap_mode="r")[:pca_dim].astype(
            np.float32, copy=True
        )
        if mean.shape[0] != input_dim:
            raise ValueError(
                f"PCA mean dim {mean.shape[0]} does not match embedding dim {input_dim}"
            )
        if components.shape != (pca_dim, input_dim):
            raise ValueError(f"Unexpected PCA component shape: {components.shape}")

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    counts = _allocate_stratified_counts(rows, sample_rows)
    rng = np.random.default_rng(seed)
    sample = np.memmap(
        _paper_location(sample_path), dtype=np.float32, mode="w+", shape=(sample_rows, training_dim)
    )

    written = 0
    started_at = time.time()
    shard_entries = []
    for shard_idx, (record, count) in enumerate(zip(records, counts)):
        count = int(count)
        shard_entry = {**asdict(record), "sample_rows": count}
        shard_entries.append(shard_entry)
        if count == 0:
            continue

        arr = np.load(_paper_location(record.embeddings_path), mmap_mode="r")
        indices = _sample_indices(int(arr.shape[0]), count, rng)
        for start in range(0, count, batch_rows):
            stop = min(count, start + batch_rows)
            x = np.asarray(arr[indices[start:stop]], dtype=np.float32)
            values = x if components is None else _project_batch(x, mean, components)
            sample[written : written + values.shape[0]] = values
            written += int(values.shape[0])

        if (shard_idx + 1) % 10 == 0 or shard_idx + 1 == len(records):
            sample.flush()
            elapsed = time.time() - started_at
            print(
                f"[sample {shard_idx + 1}/{len(records)}] written={written:,}/{sample_rows:,} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    if written != sample_rows:
        raise RuntimeError(f"Expected to write {sample_rows:,} rows, wrote {written:,}")
    sample.flush()
    del sample

    _write_json(
        sample_meta_path,
        {
            **expected_meta,
            "sample_path": str(sample_path),
            "embedding_dim": input_dim,
            "shards": shard_entries,
            "created_at_unix": time.time(),
        },
    )
    print(f"Wrote {input_space} sample: {sample_path}", flush=True)
    return sample_path, input_dim, training_dim


def _load_faiss():
    try:
        import faiss  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "FAISS is required for K-Means training. Install faiss-gpu/faiss-cpu "
            "or run in a cluster environment that provides FAISS."
        ) from exc
    return faiss


def train_kmeans(
    *,
    sample_path: Path | None,
    output_dir: Path,
    k_values: List[int],
    training_dim: int,
    input_dim: int,
    input_space: str,
    sample_rows: int,
    seed: int,
    niter: int,
    nredo: int,
    max_points_per_centroid: int,
    use_gpu: bool,
    sample_array: np.ndarray | None = None,
    ensure_niter: bool = False,
) -> None:
    faiss = _load_faiss()
    if sample_array is None:
        if sample_path is None:
            raise ValueError("Provide sample_path or sample_array")
        sample = np.memmap(
            _paper_location(sample_path), dtype=np.float32, mode="r", shape=(sample_rows, training_dim)
        )
        sample_array = np.asarray(sample)
    elif sample_path is not None:
        raise ValueError("Provide only one of sample_path and sample_array")
    if (sample_array.shape != (sample_rows, training_dim)
            or sample_array.dtype != np.float32 or not sample_array.flags.c_contiguous):
        raise ValueError("Training input must be a contiguous float32 matrix of the requested shape")
    num_gpus = int(faiss.get_num_gpus()) if hasattr(faiss, "get_num_gpus") else 0
    gpu_enabled = bool(use_gpu and num_gpus > 0)
    print(
        f"Training FAISS K-Means for k={k_values}; sample={sample_rows:,}x{training_dim}; "
        f"use_gpu={gpu_enabled}; num_gpus={num_gpus}",
        flush=True,
    )

    for k in k_values:
        k_out = output_dir / f"k{k}"
        done_path = k_out / "metadata.json"
        if done_path.exists() and (k_out / "centroids.npy").exists():
            print(f"Skipping existing k={k}: {k_out}", flush=True)
            continue
        k_out.mkdir(parents=True, exist_ok=True)

        started_at = time.time()
        kmeans = faiss.Kmeans(
            d=training_dim,
            k=int(k),
            niter=int(niter),
            nredo=int(nredo),
            verbose=True,
            gpu=gpu_enabled,
            seed=int(seed),
            spherical=False,
            max_points_per_centroid=int(max_points_per_centroid),
        )
        kmeans.train(sample_array)
        objective = [float(x) for x in kmeans.obj]
        while ensure_niter and len(objective) < niter:
            previous_centroids = np.asarray(kmeans.centroids, dtype=np.float32)
            kmeans.cp.niter = niter - len(objective)
            kmeans.train(sample_array, init_centroids=previous_centroids)
            if not len(kmeans.obj):
                raise RuntimeError("FAISS made no progress toward the requested iteration count")
            objective.extend(float(x) for x in kmeans.obj)
        if ensure_niter and len(objective) != niter:
            raise RuntimeError("FAISS iteration count differs from the requested count")
        centroids = np.asarray(kmeans.centroids, dtype=np.float32)
        if not np.isfinite(centroids).all() or not np.isfinite(objective).all():
            raise RuntimeError("FAISS returned nonfinite centroids or objectives")
        np.save(_paper_location(k_out / "centroids.npy"), centroids)

        index = faiss.IndexFlatL2(training_dim)
        index.add(centroids)
        faiss.write_index(index, str(k_out / "centroids.index"))

        _write_json(
            done_path,
            {
                "k": int(k),
                "input_space": input_space,
                "input_dim": int(input_dim),
                "training_dim": int(training_dim),
                "pca_dim": int(training_dim) if input_space == "pca" else None,
                "sample_rows": int(sample_rows),
                "sample_path": None if sample_path is None else str(sample_path),
                "seed": int(seed),
                "niter": int(niter),
                "actual_iterations": len(objective),
                "faiss_stopped_early": len(objective) < int(niter),
                "faiss_version": faiss.__version__,
                "nredo": int(nredo),
                "max_points_per_centroid": int(max_points_per_centroid),
                "use_gpu": gpu_enabled,
                "num_gpus": num_gpus,
                "objective": objective,
                "elapsed_seconds": time.time() - started_at,
                "outputs": {
                    "centroids": "centroids.npy",
                    "index": "centroids.index",
                },
            },
        )
        print(f"Finished k={k}; wrote {k_out}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-dir", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--pca-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--sample-dir",
        help="Directory for the reusable sample memmap (defaults to --output-dir).",
    )
    parser.add_argument("--k-values", nargs="+", type=int, required=True)
    parser.add_argument("--pca-dim", type=int, default=256)
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Train directly on every original embedding coordinate; do not apply PCA.",
    )
    parser.add_argument("--sample-rows", type=int, default=10_000_000)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--sample-batch-rows", type=int, default=8192)
    parser.add_argument("--niter", type=int, default=25)
    parser.add_argument("--nredo", type=int, default=1)
    parser.add_argument(
        "--max-points-per-centroid",
        type=int,
        default=256,
        help="FAISS training cap per centroid. Increase to avoid FAISS subsampling small runs.",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU FAISS even if GPUs are visible.")
    parser.add_argument("--force-resample", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = _paper_path(args.output_dir)
    sample_dir = _paper_path(args.sample_dir) if args.sample_dir else output_dir
    pca_dir = None if args.raw else _paper_path(args.pca_dir) if args.pca_dir else None
    if not args.raw and pca_dir is None:
        raise ValueError("--pca-dir is required unless --raw is used")
    sample_path, input_dim, training_dim = build_or_reuse_sample(
        embeddings_dir=str(args.embeddings_dir),
        datasets=[str(dataset) for dataset in args.datasets],
        pca_dir=pca_dir,
        output_dir=output_dir,
        sample_dir=sample_dir,
        pca_dim=None if args.raw else int(args.pca_dim),
        sample_rows=int(args.sample_rows),
        seed=int(args.seed),
        batch_rows=int(args.sample_batch_rows),
        force_resample=bool(args.force_resample),
    )
    train_kmeans(
        sample_path=sample_path,
        output_dir=output_dir,
        k_values=[int(k) for k in args.k_values],
        training_dim=training_dim,
        input_dim=input_dim,
        input_space="raw" if args.raw else "pca",
        sample_rows=int(args.sample_rows),
        seed=int(args.seed),
        niter=int(args.niter),
        nredo=int(args.nredo),
        max_points_per_centroid=int(args.max_points_per_centroid),
        use_gpu=not bool(args.cpu),
    )


if __name__ == "__main__":
    main()
