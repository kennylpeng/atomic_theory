#!/usr/bin/env python3
"""Compute exact full PCA for embedding shards by streaming sufficient statistics.

This computes PCA from all rows without materializing the full dataset:

    n, sum_x, sum_xx = count, sum(X), sum(X.T @ X)

The covariance is then:

    cov = (sum_xx - n * outer(mean, mean)) / (n - 1)

For 4096-dimensional embeddings, the float64 second-moment matrix is about
134 MB, so the expensive object is small enough to keep in memory while the
rows are streamed from shard memmaps.
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
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

try:
    import yaml
except ImportError as exc:  # pragma: no cover - requirements include PyYAML
    yaml = None
    _YAML_IMPORT_ERROR = exc
else:
    _YAML_IMPORT_ERROR = None


REPO_ROOT = _paper_path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))



@dataclass
class ShardRecord:
    dataset: str
    embeddings_path: str
    rows: int
    dim: int
    split: Optional[str]
    text_col: str


def _is_gcs_path(path: str) -> bool:
    return str(path).startswith("gs://")


def _load_yaml_config(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read config files.") from _YAML_IMPORT_ERROR
    with open(_paper_location(path), "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must parse to a mapping: {path}")
    return cfg


def _config_data_section(config_path: Optional[str]) -> Dict[str, Any]:
    if not config_path:
        return {}
    cfg = _load_yaml_config(config_path)
    data_cfg = cfg.get("data", cfg.get("dataset", {}))
    if not isinstance(data_cfg, dict):
        raise ValueError(f"Expected `data` mapping in config: {config_path}")
    return data_cfg


def _read_meta(meta_path: Path) -> Dict[str, Any]:
    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    if not isinstance(meta, dict):
        raise ValueError(f"Shard metadata must be a JSON object: {meta_path}")
    return meta


def _split_name(path: Path) -> Optional[str]:
    for part in path.parts:
        if part.startswith("split="):
            return part.split("=", 1)[1]
    return None


def _discover_local_shards(
    embeddings_dir: str,
    datasets: Iterable[str],
    split: Optional[str],
    text_col: Optional[str],
) -> List[ShardRecord]:
    records: List[ShardRecord] = []
    root = _paper_path(embeddings_dir)
    for dataset in datasets:
        dataset_root = root / dataset
        if not dataset_root.is_dir():
            raise FileNotFoundError(f"Dataset not found under embeddings dir: {dataset_root}")

        for dirpath, _, filenames in os.walk(dataset_root):
            files = set(filenames)
            if "embeddings.npy" not in files or "shard_meta.json" not in files:
                continue

            shard_dir = _paper_path(dirpath)
            shard_split = _split_name(shard_dir)
            if split and shard_split != split:
                continue

            meta = _read_meta(shard_dir / "shard_meta.json")
            shard_text_col = str(meta.get("text_column", "unknown"))
            if text_col and shard_text_col != text_col:
                continue

            emb_path = shard_dir / "embeddings.npy"
            rows = int(meta.get("rows", 0))
            dim = int(meta.get("dim", 0))
            if rows <= 0 or dim <= 0:
                arr = np.load(_paper_location(emb_path), mmap_mode="r")
                rows, dim = int(arr.shape[0]), int(arr.shape[1])
            if rows <= 0:
                continue

            records.append(
                ShardRecord(
                    dataset=str(dataset),
                    embeddings_path=str(emb_path),
                    rows=rows,
                    dim=dim,
                    split=shard_split,
                    text_col=shard_text_col,
                )
            )
    return records


def _discover_shards(
    embeddings_dir: str,
    datasets: List[str],
    split: Optional[str],
    text_col: Optional[str],
    local_cache_dir: str,
    staging_max_workers: int,
    staging_progress_interval: int,
) -> List[ShardRecord]:
    if _is_gcs_path(embeddings_dir):
        try:
            from trainer.dataset import _stage_gcs_embeddings_to_local
        except ImportError as exc:
            raise RuntimeError(
                "Reading gs:// embeddings requires the training environment dependencies "
                "because it reuses trainer.dataset staging helpers."
            ) from exc

        staged = _stage_gcs_embeddings_to_local(
            embeddings_dir=embeddings_dir,
            datasets=datasets,
            split=split,
            text_col=text_col,
            local_cache_dir=local_cache_dir,
            staging_mode="full",
            staging_max_workers=staging_max_workers,
            staging_progress_interval=staging_progress_interval,
        )
        records = []
        for record in staged["records"]:
            rows = int(record.get("rows", 0))
            dim = int(record.get("dim", 0))
            if rows <= 0 or dim <= 0:
                arr = np.load(_paper_location(record["local_emb_path"]), mmap_mode="r")
                rows, dim = int(arr.shape[0]), int(arr.shape[1])
            if rows <= 0:
                continue
            records.append(
                ShardRecord(
                    dataset=str(record["dataset"]),
                    embeddings_path=str(record["local_emb_path"]),
                    rows=rows,
                    dim=dim,
                    split=record.get("split"),
                    text_col=str(record.get("text_col", "unknown")),
                )
            )
        return records

    return _discover_local_shards(
        embeddings_dir=embeddings_dir,
        datasets=datasets,
        split=split,
        text_col=text_col,
    )


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _save_stats(
    path: Path,
    *,
    n: int,
    sum_x: np.ndarray,
    sum_xx: np.ndarray,
    next_shard_idx: int,
    processed_rows_by_dataset: Dict[str, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    np.savez(
        _paper_location(tmp_path),
        n=np.array(n, dtype=np.int64),
        sum_x=sum_x,
        sum_xx=sum_xx,
        next_shard_idx=np.array(next_shard_idx, dtype=np.int64),
        processed_rows_by_dataset=np.array(json.dumps(processed_rows_by_dataset), dtype=object),
    )
    # np.savez_compressed appends .npz if the suffix is unfamiliar.
    actual_tmp_path = tmp_path
    if not actual_tmp_path.exists() and tmp_path.with_suffix(tmp_path.suffix + ".npz").exists():
        actual_tmp_path = tmp_path.with_suffix(tmp_path.suffix + ".npz")
    actual_tmp_path.replace(path)


def _load_stats(path: Path) -> Dict[str, Any]:
    payload = np.load(_paper_location(path), allow_pickle=True)
    return {
        "n": int(payload["n"]),
        "sum_x": payload["sum_x"],
        "sum_xx": payload["sum_xx"],
        "next_shard_idx": int(payload["next_shard_idx"]),
        "processed_rows_by_dataset": json.loads(str(payload["processed_rows_by_dataset"].item())),
    }


def _iter_slices(num_rows: int, batch_rows: int):
    for start in range(0, num_rows, batch_rows):
        stop = min(num_rows, start + batch_rows)
        yield start, stop


def _accumulate_numpy(
    records: List[ShardRecord],
    output_dir: Path,
    batch_rows: int,
    accumulator_dtype: np.dtype,
    checkpoint_every_shards: int,
    resume: bool,
) -> Dict[str, Any]:
    stats_path = output_dir / "pca_stats.npz"
    if resume and stats_path.exists():
        stats = _load_stats(stats_path)
        n = int(stats["n"])
        sum_x = np.asarray(stats["sum_x"], dtype=accumulator_dtype)
        sum_xx = np.asarray(stats["sum_xx"], dtype=accumulator_dtype)
        start_shard_idx = int(stats["next_shard_idx"])
        processed_rows_by_dataset = {
            str(key): int(value) for key, value in stats["processed_rows_by_dataset"].items()
        }
        print(f"Resuming stats from {stats_path} at shard index {start_shard_idx}", flush=True)
    else:
        dim = records[0].dim
        n = 0
        sum_x = np.zeros(dim, dtype=accumulator_dtype)
        sum_xx = np.zeros((dim, dim), dtype=accumulator_dtype)
        start_shard_idx = 0
        processed_rows_by_dataset = {}

    started_at = time.time()
    total_rows = sum(record.rows for record in records)
    total_remaining_rows = sum(record.rows for record in records[start_shard_idx:])
    print(
        f"Accumulating PCA stats over {len(records)} shard(s), "
        f"{total_rows:,} total row(s), {total_remaining_rows:,} remaining row(s).",
        flush=True,
    )

    for shard_idx, record in enumerate(records[start_shard_idx:], start=start_shard_idx):
        shard_started = time.time()
        arr = np.load(_paper_location(record.embeddings_path), mmap_mode="r")
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D embeddings array, got shape {arr.shape}: {record.embeddings_path}")
        if int(arr.shape[1]) != int(sum_x.shape[0]):
            raise ValueError(
                f"Embedding dim mismatch in {record.embeddings_path}: "
                f"expected {sum_x.shape[0]}, got {arr.shape[1]}"
            )

        for start, stop in _iter_slices(int(arr.shape[0]), batch_rows):
            x = np.asarray(arr[start:stop], dtype=accumulator_dtype)
            sum_x += x.sum(axis=0)
            sum_xx += x.T @ x
            n += int(stop - start)

        processed_rows_by_dataset[record.dataset] = (
            processed_rows_by_dataset.get(record.dataset, 0) + int(arr.shape[0])
        )

        elapsed = time.time() - started_at
        shard_elapsed = time.time() - shard_started
        print(
            f"[{shard_idx + 1}/{len(records)}] {record.dataset} "
            f"rows={arr.shape[0]:,} total_n={n:,} "
            f"shard_time={shard_elapsed:.1f}s elapsed={elapsed:.1f}s",
            flush=True,
        )

        next_shard_idx = shard_idx + 1
        if checkpoint_every_shards > 0 and (
            next_shard_idx == len(records)
            or next_shard_idx % checkpoint_every_shards == 0
        ):
            _save_stats(
                stats_path,
                n=n,
                sum_x=sum_x,
                sum_xx=sum_xx,
                next_shard_idx=next_shard_idx,
                processed_rows_by_dataset=processed_rows_by_dataset,
            )
            print(f"Wrote stats checkpoint: {stats_path}", flush=True)

    _save_stats(
        stats_path,
        n=n,
        sum_x=sum_x,
        sum_xx=sum_xx,
        next_shard_idx=len(records),
        processed_rows_by_dataset=processed_rows_by_dataset,
    )
    print(f"Wrote final stats checkpoint: {stats_path}", flush=True)

    return {
        "n": n,
        "sum_x": sum_x,
        "sum_xx": sum_xx,
        "processed_rows_by_dataset": processed_rows_by_dataset,
    }


def _finalize_pca(
    output_dir: Path,
    stats: Dict[str, Any],
    records: List[ShardRecord],
    metadata: Dict[str, Any],
) -> None:
    n = int(stats["n"])
    if n < 2:
        raise ValueError(f"Need at least two rows to compute covariance, got n={n}")

    sum_x = np.asarray(stats["sum_x"], dtype=np.float64)
    sum_xx = np.asarray(stats["sum_xx"], dtype=np.float64)
    mean = sum_x / float(n)
    covariance = (sum_xx - float(n) * np.outer(mean, mean)) / float(n - 1)
    covariance = np.asarray((covariance + covariance.T) * 0.5, dtype=np.float64)

    print("Running full covariance eigendecomposition...", flush=True)
    evals, evecs = np.linalg.eigh(covariance)
    order = np.argsort(evals)[::-1]
    evals = np.maximum(evals[order], 0.0)
    components = evecs[:, order].T.copy()
    total_variance = float(evals.sum())
    if total_variance > 0:
        explained_variance_ratio = evals / total_variance
    else:
        explained_variance_ratio = np.zeros_like(evals)

    np.save(_paper_location(output_dir / "mean.npy"), mean)
    np.save(_paper_location(output_dir / "components.npy"), components)
    np.save(_paper_location(output_dir / "explained_variance.npy"), evals)
    np.save(_paper_location(output_dir / "explained_variance_ratio.npy"), explained_variance_ratio)
    np.save(_paper_location(output_dir / "covariance.npy"), covariance)

    manifest = {
        **metadata,
        "row_count": n,
        "dim": int(mean.shape[0]),
        "num_components": int(components.shape[0]),
        "total_variance": total_variance,
        "processed_rows_by_dataset": stats["processed_rows_by_dataset"],
        "outputs": {
            "mean": "mean.npy",
            "components": "components.npy",
            "explained_variance": "explained_variance.npy",
            "explained_variance_ratio": "explained_variance_ratio.npy",
            "covariance": "covariance.npy",
            "stats": "pca_stats.npz",
        },
        "shards": [asdict(record) for record in records],
    }
    _write_json(output_dir / "metadata.json", manifest)
    print(f"Wrote PCA outputs to {output_dir}", flush=True)


def _sync_to_gcs(local_dir: Path, output_gcs_dir: str) -> None:
    if shutil.which("gsutil") is None:
        raise RuntimeError("GCS sync requested but `gsutil` is not available.")
    subprocess.run(["gsutil", "-m", "rsync", "-r", str(local_dir), output_gcs_dir], check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute exact full PCA over embedding shards by streaming X and X.T @ X."
    )
    parser.add_argument("--config", help="Training-style YAML config with a data section.")
    parser.add_argument("--embeddings-dir", help="Embedding root directory or gs:// URI.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Dataset names under embeddings-dir. Defaults to config data.datasets.",
    )
    parser.add_argument("--split", default=None, help="Optional split filter.")
    parser.add_argument("--text-col", default=None, help="Optional shard text_column filter.")
    parser.add_argument("--output-dir", required=True, help="Local output directory for PCA artifacts.")
    parser.add_argument("--output-gcs-dir", help="Optional gs:// directory to rsync outputs to at the end.")
    parser.add_argument("--local-cache-dir", default=get_path("local_cache_dir"), help="Cache dir for GCS inputs.")
    parser.add_argument("--batch-rows", type=int, default=8192, help="Rows per X.T @ X chunk.")
    parser.add_argument(
        "--accumulator-dtype",
        choices=("float64", "float32"),
        default="float64",
        help="Accumulator dtype. float64 is recommended for full-dataset PCA.",
    )
    parser.add_argument(
        "--checkpoint-every-shards",
        type=int,
        default=10,
        help="Write pca_stats.npz after this many shards. Set 0 to disable intermediate checkpoints.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from output-dir/pca_stats.npz if present.")
    parser.add_argument("--stats-only", action="store_true", help="Only compute sufficient stats; skip eigendecomp.")
    parser.add_argument(
        "--staging-max-workers",
        type=int,
        default=16,
        help="Parallel GCS staging workers when embeddings-dir is gs://.",
    )
    parser.add_argument(
        "--staging-progress-interval",
        type=int,
        default=25,
        help="Progress interval for GCS staging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_cfg = _config_data_section(args.config)

    embeddings_dir = args.embeddings_dir or data_cfg.get("embeddings_dir")
    datasets = args.datasets or data_cfg.get("datasets")
    split = args.split if args.split is not None else data_cfg.get("split")
    text_col = args.text_col if args.text_col is not None else data_cfg.get("text_col")

    if not embeddings_dir:
        raise ValueError("Provide --embeddings-dir or --config with data.embeddings_dir.")
    if not datasets:
        raise ValueError("Provide --datasets or --config with data.datasets.")

    datasets = [str(dataset) for dataset in datasets]
    output_dir = _paper_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = _discover_shards(
        embeddings_dir=str(embeddings_dir),
        datasets=datasets,
        split=split,
        text_col=text_col,
        local_cache_dir=args.local_cache_dir,
        staging_max_workers=args.staging_max_workers,
        staging_progress_interval=args.staging_progress_interval,
    )
    if not records:
        raise RuntimeError("No matching embedding shards found.")

    dims = sorted({record.dim for record in records})
    if len(dims) != 1:
        raise RuntimeError(f"Expected one embedding dim across shards, got {dims}")

    _write_json(
        output_dir / "shards.json",
        {
            "embeddings_dir": str(embeddings_dir),
            "datasets": datasets,
            "split": split,
            "text_col": text_col,
            "num_shards": len(records),
            "row_count": sum(record.rows for record in records),
            "dim": dims[0],
            "shards": [asdict(record) for record in records],
        },
    )

    stats = _accumulate_numpy(
        records=records,
        output_dir=output_dir,
        batch_rows=int(args.batch_rows),
        accumulator_dtype=np.dtype(args.accumulator_dtype),
        checkpoint_every_shards=int(args.checkpoint_every_shards),
        resume=bool(args.resume),
    )

    metadata = {
        "config": args.config,
        "embeddings_dir": str(embeddings_dir),
        "datasets": datasets,
        "split": split,
        "text_col": text_col,
        "batch_rows": int(args.batch_rows),
        "accumulator_dtype": args.accumulator_dtype,
        "created_at_unix": time.time(),
    }
    _write_json(
        output_dir / "stats_metadata.json",
        {
            **metadata,
            "row_count": int(stats["n"]),
            "dim": dims[0],
            "processed_rows_by_dataset": stats["processed_rows_by_dataset"],
        },
    )

    if not args.stats_only:
        _finalize_pca(output_dir=output_dir, stats=stats, records=records, metadata=metadata)
    else:
        print(f"Stats-only run complete. Stats checkpoint: {output_dir / 'pca_stats.npz'}", flush=True)

    if args.output_gcs_dir:
        _sync_to_gcs(output_dir, args.output_gcs_dir)
        print(f"Synced PCA outputs to {args.output_gcs_dir}", flush=True)


if __name__ == "__main__":
    main()
