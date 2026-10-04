from project_paths import resource_path as _paper_path, resource_location as _paper_location
import sys
from pathlib import Path
sys.path.insert(0, str(_paper_path(__file__).resolve().parents[1]))
from project_paths import get_path

import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def _is_gcs_path(path: str) -> bool:
    return str(path).startswith("gs://")


def _normalize_gcs_uri(uri: str) -> str:
    return uri.rstrip("/")


def _extract_split_name(path: str, separator: str) -> Optional[str]:
    for part in path.split(separator):
        if part.startswith("split="):
            return part.split("=", 1)[1]
    return None


def _run_cmd(args: List[str]) -> str:
    result = subprocess.run(args, capture_output=True, text=True, check=True)
    return result.stdout


def _ensure_gsutil_available() -> None:
    if shutil.which("gsutil") is None:
        raise RuntimeError(
            "embeddings_dir is a gs:// path but `gsutil` is not available in PATH."
        )


def _gcs_to_local_path(gcs_uri: str, gcs_root: str, local_root: str) -> str:
    root_with_sep = _normalize_gcs_uri(gcs_root) + "/"
    if not gcs_uri.startswith(root_with_sep):
        raise ValueError(f"Unexpected GCS URI outside root: {gcs_uri}")
    relative = gcs_uri[len(root_with_sep) :]
    return os.path.join(local_root, relative)


def _list_gcs_meta_files(dataset_uri: str) -> List[str]:
    listing = _run_cmd(["gsutil", "ls", "-r", f"{dataset_uri}/**/shard_meta.json"])
    return [line.strip() for line in listing.splitlines() if line.strip().startswith("gs://")]


def _copy_one_gcs_file(src_uri: str, dst_path: str) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    subprocess.run(
        ["gsutil", "cp", src_uri, dst_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _copy_files_parallel(
    copy_pairs: List[Tuple[str, str]],
    max_workers: int,
    progress_interval: int,
    label: str,
) -> int:
    if not copy_pairs:
        return 0

    workers = max(1, int(max_workers))
    interval = max(1, int(progress_interval))
    copied = 0
    started_at = time.time()
    total = len(copy_pairs)
    print(f"{label}: starting parallel copy for {total} file(s) with workers={workers}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_copy_one_gcs_file, src, dst) for src, dst in copy_pairs]
        for future in concurrent.futures.as_completed(futures):
            future.result()
            copied += 1
            if copied % interval == 0 or copied == total:
                elapsed = time.time() - started_at
                print(
                    f"{label}: copied {copied}/{total} file(s) in {elapsed:.1f}s",
                    flush=True,
                )

    return copied


def _manifest_cache_path(local_cache_dir: str, cache_key: str) -> str:
    manifests_dir = os.path.join(local_cache_dir, "manifests")
    os.makedirs(manifests_dir, exist_ok=True)
    return os.path.join(manifests_dir, f"{cache_key}.json")


def _load_manifest(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(_paper_location(path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_manifest(path: str, payload: Dict[str, Any]) -> None:
    with open(_paper_location(path), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _build_manifest_records(
    root_uri: str,
    local_root: str,
    datasets: List[str],
    split: Optional[str],
    text_col: Optional[str],
    max_workers: int,
    progress_interval: int,
) -> List[Dict[str, Any]]:
    preliminary_records: List[Dict[str, Any]] = []
    for dataset_name in datasets:
        dataset_uri = f"{root_uri}/{dataset_name}"
        meta_files = _list_gcs_meta_files(dataset_uri)
        if not meta_files:
            raise FileNotFoundError(f"No shard_meta.json files found in {dataset_uri}")

        for meta_uri in meta_files:
            shard_uri = meta_uri.rsplit("/", 1)[0]
            split_name = _extract_split_name(shard_uri, "/")
            if split and split_name != split:
                continue
            preliminary_records.append(
                {
                    "dataset": dataset_name,
                    "meta_uri": meta_uri,
                    "embeddings_uri": f"{shard_uri}/embeddings.npy",
                    "split": split_name,
                }
            )

    if not preliminary_records:
        return []

    meta_copy_pairs: List[Tuple[str, str]] = []
    for record in preliminary_records:
        local_meta_path = _gcs_to_local_path(record["meta_uri"], root_uri, local_root)
        record["local_meta_path"] = local_meta_path
        if not os.path.exists(local_meta_path):
            meta_copy_pairs.append((record["meta_uri"], local_meta_path))

    _copy_files_parallel(
        copy_pairs=meta_copy_pairs,
        max_workers=max_workers,
        progress_interval=progress_interval,
        label="Meta staging",
    )

    filtered_records: List[Dict[str, Any]] = []
    for record in preliminary_records:
        with open(_paper_location(record["local_meta_path"]), "r", encoding="utf-8") as handle:
            meta = json.load(handle)

        if text_col and meta.get("text_column") != text_col:
            continue

        emb_local_path = _gcs_to_local_path(record["embeddings_uri"], root_uri, local_root)
        filtered_records.append(
            {
                "dataset": record["dataset"],
                "meta_uri": record["meta_uri"],
                "embeddings_uri": record["embeddings_uri"],
                "local_meta_path": record["local_meta_path"],
                "local_emb_path": emb_local_path,
                "rows": int(meta.get("rows", 0)),
                "dim": int(meta.get("dim", 0)),
                "split": record["split"],
                "text_col": meta.get("text_column", "unknown"),
            }
        )

    return filtered_records


class _BackgroundStager:
    def __init__(
        self,
        copy_pairs: List[Tuple[str, str]],
        max_workers: int,
        progress_interval: int,
    ):
        self.copy_pairs = copy_pairs
        self.max_workers = max_workers
        self.progress_interval = progress_interval
        self.exception: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        try:
            _copy_files_parallel(
                copy_pairs=self.copy_pairs,
                max_workers=self.max_workers,
                progress_interval=self.progress_interval,
                label="Background embedding staging",
            )
        except BaseException as exc:  # pragma: no cover - defensive path for background thread
            self.exception = exc

    def start(self) -> None:
        if self.copy_pairs:
            self._thread.start()

    def check_healthy(self) -> None:
        if self.exception is not None:
            raise RuntimeError(f"Background staging failed: {self.exception}") from self.exception


_ACTIVE_BACKGROUND_STAGERS: List[_BackgroundStager] = []


def _stage_gcs_embeddings_to_local(
    embeddings_dir: str,
    datasets: List[str],
    split: Optional[str],
    text_col: Optional[str],
    local_cache_dir: str,
    staging_mode: str = "full",
    min_ready_shards: int = 32,
    staging_max_workers: int = 16,
    staging_progress_interval: int = 25,
) -> Dict[str, Any]:
    _ensure_gsutil_available()

    mode = str(staging_mode).lower()
    if mode not in {"full", "progressive"}:
        raise ValueError("data.staging_mode must be either `full` or `progressive`.")

    root_uri = _normalize_gcs_uri(embeddings_dir)
    cache_key = hashlib.sha1(
        json.dumps(
            {
                "root": root_uri,
                "datasets": sorted(datasets),
                "split": split,
                "text_col": text_col,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:12]
    local_root = os.path.join(local_cache_dir, f"gcs_stage_{cache_key}")
    os.makedirs(local_root, exist_ok=True)

    print(f"Staging embeddings from {root_uri} to {local_root}", flush=True)
    started_at = time.time()

    manifest_path = _manifest_cache_path(local_cache_dir, cache_key)
    manifest = _load_manifest(manifest_path)
    if manifest is None:
        print("Manifest cache miss: discovering shards from GCS.", flush=True)
        records = _build_manifest_records(
            root_uri=root_uri,
            local_root=local_root,
            datasets=datasets,
            split=split,
            text_col=text_col,
            max_workers=staging_max_workers,
            progress_interval=staging_progress_interval,
        )
        manifest = {
            "version": 1,
            "root_uri": root_uri,
            "datasets": datasets,
            "split": split,
            "text_col": text_col,
            "records": records,
        }
        _save_manifest(manifest_path, manifest)
        print(f"Manifest written: {manifest_path}", flush=True)
    else:
        print(f"Manifest cache hit: {manifest_path}", flush=True)

    records = manifest.get("records", [])
    if not records:
        raise RuntimeError("No matching shards found in GCS for requested filters.")

    missing_meta_pairs: List[Tuple[str, str]] = []
    missing_emb_pairs: List[Tuple[str, str]] = []
    existing_emb_count = 0
    for record in records:
        if not os.path.exists(record["local_meta_path"]):
            missing_meta_pairs.append((record["meta_uri"], record["local_meta_path"]))
        if not os.path.exists(record["local_emb_path"]):
            missing_emb_pairs.append((record["embeddings_uri"], record["local_emb_path"]))
        else:
            existing_emb_count += 1

    _copy_files_parallel(
        copy_pairs=missing_meta_pairs,
        max_workers=staging_max_workers,
        progress_interval=staging_progress_interval,
        label="Meta reconciliation",
    )

    background_stager = None
    if mode == "full":
        _copy_files_parallel(
            copy_pairs=missing_emb_pairs,
            max_workers=staging_max_workers,
            progress_interval=staging_progress_interval,
            label="Embedding staging",
        )
    else:
        target_ready = max(1, int(min_ready_shards))
        need_sync = max(0, target_ready - existing_emb_count)
        sync_pairs = missing_emb_pairs[:need_sync]
        async_pairs = missing_emb_pairs[need_sync:]

        _copy_files_parallel(
            copy_pairs=sync_pairs,
            max_workers=staging_max_workers,
            progress_interval=staging_progress_interval,
            label="Progressive initial embedding staging",
        )

        if async_pairs:
            background_stager = _BackgroundStager(
                copy_pairs=async_pairs,
                max_workers=staging_max_workers,
                progress_interval=staging_progress_interval,
            )
            background_stager.start()
            _ACTIVE_BACKGROUND_STAGERS.append(background_stager)
            print(
                f"Progressive mode: background staging started for {len(async_pairs)} shard file(s).",
                flush=True,
            )

    elapsed = time.time() - started_at
    ready_count = sum(1 for record in records if os.path.exists(record["local_emb_path"]))
    print(
        f"Prepared {ready_count}/{len(records)} shard(s) in local cache after {elapsed:.1f}s (mode={mode}).",
        flush=True,
    )

    return {
        "local_root": local_root,
        "records": records,
        "background_stager": background_stager,
        "staging_mode": mode,
    }


@dataclass
class ShardInfo:
    dataset: str
    path: str
    rows: int
    dim: int
    split: Optional[str]
    text_col: str


class RealEmbeddingActivationStore:
    """
    Streams embedding batches from on-disk shard files.
    """

    def __init__(
        self,
        embeddings_dir: str,
        datasets: List[str],
        batch_size: int,
        device: str,
        split: Optional[str] = None,
        text_col: Optional[str] = None,
        seed: int = 42,
        batches_per_shard: int = 8,
        contiguous_sampling: bool = True,
        shuffle_within_batch: bool = True,
        candidate_shards: Optional[List[Dict[str, Any]]] = None,
        shard_refresh_interval_batches: int = 200,
        background_stager: Optional[_BackgroundStager] = None,
    ):
        self.embeddings_dir = embeddings_dir
        self.datasets = datasets
        self.batch_size = int(batch_size)
        self.device = device
        self.split = split
        self.text_col = text_col
        self.batches_per_shard = max(1, int(batches_per_shard))
        self.contiguous_sampling = bool(contiguous_sampling)
        self.shuffle_within_batch = bool(shuffle_within_batch)
        self.rng = np.random.default_rng(seed)
        self.background_stager = background_stager

        self._memmaps = {}
        self._current_shard_idx = None
        self._batches_left_on_current_shard = 0

        self._candidate_shards = candidate_shards
        self._known_shard_paths = set()
        self._shard_refresh_interval_batches = max(1, int(shard_refresh_interval_batches))
        self._batches_since_refresh = 0

        if self._candidate_shards is None:
            self.shards = self._discover_shards()
        else:
            self.shards = []
            self._refresh_shards_from_candidates(force=True)

        if not self.shards:
            raise RuntimeError("No embedding shards found for requested filters.")

        dims = {shard.dim for shard in self.shards}
        if len(dims) != 1:
            raise RuntimeError(f"Expected one embedding dim across shards, got: {sorted(dims)}")
        self.d_in = next(iter(dims))
        self._recompute_shard_probs()

    def _recompute_shard_probs(self) -> None:
        rows = np.array([shard.rows for shard in self.shards], dtype=np.float64)
        self._shard_probs = rows / rows.sum()

    def _refresh_shards_from_candidates(self, force: bool = False) -> None:
        if self._candidate_shards is None:
            return

        added = 0
        for record in self._candidate_shards:
            emb_path = record["local_emb_path"]
            if emb_path in self._known_shard_paths:
                continue
            if not os.path.exists(emb_path):
                continue

            rows = int(record.get("rows", 0))
            dim = int(record.get("dim", 0))
            if rows <= 0 or dim <= 0:
                arr = np.load(_paper_location(emb_path), mmap_mode="r")
                rows, dim = int(arr.shape[0]), int(arr.shape[1])

            if rows == 0:
                continue

            self.shards.append(
                ShardInfo(
                    dataset=str(record["dataset"]),
                    path=emb_path,
                    rows=rows,
                    dim=dim,
                    split=record.get("split"),
                    text_col=str(record.get("text_col", "unknown")),
                )
            )
            self._known_shard_paths.add(emb_path)
            added += 1

        if added > 0:
            if self.shards:
                self._recompute_shard_probs()
            print(
                f"Activation store discovered {added} additional staged shard(s); total={len(self.shards)}",
                flush=True,
            )
        elif force and not self.shards:
            print("Activation store waiting for initial staged shards...", flush=True)

    def _maybe_refresh_dynamic_shards(self) -> None:
        if self._candidate_shards is None:
            return

        self._batches_since_refresh += 1
        if self._batches_since_refresh < self._shard_refresh_interval_batches:
            return
        self._batches_since_refresh = 0

        if self.background_stager is not None:
            self.background_stager.check_healthy()
        self._refresh_shards_from_candidates()

    def _discover_shards(self) -> List[ShardInfo]:
        shards: List[ShardInfo] = []

        for dataset_name in self.datasets:
            root = os.path.join(self.embeddings_dir, dataset_name)
            if not os.path.isdir(root):
                raise FileNotFoundError(f"Dataset not found in embeddings dir: {root}")

            for root_dir, _, files in os.walk(root):
                if "embeddings.npy" not in files or "shard_meta.json" not in files:
                    continue

                split_name = None
                for part in root_dir.split(os.sep):
                    if part.startswith("split="):
                        split_name = part.split("=", 1)[1]
                        break

                with open(_paper_location(os.path.join(root_dir, "shard_meta.json")), "r", encoding="utf-8") as handle:
                    meta = json.load(handle)

                if self.split and split_name != self.split:
                    continue
                if self.text_col and meta.get("text_column") != self.text_col:
                    continue

                emb_path = os.path.join(root_dir, "embeddings.npy")
                rows = int(meta.get("rows", 0))
                dim = int(meta.get("dim", 0))
                if rows <= 0 or dim <= 0:
                    arr = np.load(_paper_location(emb_path), mmap_mode="r")
                    rows, dim = int(arr.shape[0]), int(arr.shape[1])

                if rows == 0:
                    continue

                shards.append(
                    ShardInfo(
                        dataset=dataset_name,
                        path=emb_path,
                        rows=rows,
                        dim=dim,
                        split=split_name,
                        text_col=meta.get("text_column", "unknown"),
                    )
                )
        return shards

    def _get_memmap(self, shard_path: str):
        arr = self._memmaps.get(shard_path)
        if arr is None:
            arr = np.load(_paper_location(shard_path), mmap_mode="r")
            self._memmaps[shard_path] = arr
        return arr

    def _choose_shard(self) -> ShardInfo:
        if self._current_shard_idx is None or self._batches_left_on_current_shard <= 0:
            self._current_shard_idx = int(self.rng.choice(len(self.shards), p=self._shard_probs))
            self._batches_left_on_current_shard = self.batches_per_shard
        self._batches_left_on_current_shard -= 1
        return self.shards[self._current_shard_idx]

    def next_batch(self) -> torch.Tensor:
        self._maybe_refresh_dynamic_shards()
        if not self.shards:
            raise RuntimeError("No staged shards are available for sampling.")

        shard = self._choose_shard()
        arr = self._get_memmap(shard.path)
        rows = shard.rows

        if self.contiguous_sampling and rows >= self.batch_size:
            start = int(self.rng.integers(0, rows - self.batch_size + 1))
            batch_np = np.asarray(arr[start : start + self.batch_size], dtype=np.float32)
            if self.shuffle_within_batch:
                permutation = self.rng.permutation(self.batch_size)
                batch_np = batch_np[permutation]
        else:
            replace = rows < self.batch_size
            indices = self.rng.choice(rows, size=self.batch_size, replace=replace)
            batch_np = np.asarray(arr[indices], dtype=np.float32)

        return torch.from_numpy(batch_np).to(self.device)

    def get_batch_tokens(self):
        # Only used in model-performance logging paths that need tokenized text.
        return None


def build_activation_store(cfg: Dict[str, Any]) -> RealEmbeddingActivationStore:
    data_cfg = cfg.get("data", cfg.get("dataset", {}))
    if not isinstance(data_cfg, dict):
        raise ValueError("Expected `data` (or `dataset`) section in config.")

    embeddings_dir = data_cfg.get("embeddings_dir")
    datasets = data_cfg.get("datasets")
    if not embeddings_dir or not datasets:
        raise ValueError("Config `data` section must include `embeddings_dir` and `datasets`.")

    resolved_embeddings_dir = str(embeddings_dir) if _is_gcs_path(str(embeddings_dir)) else str(_paper_location(embeddings_dir))
    candidate_shards = None
    background_stager = None
    shard_refresh_interval_batches = int(data_cfg.get("shard_refresh_interval_batches", 200))

    if _is_gcs_path(resolved_embeddings_dir):
        local_cache_dir = str(data_cfg.get("local_cache_dir", get_path("local_cache_dir")))
        staging_result = _stage_gcs_embeddings_to_local(
            embeddings_dir=resolved_embeddings_dir,
            datasets=[str(dataset) for dataset in datasets],
            split=data_cfg.get("split"),
            text_col=data_cfg.get("text_col"),
            local_cache_dir=local_cache_dir,
            staging_mode=str(data_cfg.get("staging_mode", "full")),
            min_ready_shards=int(data_cfg.get("min_ready_shards", 32)),
            staging_max_workers=int(data_cfg.get("staging_max_workers", 16)),
            staging_progress_interval=int(data_cfg.get("staging_progress_interval", 25)),
        )
        resolved_embeddings_dir = staging_result["local_root"]
        candidate_shards = staging_result["records"]
        background_stager = staging_result["background_stager"]

    return RealEmbeddingActivationStore(
        embeddings_dir=resolved_embeddings_dir,
        datasets=[str(dataset) for dataset in datasets],
        batch_size=int(cfg["batch_size"]),
        device=str(cfg["device"]),
        split=data_cfg.get("split"),
        text_col=data_cfg.get("text_col"),
        seed=int(cfg["seed"]),
        batches_per_shard=int(data_cfg.get("batches_per_shard", 8)),
        contiguous_sampling=bool(data_cfg.get("contiguous_sampling", True)),
        shuffle_within_batch=bool(data_cfg.get("shuffle_within_batch", True)),
        candidate_shards=candidate_shards,
        shard_refresh_interval_batches=shard_refresh_interval_batches,
        background_stager=background_stager,
    )
