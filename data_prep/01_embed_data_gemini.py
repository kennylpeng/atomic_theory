from project_paths import resource_path as _paper_path, resource_location as _paper_location
import sys
from pathlib import Path
sys.path.insert(0, str(_paper_path(__file__).resolve().parents[1]))
from project_paths import get_path

import argparse
import json
import os
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict, load_from_disk
from tqdm.auto import tqdm

INPUT_DATASET_DIR = get_path("datasets_dir")
OUTPUT_EMBED_DIR = get_path("gemini_embeddings_dir")
STATE_FILE = "embedding_state_gemini.json"

TEXT_COL_CANDIDATES = [
    "text",
    "sentence",
    "query",
    "document",
    "passage",
    "content",
    "question",
    "answer",
    "title",
]

EXCLUDED_TEXT_COLUMNS_BY_DATASET = {
    "msmarco": {"title"},
}
EMPTY_TEXT_PLACEHOLDER = "[EMPTY]"
MAX_TEXT_CHARS = 10_000


@dataclass
class SavedDatasetTarget:
    dataset_name: str
    config_name: str | None
    input_path: str

    @property
    def output_rel_path(self) -> str:
        parts = [self.dataset_name]
        if self.config_name:
            parts.append(f"config={self.config_name}")
        return os.path.join(*parts)


@dataclass
class BatchApiShardWork:
    rel_key: str
    target: SavedDatasetTarget
    split_name: str
    text_col: str
    start_idx: int
    end_idx: int
    metadata_columns: list[str]
    use_text_subdir: bool
    job_name: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, 
        description="Embed saved datasets into shard files using Gemini embeddings API."
    )
    parser.add_argument("--input-dir", default=INPUT_DATASET_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_EMBED_DIR)
    parser.add_argument(
        "--model-id",
        default="gemini-embedding-2-preview",
        help="Gemini embedding model ID.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Gemini API key. If omitted, GEMINI_API_KEY env var is used.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Texts per API request.")
    parser.add_argument(
        "--request-mode",
        choices=["live", "batch_api"],
        default="live",
        help=(
            "Request path to use. "
            "'live' uses synchronous embed_content requests; "
            "'batch_api' uses asynchronous Gemini Batch API jobs."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=32,
        help="CPU worker threads for concurrent API requests.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=100_000,
        help="Rows per output shard.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Embedding dtype on disk.",
    )
    parser.add_argument(
        "--task-type",
        default="RETRIEVAL_DOCUMENT",
        help='Embedding task type (e.g. "RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY").',
    )
    parser.add_argument(
        "--output-dimensionality",
        type=int,
        default=None,
        help="Optional output dimensionality if model supports it.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=8,
        help="Retries per API request on transient failures/rate limits.",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=1.0,
        help="Initial backoff before exponential retry (batch_api submit/collect).",
    )
    parser.add_argument(
        "--live-retry-backoff-seconds",
        type=float,
        default=60.0,
        help="Initial backoff before exponential retry for live embed_content (default: 60s).",
    )
    parser.add_argument(
        "--max-enqueued-tokens",
        type=int,
        default=10_000_000,
        help=(
            "Approximate cap for total in-flight tokens across concurrent requests. "
            "Use this to stay under Gemini enqueued-token limits."
        ),
    )
    parser.add_argument(
        "--chars-per-token",
        type=float,
        default=4.0,
        help="Heuristic used to estimate tokens from text length (default: 4.0 chars/token).",
    )
    parser.add_argument(
        "--batch-api-poll-seconds",
        type=int,
        default=15,
        help="Polling interval for Gemini Batch API jobs.",
    )
    parser.add_argument(
        "--batch-api-timeout-seconds",
        type=int,
        default=24 * 60 * 60,
        help="Timeout for waiting on a single Batch API job.",
    )
    parser.add_argument(
        "--batch-api-submit-all",
        action="store_true",
        help=(
            "batch_api only: submit every pending shard job first, then poll and write. "
            "Uses embedding_state_gemini.json pending_batch_jobs for resume after submit."
        ),
    )
    parser.add_argument(
        "--batch-api-submit-workers",
        type=int,
        default=None,
        help=(
            "Parallelism for the submit phase when --batch-api-submit-all is set "
            "(default: min(num_jobs, 32))."
        ),
    )
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Store original text in metadata.parquet (larger disk footprint).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute shards even if they already exist.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset folder names to process (e.g. emotion fever).",
    )
    return parser.parse_args()


def load_state(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {"completed_shards": {}}
    with open(_paper_location(path), "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(path: str, state: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(_paper_location(path), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def discover_targets(input_dir: str, datasets_filter: set[str] | None) -> list[SavedDatasetTarget]:
    targets: list[SavedDatasetTarget] = []
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input dataset directory does not exist: {input_dir}")

    for dataset_name in sorted(os.listdir(input_dir)):
        dataset_root = os.path.join(input_dir, dataset_name)
        if not os.path.isdir(dataset_root):
            continue
        if datasets_filter and dataset_name not in datasets_filter:
            continue

        config_dirs = [
            d for d in sorted(os.listdir(dataset_root)) if d.startswith("config=")
        ]
        if config_dirs:
            for config_dir in config_dirs:
                config_name = config_dir.split("=", 1)[1]
                input_path = os.path.join(dataset_root, config_dir)
                targets.append(
                    SavedDatasetTarget(
                        dataset_name=dataset_name,
                        config_name=config_name,
                        input_path=input_path,
                    )
                )
        else:
            targets.append(
                SavedDatasetTarget(
                    dataset_name=dataset_name,
                    config_name=None,
                    input_path=dataset_root,
                )
            )
    return targets


def choose_text_columns(ds: Dataset, dataset_name: str) -> list[str]:
    excluded_cols = EXCLUDED_TEXT_COLUMNS_BY_DATASET.get(dataset_name, set())

    matched = [
        col for col in TEXT_COL_CANDIDATES if col in ds.column_names and col not in excluded_cols
    ]
    if matched:
        return matched

    return []


def get_metadata_columns(
    ds: Dataset,
    embedded_text_cols: list[str],
    include_text: bool,
) -> list[str]:
    cols = ["row_idx"]
    embedded_text_set = set(embedded_text_cols)
    for col in ds.column_names:
        if col in embedded_text_set and not include_text:
            continue
        feature = ds.features[col]
        if hasattr(feature, "dtype"):
            cols.append(col)
    return cols


def shard_rel_key(
    target: SavedDatasetTarget,
    split_name: str,
    shard_idx: int,
    text_col: str | None = None,
    use_text_subdir: bool = False,
) -> str:
    base = target.output_rel_path
    parts = [base]
    if use_text_subdir and text_col is not None:
        parts.append(f"text_col={text_col}")
    parts.extend([f"split={split_name}", f"shard-{shard_idx:06d}"])
    return os.path.join(*parts)


def shard_paths(output_dir: str, rel_key: str) -> tuple[str, str, str]:
    shard_dir = os.path.join(output_dir, rel_key)
    emb_path = os.path.join(shard_dir, "embeddings.npy")
    meta_path = os.path.join(shard_dir, "metadata.parquet")
    shard_meta_path = os.path.join(shard_dir, "shard_meta.json")
    return emb_path, meta_path, shard_meta_path


def shard_exists(output_dir: str, rel_key: str) -> bool:
    emb_path, meta_path, shard_meta_path = shard_paths(output_dir, rel_key)
    return (
        os.path.exists(emb_path)
        and os.path.exists(meta_path)
        and os.path.exists(shard_meta_path)
    )


def _extract_vectors(response: Any) -> list[list[float]]:
    embeddings = getattr(response, "embeddings", None)
    if embeddings is None:
        raise RuntimeError("Gemini API response missing 'embeddings' field.")

    vectors: list[list[float]] = []
    for emb in embeddings:
        values = getattr(emb, "values", None)
        if values is None:
            raise RuntimeError("Gemini API embedding item missing 'values'.")
        vectors.append(list(values))
    return vectors


def _sanitize_text_for_embedding(value: Any) -> str:
    if value is None:
        return EMPTY_TEXT_PLACEHOLDER
    text = str(value)[:MAX_TEXT_CHARS].strip()
    if not text:
        return EMPTY_TEXT_PLACEHOLDER
    return text


def _should_retry(exc: Exception) -> bool:
    text = str(exc).lower()
    retry_markers = [
        "429",
        "resource_exhausted",
        "rate limit",
        "deadline exceeded",
        "timeout",
        "internal",
        "unavailable",
        "503",
    ]
    return any(marker in text for marker in retry_markers)


def _embed_batch_with_retry(
    client: Any,
    types_mod: Any,
    model_id: str,
    task_type: str,
    output_dimensionality: int | None,
    texts: list[str],
    max_retries: int,
    retry_backoff_seconds: float,
) -> np.ndarray:
    config_kwargs: dict[str, Any] = {"task_type": task_type}
    if output_dimensionality is not None:
        config_kwargs["output_dimensionality"] = output_dimensionality
    config = types_mod.EmbedContentConfig(**config_kwargs)

    attempt = 0
    while True:
        try:
            response = client.models.embed_content(
                model=model_id,
                contents=texts,
                config=config,
            )
            vectors = _extract_vectors(response)
            return np.asarray(vectors, dtype=np.float32)
        except Exception as exc:
            attempt += 1
            if attempt > max_retries or not _should_retry(exc):
                raise
            sleep_s = retry_backoff_seconds * (2 ** (attempt - 1))
            if sleep_s >= 60:
                wait_label = f"{sleep_s / 60:.1f}m"
            else:
                wait_label = f"{sleep_s:.1f}s"
            print(
                f"Gemini request failed (attempt {attempt}/{max_retries}) "
                f"for batch size {len(texts)}: {exc}. Retrying in {wait_label}..."
            )
            time.sleep(sleep_s)


def _extract_values_recursive(obj: Any) -> list[float] | None:
    if isinstance(obj, dict):
        if "values" in obj and isinstance(obj["values"], list):
            values = obj["values"]
            if values and all(isinstance(v, (int, float)) for v in values):
                return [float(v) for v in values]
        for value in obj.values():
            found = _extract_values_recursive(value)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _extract_values_recursive(item)
            if found is not None:
                return found
    return None


def _poll_batch_job(client: Any, job_name: str, poll_seconds: int, timeout_seconds: int) -> Any:
    terminal_states = {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }
    start_time = time.time()
    while True:
        job = client.batches.get(name=job_name)
        state_name = getattr(getattr(job, "state", None), "name", None) or str(
            getattr(job, "state", "UNKNOWN")
        )
        if state_name in terminal_states:
            return job
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            raise TimeoutError(
                f"Timed out waiting for batch job {job_name} after {timeout_seconds}s."
            )
        time.sleep(poll_seconds)


def _parse_batch_file_output(file_bytes: bytes, expected_count: int) -> list[list[float]]:
    lines = [line for line in file_bytes.decode("utf-8").splitlines() if line.strip()]
    by_key: dict[int, list[float]] = {}
    fallback_order: list[list[float]] = []

    for line in lines:
        payload = json.loads(line)
        if payload.get("error"):
            raise RuntimeError(f"Batch API request failed: {payload['error']}")

        values = _extract_values_recursive(payload)
        if values is None:
            continue

        key_raw = payload.get("key")
        if key_raw is None:
            metadata = payload.get("metadata")
            if isinstance(metadata, dict):
                key_raw = metadata.get("key")
        if key_raw is None:
            fallback_order.append(values)
            continue

        try:
            key_idx = int(key_raw)
        except (TypeError, ValueError):
            fallback_order.append(values)
            continue
        by_key[key_idx] = values

    if len(by_key) == expected_count:
        return [by_key[i] for i in range(expected_count)]
    if len(fallback_order) == expected_count:
        return fallback_order
    raise RuntimeError(
        f"Could not parse complete batch output: expected {expected_count} vectors, "
        f"got keyed={len(by_key)}, fallback={len(fallback_order)}."
    )


def _build_batch_request_items(
    texts: list[str],
    task_type: str,
    output_dimensionality: int | None,
) -> list[dict[str, Any]]:
    request_items = []
    for i, text in enumerate(texts):
        request: dict[str, Any] = {"content": {"parts": [{"text": text}]}, "task_type": task_type}
        if output_dimensionality is not None:
            request["output_dimensionality"] = output_dimensionality
        request_items.append({"key": str(i), "request": request})
    return request_items


def _submit_batch_api_job(
    client: Any,
    types_mod: Any,
    model_id: str,
    request_items: list[dict[str, Any]],
    display_name: str,
) -> str:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
            tmp_path = tmp.name
            for req in request_items:
                tmp.write(json.dumps(req) + "\n")

        upload_cfg = types_mod.UploadFileConfig(display_name="embed-batch", mime_type="jsonl")
        uploaded_file = client.files.upload(file=tmp_path, config=upload_cfg)
        job = client.batches.create_embeddings(
            model=model_id,
            src={"file_name": uploaded_file.name},
            config={"display_name": display_name},
        )
        return job.name
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def _collect_batch_api_job(
    client: Any,
    job_name: str,
    expected_count: int,
    poll_seconds: int,
    timeout_seconds: int,
) -> np.ndarray:
    final_job = _poll_batch_job(
        client=client,
        job_name=job_name,
        poll_seconds=poll_seconds,
        timeout_seconds=timeout_seconds,
    )
    state_name = getattr(getattr(final_job, "state", None), "name", None) or str(
        getattr(final_job, "state", "UNKNOWN")
    )
    if state_name != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"Batch API job ended with state {state_name}: {final_job}")

    dest = getattr(final_job, "dest", None)
    file_name = getattr(dest, "file_name", None)
    if file_name:
        file_bytes = client.files.download(file=file_name)
        vectors = _parse_batch_file_output(file_bytes, expected_count=expected_count)
        return np.asarray(vectors, dtype=np.float32)

    inline_responses = getattr(dest, "inlined_embed_content_responses", None)
    if inline_responses:
        vectors: list[list[float]] = []
        for inline_item in inline_responses:
            error_obj = getattr(inline_item, "error", None)
            if error_obj is not None:
                raise RuntimeError(f"Batch API inline embedding error: {error_obj}")
            response_obj = getattr(inline_item, "response", None)
            values = _extract_values_recursive(response_obj)
            if values is None:
                raise RuntimeError("Could not extract embedding values from inline response.")
            vectors.append(values)
        if len(vectors) != expected_count:
            raise RuntimeError(
                f"Inline batch response size mismatch: expected {expected_count}, got {len(vectors)}."
            )
        return np.asarray(vectors, dtype=np.float32)

    raise RuntimeError("Batch API job succeeded but no file or inline embedding responses found.")


def _submit_batch_api_with_retry(
    client: Any,
    types_mod: Any,
    model_id: str,
    task_type: str,
    output_dimensionality: int | None,
    texts: list[str],
    max_retries: int,
    retry_backoff_seconds: float,
    display_name: str,
) -> str:
    request_items = _build_batch_request_items(texts, task_type, output_dimensionality)
    attempt = 0
    while True:
        try:
            return _submit_batch_api_job(
                client=client,
                types_mod=types_mod,
                model_id=model_id,
                request_items=request_items,
                display_name=display_name,
            )
        except Exception as exc:
            attempt += 1
            if attempt > max_retries or not _should_retry(exc):
                raise
            sleep_s = retry_backoff_seconds * (2 ** (attempt - 1))
            print(
                f"Gemini batch API submit failed (attempt {attempt}/{max_retries}) "
                f"for batch size {len(texts)}: {exc}. Retrying in {sleep_s:.1f}s..."
            )
            time.sleep(sleep_s)


def _collect_batch_api_with_retry(
    client: Any,
    job_name: str,
    expected_count: int,
    max_retries: int,
    retry_backoff_seconds: float,
    poll_seconds: int,
    timeout_seconds: int,
) -> np.ndarray:
    attempt = 0
    while True:
        try:
            return _collect_batch_api_job(
                client=client,
                job_name=job_name,
                expected_count=expected_count,
                poll_seconds=poll_seconds,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            attempt += 1
            if attempt > max_retries or not _should_retry(exc):
                raise
            sleep_s = retry_backoff_seconds * (2 ** (attempt - 1))
            print(
                f"Gemini batch API collect failed (attempt {attempt}/{max_retries}) "
                f"for job {job_name}: {exc}. Retrying in {sleep_s:.1f}s..."
            )
            time.sleep(sleep_s)


def _embed_batch_api_with_retry(
    client: Any,
    types_mod: Any,
    model_id: str,
    task_type: str,
    output_dimensionality: int | None,
    texts: list[str],
    max_retries: int,
    retry_backoff_seconds: float,
    poll_seconds: int,
    timeout_seconds: int,
) -> np.ndarray:
    job_name = _submit_batch_api_with_retry(
        client=client,
        types_mod=types_mod,
        model_id=model_id,
        task_type=task_type,
        output_dimensionality=output_dimensionality,
        texts=texts,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        display_name=f"embed-batch-{int(time.time())}",
    )
    return _collect_batch_api_with_retry(
        client=client,
        job_name=job_name,
        expected_count=len(texts),
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        poll_seconds=poll_seconds,
        timeout_seconds=timeout_seconds,
    )


def encode_shard_parallel(
    client: Any,
    types_mod: Any,
    model_id: str,
    split_ds: Dataset,
    text_col: str,
    batch_size: int,
    max_workers: int,
    request_mode: str,
    task_type: str,
    output_dimensionality: int | None,
    max_retries: int,
    live_retry_backoff_seconds: float,
    batch_retry_backoff_seconds: float,
    max_enqueued_tokens: int,
    chars_per_token: float,
    batch_api_poll_seconds: int,
    batch_api_timeout_seconds: int,
    dtype: np.dtype,
    start_idx: int,
    end_idx: int,
    progress_desc: str,
) -> np.ndarray:
    texts = split_ds[start_idx:end_idx][text_col]
    texts = [_sanitize_text_for_embedding(t) for t in texts]
    rows = len(texts)
    if rows == 0:
        return np.empty((0, 0), dtype=dtype)

    batches: list[tuple[int, list[str]]] = []
    for i in range(0, rows, batch_size):
        batch_idx = i // batch_size
        batches.append((batch_idx, texts[i : i + batch_size]))

    def estimate_tokens(batch_texts: list[str]) -> int:
        approx = sum(max(1, int(np.ceil(len(t) / max(1e-6, chars_per_token)))) for t in batch_texts)
        return max(1, approx)

    outputs: dict[int, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending: list[tuple[int, list[str], int]] = [
            (batch_idx, batch_texts, estimate_tokens(batch_texts))
            for batch_idx, batch_texts in batches
        ]
        inflight: dict[Any, tuple[int, int]] = {}
        inflight_tokens = 0

        with tqdm(
            total=len(pending),
            desc=progress_desc,
            leave=False,
        ) as pbar:
            while pending or inflight:
                made_progress = False
                while pending and len(inflight) < max_workers:
                    batch_idx, batch_texts, est_tokens = pending[0]
                    if inflight and inflight_tokens + est_tokens > max_enqueued_tokens:
                        break
                    pending.pop(0)
                    retry_backoff_seconds = (
                        live_retry_backoff_seconds
                        if request_mode == "live"
                        else batch_retry_backoff_seconds
                    )
                    future = executor.submit(
                        _embed_batch_with_retry
                        if request_mode == "live"
                        else _embed_batch_api_with_retry,
                        client,
                        types_mod,
                        model_id,
                        task_type,
                        output_dimensionality,
                        batch_texts,
                        max_retries,
                        retry_backoff_seconds,
                        *(()
                          if request_mode == "live"
                          else (batch_api_poll_seconds, batch_api_timeout_seconds)),
                    )
                    inflight[future] = (batch_idx, est_tokens)
                    inflight_tokens += est_tokens
                    made_progress = True

                # If the next batch alone exceeds the budget, submit it by itself.
                if pending and not inflight and not made_progress:
                    batch_idx, batch_texts, est_tokens = pending.pop(0)
                    print(
                        f"Warning: single batch estimated at {est_tokens} tokens exceeds "
                        f"--max-enqueued-tokens={max_enqueued_tokens}; submitting anyway."
                    )
                    retry_backoff_seconds = (
                        live_retry_backoff_seconds
                        if request_mode == "live"
                        else batch_retry_backoff_seconds
                    )
                    future = executor.submit(
                        _embed_batch_with_retry
                        if request_mode == "live"
                        else _embed_batch_api_with_retry,
                        client,
                        types_mod,
                        model_id,
                        task_type,
                        output_dimensionality,
                        batch_texts,
                        max_retries,
                        retry_backoff_seconds,
                        *(()
                          if request_mode == "live"
                          else (batch_api_poll_seconds, batch_api_timeout_seconds)),
                    )
                    inflight[future] = (batch_idx, est_tokens)
                    inflight_tokens += est_tokens

                if inflight:
                    done, _ = wait(list(inflight.keys()), return_when=FIRST_COMPLETED)
                    for future in done:
                        batch_idx, est_tokens = inflight.pop(future)
                        inflight_tokens -= est_tokens
                        outputs[batch_idx] = future.result()
                        pbar.update(1)

    ordered = [outputs[i] for i in range(len(batches))]
    shard_embs = np.concatenate(ordered, axis=0).astype(dtype, copy=False)
    if shard_embs.shape[0] != rows:
        raise RuntimeError(
            f"Embedding row count mismatch: expected {rows}, got {shard_embs.shape[0]}"
        )
    return shard_embs


def write_shard(
    output_dir: str,
    rel_key: str,
    embeddings: np.ndarray,
    split_ds: Dataset,
    metadata_columns: list[str],
    start_idx: int,
    end_idx: int,
    text_col: str,
    model_id: str,
    task_type: str,
) -> None:
    emb_path, meta_path, shard_meta_path = shard_paths(output_dir, rel_key)
    os.makedirs(os.path.dirname(emb_path), exist_ok=True)

    np.save(_paper_location(emb_path), embeddings)

    data_slice = split_ds[start_idx:end_idx]
    metadata_payload: dict[str, list[Any]] = {"row_idx": list(range(start_idx, end_idx))}
    for col in metadata_columns:
        if col == "row_idx":
            continue
        if col == text_col and col not in data_slice:
            continue
        metadata_payload[col] = data_slice[col]

    table = pa.table(metadata_payload)
    pq.write_table(table, meta_path, compression="zstd")

    shard_meta = {
        "rows": int(embeddings.shape[0]),
        "dim": int(embeddings.shape[1]),
        "dtype": str(embeddings.dtype),
        "text_column": text_col,
        "model_id": model_id,
        "task_type": task_type,
        "embedding_file": "embeddings.npy",
        "metadata_file": "metadata.parquet",
    }
    with open(_paper_location(shard_meta_path), "w", encoding="utf-8") as f:
        json.dump(shard_meta, f, indent=2, sort_keys=True)


def _load_splits_for_target(target: SavedDatasetTarget) -> DatasetDict:
    loaded = load_from_disk(target.input_path)
    if isinstance(loaded, DatasetDict):
        return loaded
    return DatasetDict({"train": loaded})


def enumerate_pending_shard_work(
    targets: list[SavedDatasetTarget],
    args: argparse.Namespace,
    state: dict[str, Any],
) -> list[BatchApiShardWork]:
    pending_jobs: dict[str, str] = state.setdefault("pending_batch_jobs", {})
    work_items: list[BatchApiShardWork] = []

    for target in targets:
        splits = _load_splits_for_target(target)
        for split_name, split_ds in splits.items():
            if len(split_ds) == 0:
                continue

            text_cols = choose_text_columns(split_ds, target.dataset_name)
            if not text_cols:
                continue

            use_text_subdir = len(text_cols) > 1
            metadata_columns = get_metadata_columns(split_ds, text_cols, args.include_text)
            num_rows = len(split_ds)
            num_shards = (num_rows + args.shard_size - 1) // args.shard_size

            for text_col in text_cols:
                for shard_idx in range(num_shards):
                    rel_key = shard_rel_key(
                        target,
                        split_name,
                        shard_idx,
                        text_col=text_col,
                        use_text_subdir=use_text_subdir,
                    )
                    if not args.overwrite and shard_exists(args.output_dir, rel_key):
                        state["completed_shards"][rel_key] = "completed"
                        pending_jobs.pop(rel_key, None)
                        continue

                    start_idx = shard_idx * args.shard_size
                    end_idx = min(start_idx + args.shard_size, num_rows)
                    job_name = pending_jobs.get(rel_key)
                    work_items.append(
                        BatchApiShardWork(
                            rel_key=rel_key,
                            target=target,
                            split_name=split_name,
                            text_col=text_col,
                            start_idx=start_idx,
                            end_idx=end_idx,
                            metadata_columns=metadata_columns,
                            use_text_subdir=use_text_subdir,
                            job_name=job_name,
                        )
                    )
    return work_items


def _submit_batch_shard_work(
    client: Any,
    types_mod: Any,
    args: argparse.Namespace,
    work: BatchApiShardWork,
    split_cache: dict[str, DatasetDict],
) -> BatchApiShardWork:
    if work.job_name:
        return work

    splits = split_cache.get(work.target.input_path)
    if splits is None:
        splits = _load_splits_for_target(work.target)
        split_cache[work.target.input_path] = splits

    split_ds = splits[work.split_name]
    texts = split_ds[work.start_idx : work.end_idx][work.text_col]
    texts = [_sanitize_text_for_embedding(t) for t in texts]
    if not texts:
        raise RuntimeError(f"No texts to embed for {work.rel_key}")

    work.job_name = _submit_batch_api_with_retry(
        client=client,
        types_mod=types_mod,
        model_id=args.model_id,
        task_type=args.task_type,
        output_dimensionality=args.output_dimensionality,
        texts=texts,
        max_retries=args.max_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
        display_name=f"embed-{work.rel_key.replace('/', '-')}-{int(time.time())}",
    )
    return work


def _collect_batch_shard_work(
    client: Any,
    args: argparse.Namespace,
    work: BatchApiShardWork,
    split_cache: dict[str, DatasetDict],
    state: dict[str, Any],
    state_path: str,
) -> None:
    if not work.job_name:
        raise RuntimeError(f"Missing batch job name for {work.rel_key}")

    splits = split_cache.get(work.target.input_path)
    if splits is None:
        splits = _load_splits_for_target(work.target)
        split_cache[work.target.input_path] = splits

    split_ds = splits[work.split_name]
    expected_rows = work.end_idx - work.start_idx
    dtype = np.float16 if args.dtype == "float16" else np.float32

    embeddings = _collect_batch_api_with_retry(
        client=client,
        job_name=work.job_name,
        expected_count=expected_rows,
        max_retries=args.max_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
        poll_seconds=args.batch_api_poll_seconds,
        timeout_seconds=args.batch_api_timeout_seconds,
    ).astype(dtype, copy=False)

    write_shard(
        output_dir=args.output_dir,
        rel_key=work.rel_key,
        embeddings=embeddings,
        split_ds=split_ds,
        metadata_columns=work.metadata_columns,
        start_idx=work.start_idx,
        end_idx=work.end_idx,
        text_col=work.text_col,
        model_id=args.model_id,
        task_type=args.task_type,
    )
    state["completed_shards"][work.rel_key] = "completed"
    state.setdefault("pending_batch_jobs", {}).pop(work.rel_key, None)
    save_state(state_path, state)


def run_batch_api_submit_all(
    targets: list[SavedDatasetTarget],
    client: Any,
    types_mod: Any,
    args: argparse.Namespace,
    state: dict[str, Any],
    state_path: str,
) -> None:
    work_items = enumerate_pending_shard_work(targets, args, state)
    save_state(state_path, state)

    if not work_items:
        print("No pending shards to embed.")
        return

    to_submit = [work for work in work_items if not work.job_name]
    already_submitted = [work for work in work_items if work.job_name]

    submit_workers = args.batch_api_submit_workers
    if submit_workers is None:
        submit_workers = min(len(to_submit), 32)
    submit_workers = max(1, submit_workers)

    print(
        f"Batch submit-all: {len(work_items)} shard job(s) "
        f"({len(to_submit)} to submit now, {len(already_submitted)} resumed from state), "
        f"submit_workers={submit_workers}, collect_workers={args.max_workers}"
    )

    split_cache: dict[str, DatasetDict] = {}
    pending_jobs: dict[str, str] = state.setdefault("pending_batch_jobs", {})

    if to_submit:
        with ThreadPoolExecutor(max_workers=submit_workers) as executor:
            futures = {
                executor.submit(
                    _submit_batch_shard_work,
                    client,
                    types_mod,
                    args,
                    work,
                    split_cache,
                ): work
                for work in to_submit
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="batch-submit",
            ):
                work = futures[future]
                completed = future.result()
                pending_jobs[completed.rel_key] = completed.job_name
                print(f"Submitted {completed.rel_key} -> {completed.job_name}")
                save_state(state_path, state)

    collect_workers = args.max_workers
    with ThreadPoolExecutor(max_workers=collect_workers) as executor:
        futures = {
            executor.submit(
                _collect_batch_shard_work,
                client,
                args,
                work,
                split_cache,
                state,
                state_path,
            ): work
            for work in work_items
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="batch-collect",
        ):
            work = futures[future]
            future.result()
            print(f"Completed {work.rel_key}")


def process_target(
    target: SavedDatasetTarget,
    client: Any,
    types_mod: Any,
    args: argparse.Namespace,
    state: dict[str, Any],
    state_path: str,
) -> None:
    loaded = load_from_disk(target.input_path)
    if isinstance(loaded, DatasetDict):
        splits = loaded
    else:
        splits = DatasetDict({"train": loaded})

    dtype = np.float16 if args.dtype == "float16" else np.float32

    for split_name, split_ds in splits.items():
        if len(split_ds) == 0:
            print(f"Skipping empty split: {target.output_rel_path} [{split_name}]")
            continue

        text_cols = choose_text_columns(split_ds, target.dataset_name)
        if not text_cols:
            print(
                f"Skipping {target.output_rel_path} [{split_name}]: "
                f"no allowed text columns found. "
                f"Expected one of {TEXT_COL_CANDIDATES}, got {split_ds.column_names}"
            )
            continue
        use_text_subdir = len(text_cols) > 1
        metadata_columns = get_metadata_columns(split_ds, text_cols, args.include_text)
        num_rows = len(split_ds)
        num_shards = (num_rows + args.shard_size - 1) // args.shard_size

        for text_col in text_cols:
            print(
                f"Embedding {target.output_rel_path} [{split_name}] "
                f"rows={num_rows}, text_col={text_col}, shards={num_shards}, "
                f"batch_size={args.batch_size}, workers={args.max_workers}, mode={args.request_mode}"
            )
            for shard_idx in tqdm(
                range(num_shards),
                desc=f"{target.output_rel_path}:{split_name}:{text_col}",
            ):
                rel_key = shard_rel_key(
                    target,
                    split_name,
                    shard_idx,
                    text_col=text_col,
                    use_text_subdir=use_text_subdir,
                )
                if not args.overwrite and shard_exists(args.output_dir, rel_key):
                    state["completed_shards"][rel_key] = "completed"
                    continue

                start_idx = shard_idx * args.shard_size
                end_idx = min(start_idx + args.shard_size, num_rows)
                embeddings = encode_shard_parallel(
                    client=client,
                    types_mod=types_mod,
                    model_id=args.model_id,
                    split_ds=split_ds,
                    text_col=text_col,
                    batch_size=args.batch_size,
                    max_workers=args.max_workers,
                    request_mode=args.request_mode,
                    task_type=args.task_type,
                    output_dimensionality=args.output_dimensionality,
                    max_retries=args.max_retries,
                    live_retry_backoff_seconds=args.live_retry_backoff_seconds,
                    batch_retry_backoff_seconds=args.retry_backoff_seconds,
                    max_enqueued_tokens=args.max_enqueued_tokens,
                    chars_per_token=args.chars_per_token,
                    batch_api_poll_seconds=args.batch_api_poll_seconds,
                    batch_api_timeout_seconds=args.batch_api_timeout_seconds,
                    dtype=dtype,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    progress_desc=f"api:{split_name}:{text_col}:shard-{shard_idx:06d}",
                )
                write_shard(
                    output_dir=args.output_dir,
                    rel_key=rel_key,
                    embeddings=embeddings,
                    split_ds=split_ds,
                    metadata_columns=metadata_columns,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    text_col=text_col,
                    model_id=args.model_id,
                    task_type=args.task_type,
                )
                state["completed_shards"][rel_key] = "completed"
                save_state(state_path, state)


def create_gemini_client(api_key: str):
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'google-genai'. Install with: pip install google-genai"
        ) from exc

    client = genai.Client(api_key=api_key)
    return client, types


def main() -> None:
    args = parse_args()
    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Gemini API key is required. Pass --api-key or set GEMINI_API_KEY."
        )
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.max_workers < 1:
        raise ValueError("--max-workers must be >= 1")
    if args.max_enqueued_tokens < 1:
        raise ValueError("--max-enqueued-tokens must be >= 1")
    if args.chars_per_token <= 0:
        raise ValueError("--chars-per-token must be > 0")
    if args.batch_api_submit_all and args.request_mode != "batch_api":
        raise ValueError("--batch-api-submit-all requires --request-mode batch_api")
    if args.batch_api_submit_workers is not None and args.batch_api_submit_workers < 1:
        raise ValueError("--batch-api-submit-workers must be >= 1")
    if args.live_retry_backoff_seconds <= 0:
        raise ValueError("--live-retry-backoff-seconds must be > 0")

    os.makedirs(args.output_dir, exist_ok=True)
    state_path = os.path.join(args.output_dir, STATE_FILE)
    state = load_state(state_path)

    datasets_filter = set(args.datasets) if args.datasets else None
    targets = discover_targets(args.input_dir, datasets_filter)
    if not targets:
        print("No datasets found to embed.")
        return

    client, types_mod = create_gemini_client(api_key)
    if args.request_mode == "batch_api" and args.batch_api_submit_all:
        run_batch_api_submit_all(targets, client, types_mod, args, state, state_path)
    else:
        for target in targets:
            process_target(target, client, types_mod, args, state, state_path)

    print("Gemini embedding complete.")


if __name__ == "__main__":
    main()
