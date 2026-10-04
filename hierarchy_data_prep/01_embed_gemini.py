#!/usr/bin/env python3
"""Embed a CSV text column with Gemini and write a single NumPy array."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm


EMPTY_TEXT_PLACEHOLDER = "[EMPTY]"
MAX_TEXT_CHARS = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, )
    parser.add_argument("--rows", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--model-id", default="gemini-embedding-2-preview")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--task-type", default="RETRIEVAL_DOCUMENT")
    parser.add_argument("--output-dimensionality", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--retry-backoff-seconds", type=float, default=60.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def create_gemini_client(api_key: str):
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'google-genai'. Install with: pip install google-genai") from exc
    return genai.Client(api_key=api_key), types


def extract_vectors(response: Any) -> list[list[float]]:
    embeddings = getattr(response, "embeddings", None)
    if embeddings is None:
        raise RuntimeError("Gemini API response missing embeddings")
    vectors = []
    for emb in embeddings:
        values = getattr(emb, "values", None)
        if values is None:
            raise RuntimeError("Gemini API embedding item missing values")
        vectors.append(list(values))
    return vectors


def should_retry(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in ["429", "resource_exhausted", "rate limit", "deadline exceeded", "timeout", "internal", "unavailable", "503"]
    )


def sanitize_text(value: Any) -> str:
    if value is None:
        return EMPTY_TEXT_PLACEHOLDER
    text = str(value)[:MAX_TEXT_CHARS].strip()
    return text or EMPTY_TEXT_PLACEHOLDER


def embed_batch_with_retry(
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
            # A list[str] is interpreted as multiple parts of one Content by some
            # google-genai/model combinations. Construct one Content per row so a
            # batch always produces one embedding per input text.
            contents = [
                types_mod.Content(parts=[types_mod.Part(text=text)])
                for text in texts
            ]
            response = client.models.embed_content(model=model_id, contents=contents, config=config)
            vectors = extract_vectors(response)
            if len(vectors) != len(texts):
                raise RuntimeError(
                    f"Gemini returned {len(vectors)} embeddings for {len(texts)} inputs"
                )
            return np.asarray(vectors, dtype=np.float32)
        except Exception as exc:
            attempt += 1
            if attempt > max_retries or not should_retry(exc):
                raise
            sleep_s = retry_backoff_seconds * (2 ** (attempt - 1))
            label = f"{sleep_s / 60:.1f}m" if sleep_s >= 60 else f"{sleep_s:.1f}s"
            print(f"Gemini request failed attempt {attempt}/{max_retries}: {exc}. Retrying in {label}...", flush=True)
            time.sleep(sleep_s)


def read_texts(path: Path, text_column: str) -> list[str]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or text_column not in reader.fieldnames:
            raise ValueError(f"{path} lacks text column {text_column!r}; columns={reader.fieldnames}")
        return [sanitize_text(row.get(text_column)) for row in reader]


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        print(f"Output exists, skipping: {args.output}")
        return
    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Gemini API key is required. Pass --api-key or set GEMINI_API_KEY.")

    texts = read_texts(args.rows, args.text_column)
    if not texts:
        raise ValueError(f"No texts in {args.rows}")
    client, types_mod = create_gemini_client(api_key)
    output_dtype = np.float16 if args.dtype == "float16" else np.float32

    batches = [(i, texts[i : i + args.batch_size]) for i in range(0, len(texts), args.batch_size)]
    results: dict[int, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        pending = {
            executor.submit(
                embed_batch_with_retry,
                client,
                types_mod,
                args.model_id,
                args.task_type,
                args.output_dimensionality,
                batch,
                args.max_retries,
                args.retry_backoff_seconds,
            ): start
            for start, batch in batches
        }
        with tqdm(total=len(pending), desc="gemini batches") as progress:
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    start = pending.pop(future)
                    results[start] = future.result().astype(output_dtype, copy=False)
                    progress.update(1)

    ordered_starts = sorted(results)
    dim = int(results[ordered_starts[0]].shape[1])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = args.output.with_name(args.output.name + ".tmp")
    if tmp_output.exists():
        tmp_output.unlink()
    mmap = np.lib.format.open_memmap(tmp_output, mode="w+", dtype=output_dtype, shape=(len(texts), dim))
    for start in ordered_starts:
        arr = results[start]
        mmap[start : start + arr.shape[0]] = arr
    mmap.flush()
    del mmap
    tmp_output.rename(args.output)
    meta = {
        "rows": str(args.rows),
        "output": str(args.output),
        "text_column": args.text_column,
        "model_id": args.model_id,
        "task_type": args.task_type,
        "n_rows": len(texts),
        "dim": dim,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "max_workers": args.max_workers,
    }
    args.output.with_suffix(args.output.suffix + ".meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
