from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


MAX_TEXT_CHARS = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed a hierarchy rows.csv file with Nemotron.")
    parser.add_argument("--rows", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--text-column", required=True)
    parser.add_argument("--model-id", default="nvidia/llama-embed-nemotron-8b")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def count_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as f:
        return max(0, sum(1 for _ in f) - 1)


def iter_text_batches(path: Path, text_column: str, batch_size: int):
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or text_column not in reader.fieldnames:
            raise ValueError(f"{path} does not contain text column {text_column!r}; columns={reader.fieldnames}")
        batch: list[str] = []
        for row in reader:
            value = row.get(text_column)
            batch.append(("" if value is None else str(value))[:MAX_TEXT_CHARS])
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def torch_dtype(name: str):
    return {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def encode_with_retry(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
    normalize: bool,
    dtype: np.dtype,
) -> np.ndarray:
    current_batch_size = batch_size
    while True:
        try:
            embeddings = model.encode(
                texts,
                batch_size=current_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
            )
            return embeddings.astype(dtype, copy=False)
        except torch.OutOfMemoryError:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if current_batch_size == 1:
                raise
            current_batch_size = max(1, current_batch_size // 2)
            print(f"OOM while encoding batch; retrying with batch_size={current_batch_size}", flush=True)


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        print(f"Output exists, skipping: {args.output}")
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    n_rows = count_rows(args.rows)
    if n_rows == 0:
        raise ValueError(f"No rows found in {args.rows}")

    output_dtype = np.float16 if args.dtype == "float16" else np.float32
    tmp_output = args.output.with_name(args.output.name + ".tmp")
    if tmp_output.exists():
        tmp_output.unlink()

    print(f"Loading model: {args.model_id}", flush=True)
    model = SentenceTransformer(
        args.model_id,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch_dtype(args.torch_dtype)},
        device=args.device,
    )
    if args.max_seq_length:
        model.max_seq_length = args.max_seq_length

    mmap: np.memmap | None = None
    written = 0
    dim: int | None = None
    for texts in tqdm(iter_text_batches(args.rows, args.text_column, args.batch_size), total=(n_rows + args.batch_size - 1) // args.batch_size):
        embs = encode_with_retry(model, texts, args.batch_size, args.normalize, output_dtype)
        if mmap is None:
            dim = int(embs.shape[1])
            args.output.parent.mkdir(parents=True, exist_ok=True)
            mmap = np.lib.format.open_memmap(tmp_output, mode="w+", dtype=output_dtype, shape=(n_rows, dim))
        end = written + int(embs.shape[0])
        mmap[written:end] = embs
        written = end

    if mmap is None or dim is None:
        raise RuntimeError("No embeddings were written.")
    mmap.flush()
    del mmap
    if written != n_rows:
        raise RuntimeError(f"Expected to write {n_rows} rows, wrote {written}")
    tmp_output.rename(args.output)

    meta: dict[str, Any] = {
        "rows": str(args.rows),
        "output": str(args.output),
        "text_column": args.text_column,
        "model_id": args.model_id,
        "n_rows": n_rows,
        "dim": dim,
        "dtype": args.dtype,
        "torch_dtype": args.torch_dtype,
        "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size,
        "normalize_embeddings": args.normalize,
        "device": args.device,
    }
    args.output.with_suffix(args.output.suffix + ".meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
