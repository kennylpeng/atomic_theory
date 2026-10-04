from project_paths import resource_path as _paper_path, resource_location as _paper_location
import sys
from pathlib import Path
sys.path.insert(0, str(_paper_path(__file__).resolve().parents[1]))
from project_paths import get_path

import argparse
import gc
import json
import os
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

INPUT_DATASET_DIR = get_path("datasets_dir")
OUTPUT_EMBED_DIR = get_path("nemotron_embedding_output_dir")
STATE_FILE = "embedding_state.json"
MAX_TEXT_CHARS = 10_000

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed saved datasets into mmap-friendly shards for large-scale training."
    )
    parser.add_argument("--input-dir", default=INPUT_DATASET_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_EMBED_DIR)
    parser.add_argument(
        "--model-id",
        default="nvidia/llama-embed-nemotron-8b",
        help="SentenceTransformers model ID.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
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
        "--device",
        default="cuda",
        help='Device for SentenceTransformer (default: "cuda").',
    )
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="float16",
        help='Model dtype for loading on GPU (default: "float16").',
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=256,
        help="Maximum tokenized sequence length for encoding (default: 256).",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize embeddings to unit length before saving.",
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
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help=(
            "Optional config names within multi-config datasets "
            "(e.g. en fr for miracl-corpus-text/config=en)."
        ),
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


def discover_targets(
    input_dir: str,
    datasets_filter: set[str] | None,
    configs_filter: set[str] | None = None,
) -> list[SavedDatasetTarget]:
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
                if configs_filter and config_name not in configs_filter:
                    continue
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
        # Keep scalar columns only; skip nested/list columns for compact metadata.
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


def encode_shard(
    model: SentenceTransformer,
    split_ds: Dataset,
    text_col: str,
    batch_size: int,
    normalize: bool,
    dtype: np.dtype,
    start_idx: int,
    end_idx: int,
) -> np.ndarray:
    rows = end_idx - start_idx
    chunks: list[np.ndarray] = []
    current_batch_size = batch_size
    batch_start = start_idx
    while batch_start < end_idx:
        batch_end = min(batch_start + current_batch_size, end_idx)
        batch = split_ds[batch_start:batch_end]
        texts = [
            ("" if t is None else str(t))[:MAX_TEXT_CHARS]
            for t in batch[text_col]
        ]
        try:
            embs = model.encode(
                texts,
                batch_size=current_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
            )
            chunks.append(embs.astype(dtype, copy=False))
            batch_start = batch_end
        except torch.OutOfMemoryError:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if current_batch_size == 1:
                raise RuntimeError(
                    "CUDA OOM even at batch_size=1. "
                    "Try reducing --max-seq-length, using a smaller model, or a larger GPU."
                )
            next_batch_size = max(1, current_batch_size // 2)
            print(
                f"OOM at batch_size={current_batch_size}. "
                f"Retrying with batch_size={next_batch_size}."
            )
            current_batch_size = next_batch_size
    if not chunks:
        return np.empty((0, 0), dtype=dtype)
    shard_embs = np.concatenate(chunks, axis=0)
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
    normalize: bool,
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
        "normalize_embeddings": normalize,
        "text_column": text_col,
        "embedding_file": "embeddings.npy",
        "metadata_file": "metadata.parquet",
    }
    with open(_paper_location(shard_meta_path), "w", encoding="utf-8") as f:
        json.dump(shard_meta, f, indent=2, sort_keys=True)


def process_target(
    target: SavedDatasetTarget,
    model: SentenceTransformer,
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
                f"rows={num_rows}, text_col={text_col}, shards={num_shards}"
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

                embeddings = encode_shard(
                    model=model,
                    split_ds=split_ds,
                    text_col=text_col,
                    batch_size=args.batch_size,
                    normalize=args.normalize,
                    dtype=dtype,
                    start_idx=start_idx,
                    end_idx=end_idx,
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
                    normalize=args.normalize,
                )
                state["completed_shards"][rel_key] = "completed"
                save_state(state_path, state)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    state_path = os.path.join(args.output_dir, STATE_FILE)
    state = load_state(state_path)

    datasets_filter = set(args.datasets) if args.datasets else None
    configs_filter = set(args.configs) if args.configs else None
    targets = discover_targets(args.input_dir, datasets_filter, configs_filter)
    if not targets:
        print("No datasets found to embed.")
        return

    if configs_filter:
        print(f"Config filter: {', '.join(sorted(configs_filter))}")
    print(f"Targets ({len(targets)}):")
    for target in targets:
        print(f"  - {target.output_rel_path}")

    print(f"Loading model: {args.model_id}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested but not available. "
            "This pipeline is configured for GPU runs."
        )
    torch_dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    model_kwargs = {
        "trust_remote_code": True,
        "model_kwargs": {"torch_dtype": torch_dtype_map[args.torch_dtype]},
    }
    model_kwargs["device"] = args.device
    model = SentenceTransformer(args.model_id, **model_kwargs)
    if args.max_seq_length:
        model.max_seq_length = args.max_seq_length

    for target in targets:
        process_target(target, model, args, state, state_path)

    print("Embedding complete.")


if __name__ == "__main__":
    main()
