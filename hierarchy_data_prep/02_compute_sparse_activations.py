from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_encoder(model_path: Path, device: str, dtype_name: str) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    try:
        payload = torch.load(_paper_location(model_path), map_location="cpu", mmap=True, weights_only=True)
    except TypeError:
        payload = torch.load(_paper_location(model_path), map_location="cpu")
    except Exception:
        payload = torch.load(_paper_location(model_path), map_location="cpu", mmap=True, weights_only=False)

    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict checkpoint, got {type(payload)} from {model_path}")
    state = payload.get("model_state_dict", payload.get("state_dict", payload))
    if "W_enc" not in state or "b_dec" not in state:
        raise KeyError(f"Checkpoint missing W_enc/b_dec. Keys include: {list(state)[:20]}")

    dtype = DTYPE_MAP[dtype_name]
    W_enc = state["W_enc"].detach().to(device=device, dtype=dtype)
    b_dec = state["b_dec"].detach().to(device=device, dtype=dtype)
    config = payload.get("config", {})
    return W_enc, b_dec, config if isinstance(config, dict) else {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute sparse activations for hierarchy datasets with flat Gemini SAE artifacts.")
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--rows", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top-k", required=True, type=int)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=sorted(DTYPE_MAP), default="float32")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        print(f"Output exists, skipping: {args.output}")
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    W_enc, b_dec, config = load_encoder(args.model_path, args.device, args.dtype)
    dict_size = int(W_enc.shape[1])
    input_dim = int(W_enc.shape[0])
    top_k = min(int(args.top_k), dict_size)

    embeddings = np.load(_paper_location(args.embeddings), mmap_mode="r")
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got {embeddings.shape}")
    if int(embeddings.shape[1]) != input_dim:
        raise ValueError(f"Embedding dim {embeddings.shape[1]} does not match SAE input dim {input_dim}")

    n_rows = int(embeddings.shape[0])
    row_chunks: list[np.ndarray] = []
    feat_chunks: list[np.ndarray] = []
    val_chunks: list[np.ndarray] = []

    with torch.inference_mode():
        for start in tqdm(range(0, n_rows, args.batch_size), desc=f"d{dict_size} activations"):
            end = min(start + args.batch_size, n_rows)
            x_np = np.asarray(embeddings[start:end]).copy()
            x = torch.from_numpy(x_np).to(device=args.device, dtype=W_enc.dtype)
            acts = F.relu((x - b_dec) @ W_enc)
            values, indices = torch.topk(acts, k=top_k, dim=1)
            mask = values > 0
            nz_batch_rows, nz_pos = mask.nonzero(as_tuple=True)
            if nz_batch_rows.numel() == 0:
                continue
            row_chunks.append((nz_batch_rows.cpu().numpy().astype(np.int64) + start).astype(np.int32))
            feat_chunks.append(indices[nz_batch_rows, nz_pos].cpu().numpy().astype(np.int32))
            val_chunks.append(values[nz_batch_rows, nz_pos].float().cpu().numpy().astype(np.float16))

    row_indices = np.concatenate(row_chunks) if row_chunks else np.empty(0, dtype=np.int32)
    feature_indices = np.concatenate(feat_chunks) if feat_chunks else np.empty(0, dtype=np.int32)
    values = np.concatenate(val_chunks) if val_chunks else np.empty(0, dtype=np.float16)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        _paper_location(args.output),
        row_indices=row_indices,
        feature_indices=feature_indices,
        values=values,
        shape=np.asarray([n_rows, dict_size], dtype=np.int64),
        top_k=np.asarray([top_k], dtype=np.int64),
    )
    meta = {
        "model_path": str(args.model_path),
        "embeddings": str(args.embeddings),
        "rows": str(args.rows),
        "output": str(args.output),
        "n_rows": n_rows,
        "dict_size": dict_size,
        "top_k": top_k,
        "batch_size": int(args.batch_size),
        "dtype": args.dtype,
        "nnz": int(values.shape[0]),
        "value_dtype": str(values.dtype),
        "row_index_dtype": str(row_indices.dtype),
        "feature_index_dtype": str(feature_indices.dtype),
        "checkpoint_config": config,
    }
    args.output.with_suffix(args.output.suffix + ".meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True, default=str))
    print(json.dumps(meta, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
