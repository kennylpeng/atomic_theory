#!/usr/bin/env python3
"""Reproduce the day-of-year activation plots and 2D PCA experiment."""

from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

ROOT = _paper_path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from trainer.sae import BatchTopKSAE, JumpReLUSAE, TopKSAE, VanillaSAE


DEFAULT_GEMINI_RUN = _paper_path("/resources/date_reference_models_dir/d65536_k128_f32")
DEFAULT_NEMOTRON_RUN = _paper_path(
    "/resources/date_reference_models_dir/d65536_k128_f32_nemotron"
)
GEMINI_FEATURES = [
    54190, 11053, 29452, 13219, 61549, 45772,
    63515, 11312, 18738, 17281, 44782, 15540,
]


def dates() -> list[str]:
    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    counts = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return [f"{month} {day}" for month, count in zip(months, counts) for day in range(1, count + 1)]


def load_sae(run_dir: Path, device: str) -> torch.nn.Module:
    model_map = {
        "topk": TopKSAE,
        "batch_topk": BatchTopKSAE,
        "vanilla": VanillaSAE,
        "jumprelu": JumpReLUSAE,
    }
    dtype_map = {
        "float32": torch.float32, "torch.float32": torch.float32,
        "float16": torch.float16, "torch.float16": torch.float16,
        "bfloat16": torch.bfloat16, "torch.bfloat16": torch.bfloat16,
    }
    with (run_dir / "runtime_config.json").open() as handle:
        runtime = json.load(handle)
    payload = torch.load(_paper_location(run_dir / "final_model.pt"), map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload.get("state_dict", payload))
    checkpoint_cfg = payload.get("config", {}) if isinstance(payload, dict) else {}
    cfg = {**runtime, **checkpoint_cfg, "device": device}
    if isinstance(cfg.get("dtype"), str):
        cfg["dtype"] = dtype_map[cfg["dtype"]]
    sae = model_map[str(cfg.get("model_type", "topk")).lower()](cfg)
    sae.load_state_dict(state, strict=True)
    return sae.eval().to(device)


def feature_acts(sae: torch.nn.Module, embeddings: np.ndarray, batch_size: int = 64) -> np.ndarray:
    result = []
    device = next(sae.parameters()).device
    for start in range(0, len(embeddings), batch_size):
        x = torch.from_numpy(embeddings[start : start + batch_size]).to(
            device=device, dtype=sae.W_enc.dtype
        )
        with torch.inference_mode():
            result.append(sae(x)["feature_acts"].float().cpu().numpy())
    return np.concatenate(result)


def gemini_embeddings(texts: list[str]) -> np.ndarray:
    import google.generativeai as genai

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY")
    genai.configure(api_key=key)
    rows: list[list[float]] = []
    for start in range(0, len(texts), 100):
        response = genai.embed_content(
            model="gemini-embedding-2-preview",
            content=texts[start : start + 100],
            task_type="RETRIEVAL_DOCUMENT",
        )
        embedding = response["embedding"] if isinstance(response, dict) else response.embedding
        rows.extend(embedding)
    return np.asarray(rows, dtype=np.float32)


def nemotron_embeddings(texts: list[str], device: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        "nvidia/llama-embed-nemotron-8b",
        trust_remote_code=True,
        device=device,
        model_kwargs={"attn_implementation": "eager", "torch_dtype": "bfloat16"},
        processor_kwargs={"padding_side": "left"},
    )
    return np.asarray(model.encode_document(texts, batch_size=8, show_progress_bar=True), dtype=np.float32)


def compute(name: str, texts: list[str], run_dir: Path, output: Path, device: str) -> np.ndarray:
    cache = output / f"{name}_activations.npz"
    if cache.exists():
        data = np.load(_paper_location(cache))
        if data["texts"].tolist() != texts:
            raise RuntimeError(f"Text mismatch in {cache}")
        print(f"Using cached {name} activations: {cache}")
        return data["activations"]
    embedding_cache = output / f"{name}_embeddings.npz"
    if embedding_cache.exists():
        embeddings = np.load(_paper_location(embedding_cache))["embeddings"]
    elif (output / f"{name}_embeddings.npy").exists():
        embeddings = np.load(_paper_location(output / f"{name}_embeddings.npy"))
    else:
        embeddings = gemini_embeddings(texts) if name == "gemini" else nemotron_embeddings(texts, device)
        np.savez_compressed(_paper_location(embedding_cache), texts=np.asarray(texts), embeddings=embeddings)
    print(f"Loading {name} SAE from {run_dir}")
    sae = load_sae(run_dir, device)
    activations = feature_acts(sae, embeddings)
    np.savez_compressed(_paper_location(cache), texts=np.asarray(texts), activations=activations)
    del sae
    return activations


def plot_traces(acts: np.ndarray, features: list[int], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    for feature in features:
        ax.plot(np.arange(1, 367), acts[:, feature], label=str(feature), linewidth=1.2)
    ax.set(xlabel="Day index (January 1 = 1)", ylabel="SAE activation", title=title)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_pca(acts: np.ndarray, features: list[int], path: Path, array_path: Path, title: str) -> list[float]:
    pca = PCA(n_components=2).fit(acts[:, features])
    projected = pca.transform(acts[:, features])
    np.save(_paper_location(array_path), projected)
    fig, ax = plt.subplots(figsize=(8, 7))
    points = ax.scatter(projected[:, 0], projected[:, 1], c=np.arange(366), cmap="hsv", s=18)
    ax.plot(projected[:, 0], projected[:, 1], alpha=0.25, linewidth=0.7)
    ax.set(xlabel="PC1", ylabel="PC2", title=title)
    fig.colorbar(points, ax=ax, label="Day index")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return pca.explained_variance_ratio_.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gemini", "nemotron", "both"], default="both")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/day_of_year/results")
    parser.add_argument("--gemini-run", type=Path, default=DEFAULT_GEMINI_RUN)
    parser.add_argument("--nemotron-run", type=Path, default=DEFAULT_NEMOTRON_RUN)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    texts = dates()
    (args.output / "dates.txt").write_text("\n".join(texts) + "\n")
    summary: dict[str, Any] = {"n_dates": len(texts), "gemini_features": GEMINI_FEATURES}
    gemini = nemotron = None
    if args.model in {"gemini", "both"}:
        gemini = compute("gemini", texts, args.gemini_run, args.output, args.device)
        plot_traces(gemini, GEMINI_FEATURES, args.output / "gemini_feature_traces.png", "Gemini SAE features across days of the year")
        summary["gemini_activation_shape"] = list(gemini.shape)
        summary["gemini_pca_explained_variance_ratio"] = plot_pca(
            gemini, GEMINI_FEATURES,
            args.output / "gemini_selected_features_pca2.png",
            args.output / "gemini_selected_features_pca2.npy",
            "2D PCA of selected Gemini SAE feature activations",
        )
    if args.model in {"nemotron", "both"}:
        nemotron = compute("nemotron", texts, args.nemotron_run, args.output, args.device)
        summary["nemotron_activation_shape"] = list(nemotron.shape)
        nemotron_features = np.argsort(nemotron.var(axis=0))[-12:][::-1].tolist()
        summary["nemotron_high_variance_features"] = nemotron_features
        plot_traces(nemotron, nemotron_features, args.output / "nemotron_feature_traces.png", "High-variance Nemotron SAE features across days of the year")
        summary["nemotron_pca_explained_variance_ratio"] = plot_pca(
            nemotron, nemotron_features,
            args.output / "nemotron_selected_features_pca2.png",
            args.output / "nemotron_selected_features_pca2.npy",
            "2D PCA of selected Nemotron SAE feature activations",
        )
    if gemini is not None and nemotron is not None:
        matches = []
        matched_features = []
        centered_n = nemotron - nemotron.mean(axis=0, keepdims=True)
        n_norm = np.linalg.norm(centered_n, axis=0)
        for feature in GEMINI_FEATURES:
            g = gemini[:, feature] - gemini[:, feature].mean()
            corr = (g @ centered_n) / np.maximum(np.linalg.norm(g) * n_norm, 1e-12)
            match = int(np.nanargmax(corr))
            matched_features.append(match)
            matches.append({"gemini": feature, "nemotron": match, "correlation": float(corr[match])})
        plot_traces(nemotron, matched_features, args.output / "nemotron_matched_feature_traces.png", "Best-matching Nemotron SAE features across days of the year")
        summary["activation_correlation_matches"] = matches
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
