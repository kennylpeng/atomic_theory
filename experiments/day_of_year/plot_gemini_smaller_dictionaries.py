#!/usr/bin/env python3
"""Plot day-of-year activations for 65K Gemini features matched at smaller widths.

The selected day-of-year feature IDs belong to the
seed-42 65,536-feature checkpoint. For each smaller checkpoint from that
training family, this script finds the decoder row with maximum cosine
similarity to each selected 65K decoder row, computes the matched post-TopK
activations on the cached date embeddings, and reproduces the trace and PCA
plots.
"""

from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

ROOT = _paper_path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.day_of_year.run import GEMINI_FEATURES, dates


DATE_REFERENCE_ROOT = _paper_path("/resources/date_reference_models_dir")
DEFAULT_REFERENCE_RUN = DATE_REFERENCE_ROOT / "d65536_k128_f32"
DEFAULT_SMALLER_RUNS = {
    1024: DATE_REFERENCE_ROOT / "d1024_k16_f32",
    2048: DATE_REFERENCE_ROOT / "d2048_k16_f32",
    4096: DATE_REFERENCE_ROOT / "d4096_k32_f32",
    8192: DATE_REFERENCE_ROOT / "d8192_k32_f32",
    16384: DATE_REFERENCE_ROOT / "d16384_k32_f32",
    32768: DATE_REFERENCE_ROOT / "d32768_k64_f32",
}
NEWER_131K_CHECKPOINT = _paper_path(
    "/resources/models_dir/gemini_m131072_k128"
)
AVAILABLE_TARGETS = {**DEFAULT_SMALLER_RUNS, 131072: NEWER_131K_CHECKPOINT}


def load_checkpoint(checkpoint_or_run: Path) -> dict[str, Any]:
    path = (
        checkpoint_or_run / "final_model.pt"
        if checkpoint_or_run.is_dir()
        else checkpoint_or_run
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        return torch.load(_paper_location(path), map_location="cpu", mmap=True, weights_only=False)
    except TypeError:
        return torch.load(_paper_location(path), map_location="cpu", weights_only=False)


def reference_decoder_rows(run_dir: Path, features: list[int]) -> torch.Tensor:
    checkpoint = load_checkpoint(run_dir)
    decoder = checkpoint["model_state_dict"]["W_dec"]
    if max(features) >= decoder.shape[0]:
        raise ValueError(
            f"Reference feature {max(features)} is outside decoder shape {tuple(decoder.shape)}"
        )
    rows = decoder[features].detach().float().clone()
    del decoder, checkpoint
    gc.collect()
    return F.normalize(rows, dim=1)


def best_decoder_matches(
    reference_rows: torch.Tensor, decoder: torch.Tensor
) -> tuple[list[int], list[float]]:
    candidate_rows = F.normalize(decoder.detach().float(), dim=1)
    similarities = reference_rows @ candidate_rows.T
    scores, indices = similarities.max(dim=1)
    return indices.tolist(), scores.tolist()


@torch.inference_mode()
def selected_topk_activations(
    embeddings: np.ndarray,
    state: dict[str, torch.Tensor],
    top_k: int,
    features: list[int],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Compute only requested columns of exact post-TopK activations."""
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if max(features) >= state["W_enc"].shape[1]:
        raise ValueError(
            f"Matched feature {max(features)} is outside encoder shape "
            f"{tuple(state['W_enc'].shape)}"
        )

    weight = state["W_enc"].detach().to(device=device, dtype=torch.float32)
    bias = state["b_dec"].detach().to(device=device, dtype=torch.float32)
    feature_index = torch.tensor(features, device=device, dtype=torch.long)
    result: list[np.ndarray] = []

    for start in range(0, len(embeddings), batch_size):
        x = torch.from_numpy(embeddings[start : start + batch_size]).to(
            device=device, dtype=torch.float32
        )
        pre_topk = torch.relu((x - bias) @ weight)
        top_indices = torch.topk(pre_topk, top_k, dim=1).indices
        selected = pre_topk.index_select(1, feature_index)
        selected_in_topk = (
            top_indices.unsqueeze(2) == feature_index.view(1, 1, -1)
        ).any(dim=1)
        selected = selected * selected_in_topk
        result.append(selected.cpu().numpy())

    return np.concatenate(result, axis=0)


def plot_traces(
    activations: np.ndarray,
    reference_features: list[int],
    matched_features: list[int],
    path: Path,
    width: int,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(1, len(activations) + 1)
    for column, (reference, matched) in enumerate(
        zip(reference_features, matched_features)
    ):
        ax.plot(
            x,
            activations[:, column],
            label=f"{reference}→{matched}",
            linewidth=1.2,
        )
    ax.set(
        xlabel="Day index (January 1 = 1)",
        ylabel="SAE activation",
        title=f"Best-matching Gemini SAE features across days (m={width:,})",
    )
    ax.legend(ncol=3, fontsize=8, title="65K→matched feature", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_pca(
    activations: np.ndarray,
    path: Path,
    array_path: Path,
    width: int,
) -> list[float]:
    pca = PCA(n_components=2).fit(activations)
    projected = pca.transform(activations)
    np.save(_paper_location(array_path), projected)

    fig, ax = plt.subplots(figsize=(8, 7))
    points = ax.scatter(
        projected[:, 0],
        projected[:, 1],
        c=np.arange(len(projected)),
        cmap="hsv",
        s=18,
    )
    ax.plot(projected[:, 0], projected[:, 1], alpha=0.25, linewidth=0.7)
    ax.set(
        xlabel="PC1",
        ylabel="PC2",
        title=f"2D PCA of matched Gemini SAE feature activations (m={width:,})",
    )
    fig.colorbar(points, ax=ax, label="Day index")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return pca.explained_variance_ratio_.tolist()


def parse_widths(value: str) -> list[int]:
    widths = [int(item) for item in value.split(",") if item.strip()]
    unknown = sorted(set(widths) - set(AVAILABLE_TARGETS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown widths {unknown}; choose from {sorted(AVAILABLE_TARGETS)}"
        )
    if not widths:
        raise argparse.ArgumentTypeError("at least one width is required")
    return widths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=ROOT / "experiments/day_of_year/results/gemini_embeddings.npy",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/day_of_year/results/gemini_smaller_dictionaries",
    )
    parser.add_argument("--reference-run", type=Path, default=DEFAULT_REFERENCE_RUN)
    parser.add_argument(
        "--widths",
        type=parse_widths,
        default=sorted(DEFAULT_SMALLER_RUNS),
        help="Comma-separated widths (default: all seed-42 Gemini reference SAEs below 65K)",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    embeddings = np.load(_paper_location(args.embeddings))
    expected_dates = dates()
    if embeddings.ndim != 2 or len(embeddings) != len(expected_dates):
        raise ValueError(
            f"Expected {len(expected_dates)} embedding rows, got {embeddings.shape}"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    reference_rows = reference_decoder_rows(args.reference_run, GEMINI_FEATURES)
    device = torch.device(args.device)
    summary_path = args.output / "summary.json"
    if summary_path.exists():
        summary: dict[str, Any] = json.loads(summary_path.read_text())
    else:
        summary = {
            "reference_width": 65536,
            "reference_run": str(args.reference_run),
            "reference_features": GEMINI_FEATURES,
            "matching_method": "maximum cosine similarity between decoder rows",
            "widths": {},
        }

    for width in args.widths:
        target = AVAILABLE_TARGETS[width]
        print(f"Loading Gemini m={width:,} from {target}", flush=True)
        checkpoint = load_checkpoint(target)
        state = checkpoint["model_state_dict"]
        config = checkpoint.get("config", {})
        actual_width = int(state["W_dec"].shape[0])
        if actual_width != width:
            raise ValueError(
                f"{target} has decoder width {actual_width}, expected {width}"
            )
        if str(config.get("model_type", "topk")).lower() != "topk":
            raise ValueError(f"{target} is not a TopK SAE")
        top_k = int(config["top_k"])
        matched_features, cosine_similarities = best_decoder_matches(
            reference_rows, state["W_dec"]
        )
        activations = selected_topk_activations(
            embeddings,
            state,
            top_k,
            matched_features,
            device,
            args.batch_size,
        )

        width_output = args.output / f"m{width}"
        width_output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            _paper_location(width_output / "matched_feature_activations.npz"),
            dates=np.asarray(expected_dates),
            reference_features=np.asarray(GEMINI_FEATURES),
            matched_features=np.asarray(matched_features),
            decoder_cosine_similarities=np.asarray(cosine_similarities),
            activations=activations,
        )
        plot_traces(
            activations,
            GEMINI_FEATURES,
            matched_features,
            width_output / "gemini_matched_feature_traces.png",
            width,
        )
        explained_variance = plot_pca(
            activations,
            width_output / "gemini_matched_features_pca2.png",
            width_output / "gemini_matched_features_pca2.npy",
            width,
        )
        record = {
            "checkpoint": str(target),
            "training_family": "main width sweep" if width == 131072 else "seed-42 date reference",
            "top_k": top_k,
            "matched_features": matched_features,
            "decoder_cosine_similarities": cosine_similarities,
            "pca_explained_variance_ratio": explained_variance,
            "activation_shape": list(activations.shape),
        }
        (width_output / "summary.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        summary["widths"][str(width)] = record
        print(
            f"m={width:,}: wrote trace and PCA plots; "
            f"mean match cosine={np.mean(cosine_similarities):.4f}",
            flush=True,
        )
        del activations, state, checkpoint
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
