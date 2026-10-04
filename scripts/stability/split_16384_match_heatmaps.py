#!/usr/bin/env python
"""Exclusive split matches: absolute cosine for PCA, signed cosine otherwise."""

from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import sys
from pathlib import Path
sys.path.insert(0, str(_paper_path(__file__).resolve().parents[2]))
from project_paths import get_path
from scripts.plot_style import apply_plot_style

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from dictionary_exclusive_matches import exclusive_matches as count_exclusive_matches
from utils import load_dictionary


SPLITS: tuple[str, ...] = ("main", "wikipedia", "no_wikipedia", "random1", "random2")
REPRESENTATIONS: tuple[str, ...] = ("sae", "kmeans", "pca")
REP_LABELS = {
    "sae": "SAE decoder",
    "kmeans": "KMeans m16384 centroids",
    "pca": "PCA directions",
}


def artifact_path(models_dir: Path, model: str, split: str, representation: str) -> Path:
    prefix = model if split == "main" else f"{model}_{split}"
    if representation == "sae":
        return models_dir / (f"{model}_m16384_k64" if split == "main" else f"{prefix}_m16384_k64")
    if representation == "kmeans":
        return models_dir / f"{prefix}_kmeans_pca256_m16384"
    if representation == "pca":
        return models_dir / f"{prefix}_pca_components"
    raise ValueError(f"Unknown representation: {representation}")


def model_paths(models_dir: Path, model: str, representation: str) -> list[tuple[str, Path]]:
    paths = [(split, artifact_path(models_dir, model, split, representation)) for split in SPLITS]
    missing = [path for _, path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing split model artifact(s): " + ", ".join(str(p) for p in missing))
    return paths


def load_pca_directions(path: Path) -> torch.Tensor:
    components = np.load(_paper_location(path), mmap_mode="r")
    if components.ndim != 2:
        raise ValueError(f"Expected 2D PCA component array in {path}, got {tuple(components.shape)}")
    directions = torch.from_numpy(np.array(components, dtype=np.float32, copy=True)).cpu()
    return F.normalize(directions, dim=1)


def load_vectors(
    path: Path,
    representation: str,
    *,
    sae_direction: str = "decoder",
) -> torch.Tensor:
    if representation == "sae":
        return load_dictionary(path, "sae", sae_direction=sae_direction)
    if representation == "kmeans":
        return load_dictionary(path, "kmeans")
    if representation == "pca":
        return load_pca_directions(path)
    raise ValueError(f"Unknown representation: {representation}")


def output_stem(model: str, representation: str, sae_direction: str) -> str:
    if representation == "sae":
        direction = "" if sae_direction == "decoder" else "_encoder"
        return f"{model}_16384{direction}_split_match"
    return f"{model}_{representation}_split_match"


def plot_heatmap(
    matrix: np.ndarray,
    labels: list[str],
    model: str,
    representation: str,
    out_path: Path,
    *,
    sae_direction: str = "decoder",
) -> None:
    apply_plot_style()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    masked = np.ma.array(matrix, mask=np.eye(matrix.shape[0], dtype=bool))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#f2f2f2")
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=max(1, int(matrix.max())))
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    representation_label = (
        f"SAE {sae_direction}" if representation == "sae" else REP_LABELS[representation]
    )
    ax.set_title(f"{model} {representation_label}\nfeatures with a match")
    ax.set_xlabel("comparison split")
    ax.set_ylabel("source split")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i == j:
                continue
            value = int(matrix[i, j])
            r, g, b, _ = im.cmap(im.norm(value))
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
            color = "black" if luminance > 0.55 else "white"
            ax.text(j, i, f"{value:,}", ha="center", va="center", color=color, fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("source features with a match")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)



def plot_combined_heatmaps(
    matrices: dict[tuple[str, str], np.ndarray],
    feature_counts: dict[tuple[str, str], int],
    labels: list[str],
    out_path: Path,
    *,
    color_max: float | None = None,
) -> None:
    """Plot all model/representation heatmaps with one proportional color scale."""
    apply_plot_style()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    models = ("gemini", "nemotron")
    representations = ("sae", "kmeans", "pca")
    column_labels = {"sae": "SAE", "kmeans": "KMeans", "pca": "PCA"}
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 10.0), sharex=True, sharey=True)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#f2f2f2")
    proportions = {
        key: matrix / feature_counts[key]
        for key, matrix in matrices.items()
    }
    off_diagonal_max = max(
        float(values[~np.eye(values.shape[0], dtype=bool)].max())
        for values in proportions.values()
    )
    if color_max is not None:
        off_diagonal_max = color_max
    image = None

    for row, model in enumerate(models):
        for column, representation in enumerate(representations):
            ax = axes[row, column]
            matrix = proportions[(model, representation)]
            masked = np.ma.array(matrix, mask=np.eye(matrix.shape[0], dtype=bool))
            image = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=off_diagonal_max)
            if row == 0:
                ax.set_title(column_labels[representation], fontweight="bold", fontsize=18)
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.tick_params(axis="both", labelsize=18)
            if row == len(models) - 1:
                ax.set_xticklabels(labels, rotation=35, ha="right")
            else:
                ax.tick_params(axis="x", labelbottom=False)
            if column == 0:
                ax.set_yticklabels(labels)
            else:
                ax.tick_params(axis="y", labelleft=False)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if i == j:
                        continue
                    value = matrix[i, j]
                    r, g, b, _ = image.cmap(image.norm(value))
                    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                    color = "black" if luminance > 0.55 else "white"
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=18)

    fig.text(0.018, 0.70, "Gemini", rotation=90, va="center", ha="center", fontsize=18, fontweight="bold")
    fig.text(0.018, 0.29, "Nemotron", rotation=90, va="center", ha="center", fontsize=18, fontweight="bold")
    fig.subplots_adjust(left=0.17, right=0.90, bottom=0.18, top=0.92, wspace=0.05, hspace=0.05)
    colorbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.025)
    colorbar.set_label("matching feature proportion", fontsize=18)
    colorbar.ax.tick_params(labelsize=18)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def parse_representations(value: str) -> list[str]:
    if value == "all":
        return list(REPRESENTATIONS)
    reps = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(reps) - set(REPRESENTATIONS))
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown representation(s): {', '.join(unknown)}")
    return reps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=_paper_path(get_path("models_dir")))
    parser.add_argument("--out-dir", type=Path, default=_paper_path("full_experiments/matches"))
    parser.add_argument("--plot-dir", type=Path, default=_paper_path("full_experiments/plots"))
    parser.add_argument("--representations", type=parse_representations, default=parse_representations("sae"), help="Comma-separated subset of sae,kmeans,pca, or all.")
    parser.add_argument("--t1", type=float, default=0.7)
    parser.add_argument("--t2", type=float, default=0.7)
    parser.add_argument("--src-batch", type=int, default=512)
    parser.add_argument("--dst-batch", type=int, default=16384)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--sae-direction",
        choices=("decoder", "encoder"),
        default="decoder",
        help="Use W_dec rows or transposed W_enc columns for SAE matching.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU", flush=True)
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    print(f"Using device: {device}", flush=True)

    all_rows: list[dict[str, object]] = []
    combined_matrices: dict[tuple[str, str], np.ndarray] = {}
    combined_feature_counts: dict[tuple[str, str], int] = {}
    combined_labels: list[str] | None = None
    for representation in args.representations:
        rows: list[dict[str, object]] = []
        for model in ("gemini", "nemotron"):
            paths = model_paths(args.models_dir, model, representation)
            labels = [label for label, _ in paths]
            path_by_label = dict(paths)
            vectors = {}
            for label, path in paths:
                print(f"Loading {model} {representation} {label}: {path}", flush=True)
                vectors[label] = load_vectors(
                    path,
                    representation,
                    sae_direction=args.sae_direction,
                )

            matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
            for i, left_label in enumerate(labels):
                for j, right_label in enumerate(labels[i + 1 :], start=i + 1):
                    start = time.time()
                    print(f"Computing {model} {representation} {left_label} -> {right_label}", flush=True)
                    matched_u, _matched_v = count_exclusive_matches(
                        vectors[left_label],
                        vectors[right_label],
                        args.t1,
                        args.t2,
                        args.src_batch,
                        args.dst_batch,
                        device,
                        absolute=representation == "pca",
                    )
                    left_matches = len(set(matched_u))
                    right_matches = len(set(_matched_v))
                    matrix[i, j] = left_matches
                    matrix[j, i] = right_matches
                    row = {
                        "model": model,
                        "representation": representation,
                        "similarity": "absolute cosine" if representation == "pca" else "signed cosine",
                        "sae_direction": args.sae_direction if representation == "sae" else "",
                        "left_split": left_label,
                        "right_split": right_label,
                        "t1": args.t1,
                        "t2": args.t2,
                        "matches": left_matches,
                        "elapsed_seconds": time.time() - start,
                        "left_path": str(path_by_label[left_label]),
                        "right_path": str(path_by_label[right_label]),
                    }
                    rows.append(row)
                    rows.append({
                        **row,
                        "left_split": right_label,
                        "right_split": left_label,
                        "matches": right_matches,
                        "left_path": row["right_path"],
                        "right_path": row["left_path"],
                    })
                    print(
                        f"{model} {representation} {left_label}->{right_label}: {left_matches}; "
                        f"{right_label}->{left_label}: {right_matches}",
                        flush=True,
                    )

            stem = output_stem(model, representation, args.sae_direction)
            matrix_path = args.out_dir / f"{stem}_matrix.csv"
            with matrix_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["split", *labels])
                for i, (label, values) in enumerate(zip(labels, matrix)):
                    row_values = ["" if i == j else int(value) for j, value in enumerate(values)]
                    writer.writerow([label, *row_values])
            plot_heatmap(
                matrix,
                labels,
                model,
                representation,
                args.plot_dir / f"{stem}_heatmap.png",
                sae_direction=args.sae_direction,
            )
            combined_matrices[(model, representation)] = matrix
            feature_counts = {len(value) for value in vectors.values()}
            if len(feature_counts) != 1:
                raise ValueError(f"Feature counts differ across {model} {representation} splits: {sorted(feature_counts)}")
            combined_feature_counts[(model, representation)] = feature_counts.pop()
            combined_labels = labels
        all_rows.extend(rows)

        csv_path = args.out_dir / f"split_{representation}_match_counts.csv"
        fieldnames = ["model", "representation", "similarity", "sae_direction", "left_split", "right_split", "t1", "t2", "matches", "elapsed_seconds", "left_path", "right_path"]
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows({key: row[key] for key in fieldnames} for row in rows)
        print(f"Wrote {csv_path}", flush=True)

    if set(args.representations) == set(REPRESENTATIONS) and combined_labels is not None:
        combined_path = args.plot_dir / "combined_split_match_heatmaps.png"
        plot_combined_heatmaps(combined_matrices, combined_feature_counts, combined_labels, combined_path)
        print(f"Wrote {combined_path}", flush=True)

    if args.representations == ["sae"]:
        direction = "" if args.sae_direction == "decoder" else "_encoder"
        legacy_path = args.out_dir / f"split_16384{direction}_match_counts.csv"
        fieldnames = ["model", "left_split", "right_split", "t1", "t2", "matches", "elapsed_seconds", "left_path", "right_path"]
        with legacy_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows({key: row[key] for key in fieldnames} for row in all_rows)
        print(f"Wrote {legacy_path}", flush=True)

    print("Done", flush=True)


if __name__ == "__main__":
    main()
