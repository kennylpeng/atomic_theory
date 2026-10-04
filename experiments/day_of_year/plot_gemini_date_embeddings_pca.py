#!/usr/bin/env python3
"""Plot PCA directly on Gemini embeddings for the 366 date strings."""

from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import argparse
import calendar
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

ROOT = _paper_path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


RESULTS = ROOT / "experiments/day_of_year/results"
DEFAULT_EMBEDDINGS = RESULTS / "gemini_embeddings.npy"
DEFAULT_OUTPUT = RESULTS / "gemini_date_embeddings_pca"


def plot_2d(
    coordinates: np.ndarray,
    explained_variance: np.ndarray,
    first: int,
    second: int,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    month_ids = np.repeat(np.arange(12), [calendar.monthrange(2000, m)[1] for m in range(1, 13)])
    palette = plt.get_cmap("twilight_shifted")(np.arange(12) / 12)
    for month in range(12):
        points = coordinates[month_ids == month]
        ax.scatter(points[:, first], points[:, second], color=palette[month], s=22,
                   label=calendar.month_abbr[month + 1] + ".")
    ax.set_aspect("equal", adjustable="box")
    ax.set(
        xlabel=f"PC{first + 1} ({explained_variance[first]:.1%})",
        ylabel=f"PC{second + 1} ({explained_variance[second]:.1%})",
        title=(
            "PCA of Gemini date embeddings: "
            f"PC{first + 1} vs PC{second + 1}"
        ),
    )
    ax.legend(title="Month", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_3d(
    coordinates: np.ndarray,
    explained_variance: np.ndarray,
    path: Path,
) -> None:
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    month_ids = np.repeat(np.arange(12), [calendar.monthrange(2000, m)[1] for m in range(1, 13)])
    palette = plt.get_cmap("twilight_shifted")(np.arange(12) / 12)
    for month in range(12):
        points = coordinates[month_ids == month]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=palette[month], s=22,
                   depthshade=False, label=calendar.month_abbr[month + 1] + ".")
    spans = np.ptp(coordinates, axis=0)
    ax.set_box_aspect(np.maximum(spans, 1e-12))
    ax.view_init(elev=24, azim=40)
    ax.set(
        xlabel=f"PC1 ({explained_variance[0]:.1%})",
        ylabel=f"PC2 ({explained_variance[1]:.1%})",
        zlabel=f"PC3 ({explained_variance[2]:.1%})",
        title="PCA of Gemini date embeddings: PC1–PC2–PC3",
    )
    ax.legend(title="Month", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    from experiments.day_of_year.run import dates

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    embeddings = np.load(_paper_location(args.embeddings)).astype(np.float32)
    texts = np.asarray(dates())
    if embeddings.ndim != 2 or len(embeddings) != len(texts):
        raise ValueError(
            f"Expected {len(texts)} embedding rows, found {embeddings.shape}"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain non-finite values")

    pca = PCA(n_components=3).fit(embeddings)
    coordinates = pca.transform(embeddings)
    args.output.mkdir(parents=True, exist_ok=True)

    coordinate_path = args.output / "gemini_date_embeddings_pca3.npz"
    np.savez_compressed(
        _paper_location(coordinate_path),
        texts=texts,
        coordinates=coordinates,
        explained_variance_ratio=pca.explained_variance_ratio_,
        components=pca.components_,
        embedding_mean=pca.mean_,
    )

    plot_paths: dict[str, Path] = {}
    for first, second in ((0, 1), (0, 2), (1, 2)):
        key = f"pc{first + 1}_vs_pc{second + 1}"
        path = args.output / f"{key}.png"
        plot_2d(
            coordinates,
            pca.explained_variance_ratio_,
            first,
            second,
            path,
        )
        plot_paths[key] = path

    cloud_path = args.output / "pc1_pc2_pc3_3d.png"
    plot_3d(coordinates, pca.explained_variance_ratio_, cloud_path)

    summary = {
        "model": "gemini-embedding-2-preview",
        "input": "date embeddings (no SAE features)",
        "embedding_shape": list(embeddings.shape),
        "pca_coordinates_shape": list(coordinates.shape),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "outputs": {
            **{key: str(path) for key, path in plot_paths.items()},
            "pc1_pc2_pc3_3d": str(cloud_path),
            "pca_data": str(coordinate_path),
        },
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

