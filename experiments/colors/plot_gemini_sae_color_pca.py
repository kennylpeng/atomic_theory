#!/usr/bin/env python3
"""Reproduce Gemini 131K color-feature PCA analyses from shared inputs."""

from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

ROOT = _paper_path(__file__).resolve().parents[2]
DATA = ROOT / "experiments/colors/data"
DEFAULT_OUTPUT = ROOT / "experiments/colors/results"
DEFAULT_ROWS = DATA / "color_names_hex.csv"
DEFAULT_SPARSE = DATA / "color_names_hex_gemini_d131072_sparse.npz"
PREVALENCE_005 = (
    DATA / "color_names_hex_gemini_d131072_top100_prevalent_neurons_thr0p05.tsv"
)
PREVALENCE_01 = (
    DATA / "color_names_hex_gemini_d131072_top100_prevalent_neurons_thr0p1.tsv"
)

VARIANTS = {
    "selected_ranks_thr0p05": {
        "prevalence": PREVALENCE_005,
        "ranks": [6, 7, 8, 10, 11, 12, 15, 16],
    },
    "top10_thr0p05": {
        "prevalence": PREVALENCE_005,
        "ranks": list(range(1, 11)),
    },
    "top10_thr0p1": {
        "prevalence": PREVALENCE_01,
        "ranks": list(range(1, 11)),
    },
    "top25_thr0p05": {
        "prevalence": PREVALENCE_005,
        "ranks": list(range(1, 26)),
    },
}


def read_color_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"id", "text", "slug", "name", "r", "g", "b"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Unexpected color-row schema in {path}")
    return rows


def read_ranked_features(
    path: Path,
    selected_ranks: list[int],
) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    by_rank = {int(row["rank"]): row for row in rows}
    missing = [rank for rank in selected_ranks if rank not in by_rank]
    if missing:
        raise ValueError(f"Missing ranks {missing} from {path}")
    return [by_rank[rank] for rank in selected_ranks]


def extract_activation_matrix(
    sparse_path: Path,
    feature_ids: list[int],
    expected_rows: int,
) -> np.ndarray:
    sparse = np.load(_paper_location(sparse_path))
    shape = tuple(int(value) for value in sparse["shape"])
    if shape[0] != expected_rows or shape[1] != 131072:
        raise ValueError(
            f"Expected sparse shape ({expected_rows}, 131072), found {shape}"
        )
    matrix = np.zeros((expected_rows, len(feature_ids)), dtype=np.float32)
    feature_columns = {
        int(feature_id): column
        for column, feature_id in enumerate(feature_ids)
    }
    for row, feature, value in zip(
        sparse["row_indices"],
        sparse["feature_indices"],
        sparse["values"],
    ):
        column = feature_columns.get(int(feature))
        if column is not None:
            matrix[int(row), column] = float(value)
    return matrix


def color_hexes(rows: list[dict[str, str]]) -> list[str]:
    return [
        f"#{int(row['r']):02x}{int(row['g']):02x}{int(row['b']):02x}"
        for row in rows
    ]


def plot_2d(
    coordinates: np.ndarray,
    explained_variance: np.ndarray,
    point_colors: list[str],
    first: int,
    second: int,
    variant: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.3))
    ax.scatter(
        coordinates[:, first],
        coordinates[:, second],
        c=point_colors,
        s=24,
        edgecolors="#202020",
        linewidths=0.25,
        alpha=0.9,
    )
    ax.set(
        xlabel=f"PC{first + 1} ({explained_variance[first]:.1%})",
        ylabel=f"PC{second + 1} ({explained_variance[second]:.1%})",
        title=(
            "Gemini 131K color-feature PCA: "
            f"PC{first + 1} vs PC{second + 1}\n{variant}"
        ),
    )
    ax.grid(alpha=0.12)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_3d(
    coordinates: np.ndarray,
    explained_variance: np.ndarray,
    point_colors: list[str],
    variant: str,
    path: Path,
) -> None:
    fig = plt.figure(figsize=(9.2, 7.8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        coordinates[:, 2],
        c=point_colors,
        s=24,
        edgecolors="#202020",
        linewidths=0.25,
        depthshade=False,
        alpha=0.9,
    )
    spans = np.ptp(coordinates, axis=0)
    ax.set_box_aspect(np.maximum(spans, 1e-12))
    ax.view_init(elev=24, azim=40)
    ax.set(
        xlabel=f"PC1 ({explained_variance[0]:.1%})",
        ylabel=f"PC2 ({explained_variance[1]:.1%})",
        zlabel=f"PC3 ({explained_variance[2]:.1%})",
        title=f"Gemini 131K color-feature PCA\n{variant}",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_export_csv(
    path: Path,
    color_rows: list[dict[str, str]],
    selected_features: list[dict[str, str]],
    activations: np.ndarray,
    coordinates: np.ndarray,
) -> None:
    activation_names = [
        f"rank_{row['rank']}_neuron_{row['neuron_id']}"
        for row in selected_features
    ]
    fieldnames = [
        "row_idx",
        "id",
        "hex",
        "name",
        "slug",
        *activation_names,
        "pc1",
        "pc2",
        "pc3",
    ]
    hexes = color_hexes(color_rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(color_rows):
            record: dict[str, object] = {
                "row_idx": index,
                "id": row["id"],
                "hex": hexes[index],
                "name": row["name"],
                "slug": row["slug"],
                "pc1": float(coordinates[index, 0]),
                "pc2": float(coordinates[index, 1]),
                "pc3": float(coordinates[index, 2]),
            }
            for column, name in enumerate(activation_names):
                record[name] = float(activations[index, column])
            writer.writerow(record)


def run_variant(
    variant: str,
    color_rows: list[dict[str, str]],
    sparse_path: Path,
    output_root: Path,
) -> dict[str, object]:
    spec = VARIANTS[variant]
    ranks = list(spec["ranks"])
    prevalence_path = _paper_path(spec["prevalence"])
    selected = read_ranked_features(prevalence_path, ranks)
    feature_ids = [int(row["neuron_id"]) for row in selected]
    activations = extract_activation_matrix(
        sparse_path,
        feature_ids,
        len(color_rows),
    )

    # PCA centers raw activations; retain their original relative scales.
    pca = PCA(n_components=3, svd_solver="full")
    coordinates = pca.fit_transform(activations)
    explained = pca.explained_variance_ratio_

    output = (
        output_root
        / f"color_names_hex_gemini_d131072_{variant}_pca"
    )
    output.mkdir(parents=True, exist_ok=True)
    matrix_path = output / f"{variant}_unthresholded_activation_matrix.npy"
    coordinate_path = output / f"{variant}_unthresholded_pcs.npy"
    export_path = output / f"{variant}_hex_activations_with_pcs.csv"
    np.save(_paper_location(matrix_path), activations)
    np.save(_paper_location(coordinate_path), coordinates)
    write_export_csv(
        export_path,
        color_rows,
        selected,
        activations,
        coordinates,
    )

    point_colors = color_hexes(color_rows)
    plot_paths: dict[str, str] = {}
    for first, second in ((0, 1), (0, 2), (1, 2)):
        key = f"pc{first + 1}_vs_pc{second + 1}"
        path = output / f"{key}.png"
        plot_2d(
            coordinates,
            explained,
            point_colors,
            first,
            second,
            variant,
            path,
        )
        plot_paths[key] = str(path)
    cloud_path = output / "pc1_pc2_pc3_3d.png"
    plot_3d(coordinates, explained, point_colors, variant, cloud_path)
    plot_paths["pc1_pc2_pc3_3d"] = str(cloud_path)

    metadata = {
        "variant": variant,
        "model": "gemini",
        "dictionary_width": 131072,
        "rows": len(color_rows),
        "matrix_shape": list(activations.shape),
        "neuron_ids": feature_ids,
        "selected_ranks": ranks,
        "standardized_before_pca": False,
        "preprocessing": "Mean-centered raw post-TopK activations; no standardization or normalization",
        "activations_used_for_pca": (
            "unthresholded saved sparse activation values for selected "
            "neurons; missing entries treated as 0"
        ),
        "explained_variance_ratio": explained.tolist(),
        "inputs": {
            "rows": str(DEFAULT_ROWS),
            "sparse_activations": str(sparse_path),
            "prevalence_ranking": str(prevalence_path),
        },
        "outputs": {
            **plot_paths,
            "activation_matrix": str(matrix_path),
            "pca_coordinates": str(coordinate_path),
            "rows_with_activations_and_pcs": str(export_path),
        },
    }
    (output / "run_meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--sparse", type=Path, default=DEFAULT_SPARSE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--variant",
        choices=["all", *VARIANTS],
        default="all",
    )
    args = parser.parse_args()

    color_rows = read_color_rows(args.rows)
    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    summaries = [
        run_variant(
            variant,
            color_rows,
            args.sparse,
            args.output,
        )
        for variant in variants
    ]
    overall = {
        "analysis": "Gemini 131K SAE color-feature PCA",
        "variants": summaries,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(overall, indent=2) + "\n"
    )
    print(json.dumps(overall, indent=2), flush=True)


if __name__ == "__main__":
    main()

