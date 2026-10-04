#!/usr/bin/env python3
"""Visualize colors that activate a selected Gemini 131K feature set."""

from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import argparse
import colorsys
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import numpy as np

ROOT = _paper_path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.colors.plot_gemini_sae_color_pca import (
    DEFAULT_ROWS,
    DEFAULT_SPARSE,
    VARIANTS,
    color_hexes,
    extract_activation_matrix,
    read_color_rows,
    read_ranked_features,
)

RESULTS = ROOT / "experiments/colors/results"
DEFAULT_VARIANT = "selected_ranks_thr0p05"


def unique_hex_indices(color_rows: list[dict[str, str]]) -> list[int]:
    indices: list[int] = []
    seen: set[str] = set()
    for index, hex_value in enumerate(color_hexes(color_rows)):
        key = hex_value.lower()
        if key not in seen:
            seen.add(key)
            indices.append(index)
    return indices


def top_unique_colors(
    color_rows: list[dict[str, str]],
    activations: np.ndarray,
    top: int,
) -> list[list[int]]:
    unique_indices = np.asarray(unique_hex_indices(color_rows))
    ranked: list[list[int]] = []
    for column in range(activations.shape[1]):
        order = np.argsort(activations[unique_indices, column])[::-1]
        ranked.append(unique_indices[order[:top]].tolist())
    return ranked


def plot_swatches(
    color_rows: list[dict[str, str]],
    selected_features: list[dict[str, str]],
    activations: np.ndarray,
    ranked_indices: list[list[int]],
    path: Path,
    model: str = "Gemini",
    threshold: float = 0.05,
) -> None:
    n_features = len(selected_features)
    top = len(ranked_indices[0])
    hexes = color_hexes(color_rows)
    fig, ax = plt.subplots(
        figsize=(1.12 * top, 0.95 * n_features + 1.8)
    )
    ax.set_xlim(-1.25, top)
    ax.set_ylim(0, n_features)
    ax.axis("off")
    for row_index, (feature, color_indices) in enumerate(
        zip(selected_features, ranked_indices)
    ):
        y = n_features - row_index - 1
        ax.text(
            -0.15,
            y + 0.58,
            f"rank {feature['rank']}\nfeature {feature['neuron_id']}",
            ha="right",
            va="center",
            fontsize=9,
        )
        for color_rank, color_index in enumerate(color_indices, start=1):
            x = color_rank - 1
            ax.add_patch(
                Rectangle(
                    (x + 0.04, y + 0.34),
                    0.92,
                    0.52,
                    facecolor=hexes[color_index],
                    edgecolor="#343a40",
                    linewidth=0.6,
                )
            )
            name = color_rows[color_index]["name"]
            if len(name) > 17:
                name = name[:15] + "…"
            activation = activations[color_index, row_index]
            ax.text(
                x + 0.5,
                y + 0.31,
                f"{color_rank}. {name}\n{hexes[color_index]} · {activation:.3f}",
                ha="center",
                va="top",
                fontsize=5.4,
            )
    fig.suptitle(
        (
            f"Top distinct colors for {n_features} selected {model} 131K features\n"
            f"Raw post-TopK activations; features selected from ≥{threshold:g} "
            "prevalence ranking"
        ),
        fontsize=15,
    )
    fig.tight_layout(rect=(0.015, 0.01, 0.995, 0.93))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def hue_sorted_indices(
    color_rows: list[dict[str, str]],
) -> list[int]:
    indices = unique_hex_indices(color_rows)

    def key(index: int) -> tuple[float, float, float]:
        row = color_rows[index]
        red = int(row["r"]) / 255.0
        green = int(row["g"]) / 255.0
        blue = int(row["b"]) / 255.0
        hue, saturation, value = colorsys.rgb_to_hsv(red, green, blue)
        # Put near-grays after chromatic colors, ordered dark to light.
        if saturation < 0.08:
            return (1.1, 0.0, value)
        return (hue, -saturation, value)

    return sorted(indices, key=key)



def chromatic_indices(
    color_rows: list[dict[str, str]],
    indices: list[int],
) -> list[int]:
    """Exclude near-neutrals (HSV S < .60), darks (V < .50), and whites (HLS L > .70)."""
    kept = []
    for index in indices:
        rgb = tuple(int(color_rows[index][channel]) / 255.0 for channel in ("r", "g", "b"))
        _, saturation, value = colorsys.rgb_to_hsv(*rgb)
        _, lightness, _ = colorsys.rgb_to_hls(*rgb)
        if saturation >= 0.60 and value >= 0.50 and lightness <= 0.70:
            kept.append(index)
    return kept


def plot_hue_heatmap(
    color_rows: list[dict[str, str]],
    selected_features: list[dict[str, str]],
    activations: np.ndarray,
    ordered_indices: list[int],
    path: Path,
    model: str = "Gemini",
    threshold: float = 0.05,
) -> None:
    ordered = np.asarray(chromatic_indices(color_rows, ordered_indices), dtype=int)
    if not ordered.size:
        raise ValueError("No colors remain after filtering near-neutral colors")
    matrix = activations[ordered].T
    # Wrap horizontal positions around a circle so both edges are adjacent.
    angles = 2 * np.pi * np.arange(matrix.shape[1]) / matrix.shape[1]
    resultant = matrix @ np.exp(1j * angles)
    centers = np.mod(np.angle(resultant), 2 * np.pi)
    # Inactive or circularly balanced rows have no defined mean; put them last.
    totals = matrix.sum(axis=1, dtype=np.float64)
    centers[np.abs(resultant) <= 1e-12 * totals] = np.inf
    row_order = np.argsort(centers, kind="stable")
    matrix = matrix[row_order]
    ordered_hexes = [color_hexes(color_rows)[index] for index in ordered]
    labels = [
        f"rank {selected_features[index]['rank']} · "
        f"f{selected_features[index]['neuron_id']}"
        for index in row_order
    ]

    fig = plt.figure(figsize=(18, max(8.5, 0.42 * len(selected_features) + 3)))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.25, 4.0],
        width_ratios=[1.0, 0.035],
        hspace=0.08,
        wspace=0.06,
    )
    swatch_ax = fig.add_subplot(grid[0, 0])
    heat_ax = fig.add_subplot(grid[1, 0], sharex=swatch_ax)
    colorbar_ax = fig.add_subplot(grid[1, 1])
    swatch_ax.imshow(
        np.arange(len(ordered))[None, :],
        aspect="auto",
        cmap=ListedColormap(ordered_hexes),
        interpolation="nearest",
    )
    swatch_ax.set_axis_off()

    heatmap = heat_ax.imshow(
        matrix,
        aspect="auto",
        cmap="magma",
        interpolation="nearest",
        vmin=0.0,
        vmax=float(matrix.max()),
    )
    heat_ax.set(
        xlabel="Distinct colors sorted by hue (near-white/gray/black excluded)",
        ylabel="Selected feature",
        yticks=np.arange(len(labels)),
        yticklabels=labels,
        title="Raw activation across the hue-sorted color set",
    )
    heat_ax.set_xticks([])
    fig.colorbar(heatmap, cax=colorbar_ax, label="Raw post-TopK activation")
    fig.suptitle(
        f"{len(selected_features)} selected {model} 131K color features",
        fontsize=15,
    )
    fig.subplots_adjust(left=0.14, right=0.92, bottom=0.09, top=0.91)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_top_colors(
    path: Path,
    color_rows: list[dict[str, str]],
    selected_features: list[dict[str, str]],
    activations: np.ndarray,
    ranked_indices: list[list[int]],
    activation_threshold: float,
) -> None:
    hexes = color_hexes(color_rows)
    fieldnames = [
        "feature_rank",
        "feature_id",
        "selection_prevalence",
        "color_rank",
        "hex",
        "name",
        "slug",
        "raw_activation",
        "activation_ge_0p05",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for column, (feature, indices) in enumerate(
            zip(selected_features, ranked_indices)
        ):
            for color_rank, index in enumerate(indices, start=1):
                activation = float(activations[index, column])
                writer.writerow(
                    {
                        "feature_rank": feature["rank"],
                        "feature_id": feature["neuron_id"],
                        "selection_prevalence": next(
                            value
                            for key, value in feature.items()
                            if key.startswith("prevalence_ge_")
                        ),
                        "color_rank": color_rank,
                        "hex": hexes[index],
                        "name": color_rows[index]["name"],
                        "slug": color_rows[index]["slug"],
                        "raw_activation": activation,
                        "activation_ge_0p05": (
                            activation >= activation_threshold
                        ),
                    }
                )


def write_hue_sorted_data(
    path: Path,
    color_rows: list[dict[str, str]],
    selected_features: list[dict[str, str]],
    activations: np.ndarray,
    ordered_indices: list[int],
) -> None:
    hexes = color_hexes(color_rows)
    activation_names = [
        f"rank_{feature['rank']}_feature_{feature['neuron_id']}"
        for feature in selected_features
    ]
    fieldnames = [
        "hue_order",
        "hex",
        "name",
        "slug",
        *activation_names,
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for order, index in enumerate(ordered_indices):
            record: dict[str, object] = {
                "hue_order": order,
                "hex": hexes[index],
                "name": color_rows[index]["name"],
                "slug": color_rows[index]["slug"],
            }
            for column, name in enumerate(activation_names):
                record[name] = float(activations[index, column])
            writer.writerow(record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--sparse", type=Path, default=DEFAULT_SPARSE)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--variant",
        choices=list(VARIANTS),
        default=DEFAULT_VARIANT,
    )
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be positive")

    color_rows = read_color_rows(args.rows)
    spec = VARIANTS[args.variant]
    activation_threshold = 0.1 if "thr0p1" in args.variant else 0.05
    selected_features = read_ranked_features(
        _paper_path(spec["prevalence"]),
        list(spec["ranks"]),
    )
    feature_ids = [
        int(feature["neuron_id"]) for feature in selected_features
    ]
    activations = extract_activation_matrix(
        args.sparse,
        feature_ids,
        len(color_rows),
    )

    output = args.output or (
        RESULTS / f"color_names_hex_gemini_d131072_{args.variant}_swatches"
    )
    output.mkdir(parents=True, exist_ok=True)
    ranked_indices = top_unique_colors(
        color_rows,
        activations,
        args.top,
    )
    ordered_indices = hue_sorted_indices(color_rows)

    swatch_path = output / f"{args.variant}_top_color_swatches.png"
    heatmap_path = output / f"{args.variant}_hue_activation_heatmap.png"
    top_data_path = output / f"{args.variant}_top_colors.tsv"
    hue_data_path = output / f"{args.variant}_hue_activations.csv"
    plot_swatches(
        color_rows,
        selected_features,
        activations,
        ranked_indices,
        swatch_path,
    )
    plot_hue_heatmap(
        color_rows,
        selected_features,
        activations,
        ordered_indices,
        heatmap_path,
    )
    write_top_colors(
        top_data_path,
        color_rows,
        selected_features,
        activations,
        ranked_indices,
        activation_threshold,
    )
    write_hue_sorted_data(
        hue_data_path,
        color_rows,
        selected_features,
        activations,
        ordered_indices,
    )

    summary = {
        "model": "gemini",
        "variant": args.variant,
        "dictionary_width": 131072,
        "features": feature_ids,
        "feature_ranks": [
            int(feature["rank"]) for feature in selected_features
        ],
        "selection_threshold": activation_threshold,
        "activation_scale": "raw post-TopK",
        "top_distinct_colors_per_feature": args.top,
        "distinct_hex_colors": len(ordered_indices),
        "outputs": {
            "top_color_swatches": str(swatch_path),
            "hue_sorted_activation_heatmap": str(heatmap_path),
            "top_colors_data": str(top_data_path),
            "hue_sorted_activation_data": str(hue_data_path),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

