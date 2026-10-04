from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Colormap, LinearSegmentedColormap
import numpy as np


SUMMARY = _paper_path("/resources/hierarchy_data_dir/probe_results/all_summaries.tsv")
PLOT_DIR = _paper_path("full_experiments/plots")
WIDTHS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
GBIF_LEVEL_ORDER = ["kingdom", "phylum", "class", "order", "family", "genus"]
GBIF_CMAP = plt.get_cmap("viridis")
PARENT_COLOR = GBIF_CMAP(GBIF_LEVEL_ORDER.index("phylum") / (len(GBIF_LEVEL_ORDER) - 1))
CHILD_COLOR = GBIF_CMAP(GBIF_LEVEL_ORDER.index("family") / (len(GBIF_LEVEL_ORDER) - 1))
FIGSIZE = (7.2, 7.0)
MAIN_AX_BOUNDS = [0.14, 0.18, 0.64, 0.68]
LINE_AX_BOUNDS = MAIN_AX_BOUNDS
COLORBAR_AX_BOUNDS = [0.83, 0.18, 0.035, 0.68]


def read_summary(path: Path | None = None) -> list[dict[str, str]]:
    path = SUMMARY if path is None else path
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_heatmap_csv(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        row_labels: list[str] = []
        values: list[list[float]] = []
        for row in reader:
            row_labels.append(f"{row[0]} ({row[1]})")
            values.append([float(value) for value in row[2:]])
    return row_labels, header[2:], np.asarray(values, dtype=np.float32)


def clean_heatmap_label(label: str) -> str:
    label = label.split("(", 1)[0].strip()
    return label.split()[0]


def draw_prefix_block_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    *,
    title: str,
    colorbar_label: str,
    cbar_ax: plt.Axes | None = None,
    cmap: str | Colormap = "viridis",
) -> None:
    gap = np.full((1, matrix.shape[1]), np.nan, dtype=np.float32)
    matrix_with_gap = np.vstack([matrix[:1], gap, matrix[1:]])
    y_positions = [0, *range(2, matrix_with_gap.shape[0])]

    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad("#ffffff")

    im = ax.imshow(matrix_with_gap, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto", interpolation="none")
    n_cols = matrix_with_gap.shape[1]
    box_color = "#333333"
    box_lw = 1.0
    ax.plot([-0.5, n_cols - 0.5], [-0.5, -0.5], color=box_color, linewidth=box_lw, clip_on=False)
    ax.plot([-0.5, n_cols - 0.5], [0.5, 0.5], color=box_color, linewidth=box_lw, clip_on=False)
    ax.plot([-0.5, -0.5], [-0.5, 0.5], color=box_color, linewidth=box_lw, clip_on=False)
    ax.plot([n_cols - 0.5, n_cols - 0.5], [-0.5, 0.5], color=box_color, linewidth=box_lw, clip_on=False)
    ax.plot([-0.5, n_cols - 0.5], [1.5, 1.5], color=box_color, linewidth=box_lw, clip_on=False)
    ax.plot([-0.5, n_cols - 0.5], [matrix_with_gap.shape[0] - 0.5, matrix_with_gap.shape[0] - 0.5], color=box_color, linewidth=box_lw, clip_on=False)
    ax.plot([-0.5, -0.5], [1.5, matrix_with_gap.shape[0] - 0.5], color=box_color, linewidth=box_lw, clip_on=False)
    ax.plot([n_cols - 0.5, n_cols - 0.5], [1.5, matrix_with_gap.shape[0] - 0.5], color=box_color, linewidth=box_lw, clip_on=False)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels([clean_heatmap_label(label) for label in col_labels], fontsize=17, rotation=35, ha="right")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([clean_heatmap_label(label) for label in row_labels], fontsize=14)
    ax.set_xlabel("SAE width", fontsize=20)
    ax.set_ylabel("Prefix", fontsize=20)
    ax.set_title(title, fontsize=22)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = ax.figure.colorbar(im, cax=cbar_ax) if cbar_ax is not None else ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(colorbar_label, fontsize=18)
    cbar.ax.tick_params(labelsize=17)


def validate_prefix(prefix: str) -> str:
    prefix = prefix.lower()
    if len(prefix) != 1 or not prefix.isascii() or not prefix.isalpha():
        raise ValueError(f"Prefix must be one ASCII letter, got {prefix!r}")
    return prefix


def plot_f1_by_width(model: str, plot_dir: Path = PLOT_DIR, *, prefix: str = "a") -> None:
    prefix = validate_prefix(prefix)
    rows = [
        row
        for row in read_summary()
        if row["dataset"] == "wordfreq"
        and row["model"] == model
        and (
            row["category"] == prefix
            or (len(row["category"]) == 2 and row["category"].startswith(prefix))
        )
    ]
    if not rows:
        raise ValueError(f"No wordfreq {prefix}-prefix rows found for {model}")

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes(LINE_AX_BOUNDS)
    by_category: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)

    for category, category_rows in sorted(by_category.items(), key=lambda item: (item[0] != prefix, item[0])):
        category_rows.sort(key=lambda row: int(row["width"]))
        if len(category_rows) < 2:
            continue
        color = PARENT_COLOR if category == prefix else CHILD_COLOR
        alpha = 0.45 if category == prefix else 0.14
        linewidth = 1.3 if category == prefix else 0.8
        xs = [int(row["width"]) for row in category_rows]
        ys = [float(row["test_f1"]) for row in category_rows]
        ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=alpha, zorder=1)

    for row in rows:
        category = row["category"]
        color = PARENT_COLOR if category == prefix else CHILD_COLOR
        size = 28 if category == prefix else 18
        alpha = 0.78 if category == prefix else 0.55
        ax.scatter(
            int(row["width"]),
            float(row["test_f1"]),
            s=size,
            alpha=alpha,
            color=color,
            edgecolors="none",
            zorder=2,
        )

    parent_means = []
    child_means = []
    parent_rows = by_category.get(prefix, [])
    child_rows = [row for category, category_rows in by_category.items() if category != prefix for row in category_rows]
    for width in WIDTHS:
        vals = [float(row["test_f1"]) for row in parent_rows if int(row["width"]) == width]
        if vals:
            parent_means.append((width, float(np.mean(vals))))
        vals = [float(row["test_f1"]) for row in child_rows if int(row["width"]) == width]
        if vals:
            child_means.append((width, float(np.mean(vals))))
    if parent_means:
        xs, ys = zip(*parent_means)
        ax.plot(xs, ys, marker="o", markersize=6, linewidth=2.8, color=PARENT_COLOR, label=f"{prefix} mean", zorder=3)
    if child_means:
        xs, ys = zip(*child_means)
        ax.plot(xs, ys, marker="o", markersize=6, linewidth=2.8, color=CHILD_COLOR, label=f"{prefix}a-{prefix}z mean", zorder=3)

    child_proxy = plt.Line2D([0], [0], color=CHILD_COLOR, marker="o", linestyle="", markersize=5, alpha=0.62)
    parent_proxy = plt.Line2D([0], [0], color=PARENT_COLOR, marker="o", linestyle="", markersize=6, alpha=0.78)
    parent_mean_proxy = plt.Line2D([0], [0], color=PARENT_COLOR, marker="o", linewidth=2.8, markersize=6)
    child_mean_proxy = plt.Line2D([0], [0], color=CHILD_COLOR, marker="o", linewidth=2.8, markersize=6)
    ax.legend(
        [parent_mean_proxy, child_mean_proxy],
        [f"{prefix} mean", f"{prefix}a-{prefix}z mean"],
        loc="lower right",
        fontsize=18,
    )

    ax.set_xscale("log", base=2)
    x_pad = 2.0 ** 0.25
    ax.set_xlim(min(WIDTHS) / x_pad, max(WIDTHS) * x_pad)
    ax.set_xticks(WIDTHS)
    ax.set_xticklabels([str(w) for w in WIDTHS], rotation=35, ha="right", fontsize=17)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("SAE width", fontsize=20)
    ax.set_ylabel("Test F1", fontsize=20)
    ax.set_title(f"{prefix}-prefix ({model})", fontsize=22)
    ax.tick_params(axis="y", labelsize=17)
    ax.grid(True, axis="y", alpha=0.25)

    plot_dir.mkdir(parents=True, exist_ok=True)
    stem = f"wordfreq_{prefix}_prefix_{model}_test_f1_by_width"
    fig.savefig(plot_dir / f"{stem}.png", dpi=180)
    fig.savefig(plot_dir / f"{stem}.pdf")
    plt.close(fig)


def plot_probe_prefix_recall_heatmap(model: str, plot_dir: Path = PLOT_DIR) -> None:
    csv_path = plot_dir / f"wordfreq_a_probe_prefix_recall_heatmap_{model}.csv"
    row_labels, col_labels, matrix = read_heatmap_csv(csv_path)

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes(MAIN_AX_BOUNDS)
    cbar_ax = fig.add_axes(COLORBAR_AX_BOUNDS)
    draw_prefix_block_heatmap(
        ax,
        matrix,
        row_labels,
        col_labels,
        title=f"a-prefix probe recall ({model})",
        colorbar_label="Recall of best a-prefix SAE feature",
        cbar_ax=cbar_ax,
        # Match Table 1's xcolor orange!p: white at zero, RGB (1, .5, 0) at one.
        cmap=LinearSegmentedColormap.from_list(
            "table_recall", [(1.0, 1.0, 1.0), (1.0, 0.5, 0.0)]
        ),
    )

    stem = f"wordfreq_a_probe_prefix_recall_heatmap_{model}"
    fig.savefig(plot_dir / f"{stem}.png", dpi=180)
    fig.savefig(plot_dir / f"{stem}.pdf")
    plt.close(fig)


def wordfreq_a_prefix_rows(model: str) -> list[dict[str, str]]:
    return [
        row
        for row in read_summary()
        if row["dataset"] == "wordfreq"
        and row["model"] == model
        and (row["category"] == "a" or (len(row["category"]) == 2 and row["category"].startswith("a")))
    ]


def build_prefix_f1_heatmap(model: str) -> tuple[list[str], list[str], np.ndarray, list[int]]:
    rows = wordfreq_a_prefix_rows(model)
    if not rows:
        raise ValueError(f"No wordfreq a-prefix rows found for {model}")

    categories = ["a"] + sorted({row["category"] for row in rows if row["category"] != "a"})
    by_category_width = {(row["category"], int(row["width"])): row for row in rows}
    matrix = np.full((len(categories), len(WIDTHS)), np.nan, dtype=np.float32)
    positive_counts = []
    for i, category in enumerate(categories):
        count = None
        for j, width in enumerate(WIDTHS):
            row = by_category_width.get((category, width))
            if row is None:
                continue
            matrix[i, j] = float(row["test_f1"])
            count = int(row["positive_count"])
        positive_counts.append(0 if count is None else count)
    return categories, [str(width) for width in WIDTHS], matrix, positive_counts


def write_prefix_f1_heatmap_csv(path: Path, row_labels: list[str], col_labels: list[str], matrix: np.ndarray, positive_counts: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prefix", "positive_count", *col_labels])
        for label, count, values in zip(row_labels, positive_counts, matrix):
            writer.writerow([label, count, *[f"{float(value):.8g}" if np.isfinite(value) else "" for value in values]])


def plot_prefix_f1_heatmap(model: str, plot_dir: Path = PLOT_DIR) -> None:
    row_labels, col_labels, matrix, positive_counts = build_prefix_f1_heatmap(model)
    stem = f"wordfreq_a_prefix_best_probe_test_f1_heatmap_{model}"
    write_prefix_f1_heatmap_csv(plot_dir / f"{stem}.csv", row_labels, col_labels, matrix, positive_counts)

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes(MAIN_AX_BOUNDS)
    cbar_ax = fig.add_axes(COLORBAR_AX_BOUNDS)
    draw_prefix_block_heatmap(
        ax,
        matrix,
        row_labels,
        col_labels,
        title=f"a-prefix best probe test F1 ({model})",
        colorbar_label="test F1 of best prefix SAE feature",
        cbar_ax=cbar_ax,
    )
    fig.savefig(plot_dir / f"{stem}.png", dpi=180)
    fig.savefig(plot_dir / f"{stem}.pdf")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate wordfreq a-prefix figures.")
    parser.add_argument("--model", choices=["gemini", "nemotron", "both"], default="gemini")
    parser.add_argument(
        "--prefix",
        default="a",
        help="One-letter prefix for the F1-by-width plot (default: a).",
    )
    parser.add_argument("--figure", choices=["f1", "recall-heatmap", "prefix-f1-heatmap", "both", "all"], default="both")
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--plot-dir", type=Path, default=PLOT_DIR)
    return parser.parse_args()


def main() -> None:
    global SUMMARY
    args = parse_args()
    SUMMARY = args.summary
    models = ["gemini", "nemotron"] if args.model == "both" else [args.model]
    for model in models:
        if args.figure in {"f1", "both", "all"}:
            plot_f1_by_width(model, args.plot_dir, prefix=args.prefix)
        if args.figure in {"recall-heatmap", "both", "all"}:
            plot_probe_prefix_recall_heatmap(model, args.plot_dir)
        if args.figure in {"prefix-f1-heatmap", "all"}:
            plot_prefix_f1_heatmap(model, args.plot_dir)


if __name__ == "__main__":
    main()
