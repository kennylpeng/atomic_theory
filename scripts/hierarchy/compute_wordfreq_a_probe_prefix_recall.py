#!/usr/bin/env python3
"""Compute a-prefix probe-recall matrices from hierarchy probe outputs."""
from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import argparse
import csv
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(_paper_path(__file__).resolve().parents[2]))
from project_paths import get_path

TOPK_BY_WIDTH = {512: 32, 1024: 32, 2048: 32, 4096: 32, 8192: 64, 16384: 64, 32768: 64, 65536: 128, 131072: 128}


def read_table(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def prefix_summary(path: Path) -> list[dict[str, str]]:
    rows = read_table(path, "\t")
    rows = [row for row in rows if row["category"] == "a" or (len(row["category"]) == 2 and row["category"].startswith("a"))]
    if not rows:
        raise ValueError(f"No a-prefix probe rows found in {path}")
    return rows


def a_probe_predictions(sparse_path: Path, n_rows: int, feature: int, threshold: float) -> np.ndarray:
    sparse = np.load(_paper_location(sparse_path))
    predicted = np.zeros(n_rows, dtype=bool)
    keep = (sparse["feature_indices"] == feature) & (sparse["values"] >= threshold)
    if np.any(keep):
        predicted[sparse["row_indices"][keep].astype(np.int64, copy=False)] = True
    return predicted


def compute_matrix(hierarchy_dir: Path, results_dir: Path, model: str, widths: list[int]):
    wordfreq_dir = hierarchy_dir / "wordfreq"
    source_rows = read_table(wordfreq_dir / "rows.csv")
    words = [row["text"] for row in source_rows]
    summaries = {}
    for width in widths:
        top_k = TOPK_BY_WIDTH[width]
        path = results_dir / "wordfreq" / f"{model}_m{width}_k{top_k}" / "summary.tsv"
        summaries[width] = prefix_summary(path)

    category_sets = [{row["category"] for row in summaries[width]} for width in widths]
    categories = ["a"] + sorted(set.intersection(*category_sets) - {"a"})
    targets = {category: np.fromiter((word.startswith(category) for word in words), dtype=bool) for category in categories}
    counts = [int(targets[category].sum()) for category in categories]
    matrix = np.empty((len(categories), len(widths)), dtype=np.float32)
    columns = []

    for column, width in enumerate(widths):
        parent = [row for row in summaries[width] if row["category"] == "a"]
        if len(parent) != 1:
            raise ValueError(f"Expected one a probe for {model} width {width}, found {len(parent)}")
        feature = int(parent[0]["best_feature"])
        threshold = float(parent[0]["threshold"])
        columns.append(f"{width} f{feature}")
        predicted = a_probe_predictions(
            wordfreq_dir / f"sparse_{model}_m{width}_k{TOPK_BY_WIDTH[width]}.npz",
            len(source_rows), feature, threshold,
        )
        for row_index, category in enumerate(categories):
            target = targets[category]
            matrix[row_index, column] = np.count_nonzero(predicted & target) / np.count_nonzero(target)
    return categories, counts, columns, matrix


def write_matrix(path: Path, categories, counts, columns, matrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["prefix", "positive_count", *columns])
        for category, count, values in zip(categories, counts, matrix):
            writer.writerow([category, count, *[f"{float(value):.8g}" for value in values]])


def parse_args() -> argparse.Namespace:
    hierarchy_dir = _paper_path(get_path("hierarchy_data_dir"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hierarchy-dir", type=Path, default=hierarchy_dir)
    parser.add_argument("--probe-results-dir", type=Path, default=hierarchy_dir / "probe_results")
    parser.add_argument("--out-dir", type=Path, default=_paper_path("full_experiments/plots"))
    parser.add_argument("--model", choices=("gemini", "nemotron", "both"), default="both")
    parser.add_argument("--widths", type=int, nargs="+", default=list(TOPK_BY_WIDTH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unknown = [width for width in args.widths if width not in TOPK_BY_WIDTH]
    if unknown:
        raise ValueError(f"No Top-K mapping for widths: {unknown}")
    models = ("gemini", "nemotron") if args.model == "both" else (args.model,)
    for model in models:
        result = compute_matrix(args.hierarchy_dir, args.probe_results_dir, model, args.widths)
        output = args.out_dir / f"wordfreq_a_probe_prefix_recall_heatmap_{model}.csv"
        write_matrix(output, *result)
        print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
