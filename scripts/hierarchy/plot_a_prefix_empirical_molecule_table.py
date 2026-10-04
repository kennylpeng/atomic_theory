#!/usr/bin/env python3
"""Plot empirical a-prefix decoder molecules and held-out recall.

For each requested SAE width, the best probe features for ``a``, ``ah``,
``ab``, ``ap``, and ``as`` are de-duplicated by feature ID.  Each row is thus
one actual learned feature, even when the same feature is selected for more
than one prefix.  Molecules show the requested atoms' positive coefficients
from the full regression onto 131K a-prefix decoder directions.  Recall cells
use each feature's train-selected threshold and held-out prefix texts.
"""
from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Rectangle

sys.path.insert(0, str(_paper_path(__file__).resolve().parents[2]))
from project_paths import get_path
from scripts.plot_style import apply_plot_style
from scripts.hierarchy.circle_packing import pack_circles, validate_packing


WIDTH_TOPK = [(512, 32), (2048, 32), (131072, 128)]
PROBE_CATEGORIES = ["a", "ah", "ab", "ap", "as"]
ACTIVATION_CATEGORIES = ["ah", "ab", "ap", "as"]
# Match the five atom colors used in the feature-3290 table.
ATOM_COLORS = {
    "a": "#65C2DB",
    "ah": "#B18AD8",
    "ab": "#86C875",
    "ap": "#F3AD55",
    "as": "#E889A0",
}


@dataclass
class FeatureRow:
    width: int
    top_k: int
    feature: int
    selected_for: list[str]
    threshold_source: str
    threshold: float
    coefficients: dict[str, float]
    recalls: dict[str, float]


def read_csv(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def selected_features(
    probe_results_dir: Path,
    model: str,
) -> tuple[dict[int, list[tuple[int, list[str]]]], dict[tuple[int, int], tuple[str, float]]]:
    result: dict[int, list[tuple[int, list[str]]]] = {}
    thresholds: dict[tuple[int, int], tuple[str, float]] = {}
    for width, top_k in WIDTH_TOPK:
        summary_path = probe_results_dir / "wordfreq" / f"{model}_m{width}_k{top_k}" / "summary.tsv"
        by_category = {row["category"]: row for row in read_csv(summary_path, "\t")}
        missing = [category for category in PROBE_CATEGORIES if category not in by_category]
        if missing:
            raise ValueError(f"{summary_path} is missing categories: {missing}")

        ordered_features: list[int] = []
        labels: dict[int, list[str]] = defaultdict(list)
        for category in PROBE_CATEGORIES:
            feature = int(by_category[category]["best_feature"])
            if feature not in labels:
                ordered_features.append(feature)
                thresholds[(width, feature)] = (category, float(by_category[category]["threshold"]))
            labels[feature].append(category)
        result[width] = [(feature, labels[feature]) for feature in ordered_features]
    return result, thresholds


def requested_coefficients(
    path: Path,
    selected: dict[int, list[tuple[int, list[str]]]],
) -> dict[tuple[int, int], dict[str, float]]:
    selected_ids = {width: {feature for feature, _ in rows} for width, rows in selected.items()}
    values: dict[tuple[int, int, str], list[float]] = defaultdict(list)
    for row in read_csv(path):
        width = int(row["target_width"])
        feature = int(row["target_feature"])
        atom = row["basis_category"]
        if (
            width in selected_ids
            and feature in selected_ids[width]
            and row["target_direction"] == "decoder"
            and atom in PROBE_CATEGORIES
        ):
            values[(width, feature, atom)].append(float(row["coefficient"]))

    result: dict[tuple[int, int], dict[str, float]] = {}
    for width, feature_rows in selected.items():
        for feature, _ in feature_rows:
            coefficients = {}
            for atom in PROBE_CATEGORIES:
                candidates = values.get((width, feature, atom), [])
                if not candidates:
                    raise ValueError(f"Missing coefficient for width={width}, feature={feature}, atom={atom}")
                if not np.allclose(candidates, candidates[0], atol=1e-10, rtol=1e-10):
                    raise ValueError(
                        f"Inconsistent duplicate coefficients for width={width}, feature={feature}, atom={atom}: "
                        f"{candidates}"
                    )
                coefficients[atom] = candidates[0]
            result[(width, feature)] = coefficients
    return result


def target_masks(rows: list[dict[str, str]], split: str) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    split_mask = np.ones(len(rows), dtype=bool) if split == "all" else np.asarray(
        [row.get("split") == split for row in rows], dtype=bool
    )
    if not np.any(split_mask):
        raise ValueError(f"No rows found for split={split!r}")
    masks = {
        category: split_mask & np.fromiter(
            (row["text"].startswith(category) for row in rows),
            dtype=bool,
            count=len(rows),
        )
        for category in PROBE_CATEGORIES
    }
    counts = {category: int(np.count_nonzero(mask)) for category, mask in masks.items()}
    if any(count == 0 for count in counts.values()):
        raise ValueError(f"Empty target category in split={split}: {counts}")
    return masks, counts


def feature_recalls(
    hierarchy_dir: Path,
    model: str,
    selected: dict[int, list[tuple[int, list[str]]]],
    thresholds: dict[tuple[int, int], tuple[str, float]],
    masks: dict[str, np.ndarray],
) -> dict[tuple[int, int], dict[str, float]]:
    result: dict[tuple[int, int], dict[str, float]] = {}
    n_rows = len(next(iter(masks.values())))
    for width, top_k in WIDTH_TOPK:
        sparse_path = hierarchy_dir / "wordfreq" / f"sparse_{model}_m{width}_k{top_k}.npz"
        with np.load(_paper_location(sparse_path)) as sparse:
            row_indices = sparse["row_indices"].astype(np.int64, copy=False)
            feature_indices = sparse["feature_indices"].astype(np.int64, copy=False)
            activation_values = sparse["values"].astype(np.float32, copy=False)
            shape = tuple(int(value) for value in sparse["shape"])
        if shape[0] != n_rows:
            raise ValueError(f"{sparse_path} has {shape[0]} rows, expected {n_rows}")

        for feature, _ in selected[width]:
            _, threshold = thresholds[(width, feature)]
            predicted = np.zeros(n_rows, dtype=bool)
            keep = (feature_indices == feature) & (activation_values >= threshold)
            predicted[row_indices[keep]] = True
            result[(width, feature)] = {
                category: float(np.count_nonzero(predicted & mask) / np.count_nonzero(mask))
                for category, mask in masks.items()
            }
    return result


def assemble_rows(
    selected: dict[int, list[tuple[int, list[str]]]],
    thresholds: dict[tuple[int, int], tuple[str, float]],
    coefficients: dict[tuple[int, int], dict[str, float]],
    recalls: dict[tuple[int, int], dict[str, float]],
) -> list[FeatureRow]:
    return [
        FeatureRow(
            width=width,
            top_k=top_k,
            feature=feature,
            selected_for=categories,
            threshold_source=thresholds[(width, feature)][0],
            threshold=thresholds[(width, feature)][1],
            coefficients=coefficients[(width, feature)],
            recalls=recalls[(width, feature)],
        )
        for width, top_k in WIDTH_TOPK
        for feature, categories in selected[width]
    ]


def intended_recall(row: FeatureRow) -> float:
    return max(row.recalls[category] for category in row.selected_for)


def filter_by_intended_recall(rows: list[FeatureRow], minimum: float) -> list[FeatureRow]:
    filtered = [row for row in rows if intended_recall(row) >= minimum]
    missing_widths = [
        width for width, _top_k in WIDTH_TOPK
        if not any(row.width == width for row in filtered)
    ]
    if missing_widths:
        raise ValueError(
            f"Intended-recall threshold {minimum} removes every row at widths {missing_widths}"
        )
    return filtered


def write_values_csv(
    path: Path,
    rows: list[FeatureRow],
    target_counts: dict[str, int],
    split: str,
    minimum_intended_recall: float,
) -> None:
    fields = [
        "width", "top_k", "feature_id", "selected_for", "threshold_source",
        "threshold", "evaluation_split", "minimum_intended_recall", "intended_recall",
    ]
    fields += [f"coefficient_{atom}" for atom in PROBE_CATEGORIES]
    fields += [f"recall_{category}" for category in ACTIVATION_CATEGORIES]
    fields += [f"n_{category}" for category in ACTIVATION_CATEGORIES]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record: dict[str, object] = {
                "width": row.width,
                "top_k": row.top_k,
                "feature_id": row.feature,
                "selected_for": ";".join(row.selected_for),
                "threshold_source": row.threshold_source,
                "threshold": row.threshold,
                "evaluation_split": split,
                "minimum_intended_recall": minimum_intended_recall,
                "intended_recall": intended_recall(row),
            }
            record.update({f"coefficient_{atom}": row.coefficients[atom] for atom in PROBE_CATEGORIES})
            record.update(
                {f"recall_{category}": row.recalls[category] for category in ACTIVATION_CATEGORIES}
            )
            record.update({f"n_{category}": target_counts[category] for category in ACTIVATION_CATEGORIES})
            writer.writerow(record)


def latex_molecule(row: FeatureRow, coefficient_tolerance: float, radius_scale: float = 0.38) -> str:
    entries = [
        (atom, coefficient)
        for atom, coefficient in row.coefficients.items()
        if coefficient >= coefficient_tolerance
    ]
    packed = pack_circles(entries, radius_power=1.0 / 3.0)
    validate_packing(packed)
    lines = [r"\begin{tikzpicture}[baseline=(current bounding box.center)]"]
    for atom, _coefficient, radius, center in packed:
        x, y = center * radius_scale
        display_radius = radius * radius_scale
        diameter = 2.0 * display_radius
        if diameter >= 0.60:
            font = r"\scriptsize"
            label_scale = 1.0
        elif diameter >= 0.34:
            font = r"\tiny"
            label_scale = 0.82
        else:
            font = r"\tiny"
            label_scale = 0.55
        atom_label = f"$\\mathrm{{{atom}}}$"
        lines.extend(
            [
                f"  \\filldraw[draw,fill=aprefix{atom}color]",
                f"    ({x:.4f},{y:.4f}) circle[radius={display_radius:.4f}cm];",
                f"  \\node[inner sep=0pt,font={font},scale={label_scale:.2f}]",
                f"    at ({x:.4f},{y:.4f}) {{{atom_label}}};",
            ]
        )
    lines.append(r"\end{tikzpicture}")
    return "\n".join(lines)


def latex_decomposition(row: FeatureRow, coefficient_tolerance: float) -> str:
    entries = sorted(
        (
            (atom, coefficient)
            for atom, coefficient in row.coefficients.items()
            if coefficient >= coefficient_tolerance
        ),
        key=lambda item: (-item[1], item[0]),
    )
    terms = [
        f"{coefficient:.2f}\\,a^*_{{\\mathrm{{{atom}}}}}"
        for atom, coefficient in entries
    ]
    expression_lines = [" + ".join(terms[index:index + 2]) for index in range(0, len(terms), 2)]
    expression_lines = [
        line if index == 0 else "+ " + line
        for index, line in enumerate(expression_lines)
    ]
    return r"\shortstack[r]{" + r"\\[-1pt]".join(f"${line}$" for line in expression_lines) + "}"


def write_latex_table(
    path: Path,
    rows: list[FeatureRow],
    split: str,
    target_counts: dict[str, int],
    coefficient_tolerance: float,
    minimum_intended_recall: float,
) -> None:
    split_label = "held-out test" if split == "test" else split
    lines = [
        "% Generated by scripts/hierarchy/plot_a_prefix_empirical_molecule_table.py.",
        "% Preamble requirements:",
        "%   \\usepackage{array}",
        "%   \\usepackage{booktabs}",
        "%   \\usepackage{multirow}",
        "%   \\usepackage{tikz}",
        "%   \\usepackage[table]{xcolor}",
        "",
        *(rf"\definecolor{{aprefix{atom}color}}{{HTML}}{{{color.lstrip('#')}}}"
          for atom, color in ATOM_COLORS.items()),
        "",
    ]
    lines.extend(
        [
            "",
            r"\newcommand{\aprefixvcenter}[1]{%",
            r"  \raisebox{-.5\height}{#1}%",
            r"}",
            r"\newcommand{\aprefixrecallcell}[2]{%",
            r"  \multicolumn{1}{>{\columncolor{orange!#1}\centering\arraybackslash}m{1.25cm}}{%",
            r"    \aprefixvcenter{$#2$}%",
            r"  }%",
            r"}",
            "",
            r"\begin{table*}[t]",
            r"\centering",
            r"\setlength{\tabcolsep}{6pt}",
            r"\setlength{\aboverulesep}{0pt}",
            r"\setlength{\belowrulesep}{0pt}",
            r"\renewcommand{\arraystretch}{1.18}",
            r"\footnotesize",
            r"\begin{tabular}{",
            r"  >{\centering\arraybackslash}m{0.9cm}",
            r"  >{\centering\arraybackslash}m{1.0cm}",
            r"  >{\centering\arraybackslash}m{5.0cm}",
            r"  >{\centering\arraybackslash}m{1.25cm}",
            r"  >{\centering\arraybackslash}m{1.25cm}",
            r"  >{\centering\arraybackslash}m{1.25cm}",
            r"  >{\centering\arraybackslash}m{1.25cm}",
            r"}",
            r"\toprule",
            r"dict. size",
            r"& $i$",
            r"& $\hat a_i$ decoder molecule",
            r"& \cellcolor{aprefixahcolor}recall on \texttt{ah}",
            r"& \cellcolor{aprefixabcolor}recall on \texttt{ab}",
            r"& \cellcolor{aprefixapcolor}recall on \texttt{ap}",
            r"& \cellcolor{aprefixascolor}recall on \texttt{as}",
            r"\\",
            r"\midrule",
        ]
    )

    row_number = 0
    for width, _top_k in WIDTH_TOPK:
        width_rows = [row for row in rows if row.width == width]
        width_label = "131\\mathrm{K}" if width == 131072 else f"{width:,}"
        for within_group, row in enumerate(width_rows):
            if within_group == 0:
                if len(width_rows) == 1:
                    lines.append(f"\\aprefixvcenter{{${width_label}$}}")
                else:
                    vertical_shift = 1.5 * (len(width_rows) - 1)
                    lines.append(
                        f"\\multirow{{{len(width_rows)}}}{{*}}[-{vertical_shift:g}ex]"
                        f"{{\\aprefixvcenter{{${width_label}$}}}}"
                    )
            else:
                lines.append("")
            molecule = latex_molecule(row, coefficient_tolerance)
            decomposition = latex_decomposition(row, coefficient_tolerance)
            molecule_cell = (
                r"\aprefixvcenter{\begin{tabular}{@{}>{\raggedleft\arraybackslash}m{3.05cm}"
                r"@{\hspace{0.30cm}}>{\centering\arraybackslash}m{1.45cm}@{}}" + "\n"
                + r"\aprefixvcenter{" + decomposition + "}" + "\n"
                + "& " + r"\aprefixvcenter{" + molecule + "}" + "\n"
                + r"\end{tabular}}"
            )
            lines.extend(
                [
                    f"& \\aprefixvcenter{{{row.feature}}}",
                    f"& {molecule_cell}",
                ]
            )
            for category in ACTIVATION_CATEGORIES:
                recall = row.recalls[category]
                shade = int(round(100.0 * recall))
                lines.append(f"& \\aprefixrecallcell{{{shade}}}{{{recall:.3f}}}")
            lines.append(r"\\")
            row_number += 1
            if within_group + 1 < len(width_rows):
                lines.append(r"\cmidrule(lr){2-7}")
        if row_number < len(rows):
            lines.append(r"\midrule")

    counts = ", ".join(
        f"\\texttt{{{category}}}: $n={target_counts[category]:,}$"
        for category in ACTIVATION_CATEGORIES
    )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Empirical a-prefix feature splitting for Gemini SAEs at",
            r"$m\in\{512,2048,131072\}$. Rows are unique learned features selected",
            r"by the best probes for \texttt{a}, \texttt{ah}, \texttt{ab},",
            r"\texttt{ap}, or \texttt{as}; shared feature IDs are shown once.",
            f"Only rows reaching recall at least ${minimum_intended_recall:g}$ on one or more categories",
            r"that selected the feature are included.",
            f"Cells report recall on {split_label} ground-truth prefix texts ({counts}).",
            r"Each row uses the training-set-selected threshold for the first category",
            r"selecting that feature in the order \texttt{a}, \texttt{ah}, \texttt{ab},",
            r"\texttt{ap}, \texttt{as}; shared broad rows therefore use the \texttt{a} threshold.",
            f"Molecules show requested atoms with positive decoder coefficient at least ${coefficient_tolerance:g}$",
            r"in the full regression onto 131K a-prefix decoder directions; circle",
            r"radius is proportional to the cube root of the coefficient.}",
            r"\label{tab:a-prefix-empirical-feature-splitting}",
            r"\end{table*}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")






def parse_args() -> argparse.Namespace:
    hierarchy_dir = _paper_path(get_path("hierarchy_data_dir"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemini")
    parser.add_argument("--hierarchy-dir", type=Path, default=hierarchy_dir)
    parser.add_argument(
        "--probe-results-dir",
        type=Path,
        default=hierarchy_dir / "probe_results" / "prefix_molecule_selection",
    )
    parser.add_argument(
        "--coefficients",
        type=Path,
        default=_paper_path(
            "full_experiments/results/a_prefix_linear_reconstruction/"
            "gemini_all_prefixes_decoder_basis/reconstruction_coefficients.csv"
        ),
    )
    parser.add_argument(
        "--out-stem",
        type=Path,
        default=_paper_path("full_experiments/plots/a_prefix_empirical_molecule_table"),
    )
    parser.add_argument("--split", choices=("train", "test", "all"), default="test")
    parser.add_argument("--coefficient-tolerance", type=float, default=0.015)
    parser.add_argument("--min-intended-recall", type=float, default=0.5)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected, thresholds = selected_features(args.probe_results_dir, args.model)
    coefficients = requested_coefficients(args.coefficients, selected)
    source_rows = read_csv(args.hierarchy_dir / "wordfreq" / "rows.csv")
    masks, target_counts = target_masks(source_rows, args.split)
    recalls = feature_recalls(args.hierarchy_dir, args.model, selected, thresholds, masks)
    rows = assemble_rows(selected, thresholds, coefficients, recalls)
    rows = filter_by_intended_recall(rows, args.min_intended_recall)
    values_path = args.out_stem.with_name(args.out_stem.name + "_values").with_suffix(".csv")
    latex_path = args.out_stem.with_suffix(".tex")
    write_values_csv(
        values_path, rows, target_counts, args.split, args.min_intended_recall
    )
    write_latex_table(
        latex_path,
        rows,
        args.split,
        target_counts,
        args.coefficient_tolerance,
        args.min_intended_recall,
    )
    counts = {
        width: sum(row.width == width for row in rows)
        for width, _top_k in WIDTH_TOPK
    }
    print(
        f"Wrote {latex_path} and {values_path}; "
        f"unique learned-feature rows={counts}",
        flush=True,
    )


if __name__ == "__main__":
    main()
