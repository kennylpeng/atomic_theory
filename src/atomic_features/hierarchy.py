#!/usr/bin/env python3
"""Export the main hierarchy width-regret and mean-F1 results as LaTeX tables."""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


WIDTHS = (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
WIDTH_LABELS = {
    512: "512",
    1024: "1K",
    2048: "2K",
    4096: "4K",
    8192: "8K",
    16384: "16K",
    32768: "32K",
    65536: "65K",
    131072: "131K",
}
MODELS = ("gemini", "nemotron")


@dataclass(frozen=True)
class LevelSpec:
    domain: str
    dataset: str
    level: str
    label: str


LEVEL_SPECS = (
    LevelSpec("Prefix", "wordfreq", "one_letter_prefix", "1-letter"),
    LevelSpec("Prefix", "wordfreq", "two_letter_prefix", "2-letter"),
    LevelSpec("Cities", "geonames_continent_questions", "continent", "Continent"),
    LevelSpec("Cities", "geonames_country_questions", "country", "Country"),
    LevelSpec("GBIF", "gbif", "kingdom", "Kingdom"),
    LevelSpec("GBIF", "gbif", "phylum", "Phylum"),
    LevelSpec("GBIF", "gbif", "class", "Class"),
    LevelSpec("GBIF", "gbif", "order", "Order"),
    LevelSpec("GBIF", "gbif", "family", "Family"),
    LevelSpec("GBIF", "gbif", "genus", "Genus"),
)
SPEC_BY_DATASET_LEVEL = {(spec.dataset, spec.level): spec for spec in LEVEL_SPECS}
MAIN_DATASETS = {spec.dataset for spec in LEVEL_SPECS}


@dataclass(frozen=True)
class RegretObservation:
    domain: str
    level: str
    model: str
    category_key: tuple[str, str]
    width: int
    test_f1: float
    regret: float


@dataclass(frozen=True)
class CellSummary:
    count: int
    mean_regret: float
    median_regret: float
    within_tolerance: int
    mean_test_f1: float | None = None

    @property
    def within_fraction(self) -> float:
        return self.within_tolerance / self.count


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"dataset", "model", "width", "level", "category_id", "test_f1"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
        return list(reader)


def compute_regrets(
    rows: Iterable[dict[str, str]],
    *,
    models: Sequence[str] = MODELS,
) -> list[RegretObservation]:
    """Compute oracle regret against the best test F1 in each width trajectory."""
    selected_models = set(models)
    unknown_models = selected_models - set(MODELS)
    if unknown_models:
        raise ValueError(f"Unknown models: {sorted(unknown_models)}")

    trajectories: dict[tuple[str, str, str], dict[int, tuple[str, float]]] = (
        defaultdict(dict)
    )
    for row in rows:
        dataset = row["dataset"]
        model = row["model"]
        if dataset not in MAIN_DATASETS or model not in selected_models:
            continue

        width = int(row["width"])
        if width not in WIDTHS:
            continue
        level = row["level"]
        if (dataset, level) not in SPEC_BY_DATASET_LEVEL:
            raise ValueError(f"Unexpected level {level!r} for main dataset {dataset!r}")

        test_f1 = float(row["test_f1"])
        if not math.isfinite(test_f1) or not 0.0 <= test_f1 <= 1.0:
            raise ValueError(
                f"Invalid test_f1={row['test_f1']!r} for "
                f"{dataset}/{model}/{row['category_id']}/{width}"
            )

        key = (dataset, model, row["category_id"])
        if width in trajectories[key]:
            raise ValueError(f"Duplicate width {width} for trajectory {key}")
        trajectories[key][width] = (level, test_f1)

    if not trajectories:
        raise ValueError(
            "No rows matched the main hierarchy datasets and requested models"
        )

    expected_widths = set(WIDTHS)
    observations: list[RegretObservation] = []
    for (dataset, model, category_id), values in sorted(trajectories.items()):
        observed_widths = set(values)
        if observed_widths != expected_widths:
            missing = sorted(expected_widths - observed_widths)
            extra = sorted(observed_widths - expected_widths)
            raise ValueError(
                f"Incomplete trajectory {(dataset, model, category_id)}: "
                f"missing={missing}, extra={extra}"
            )

        levels = {level for level, _ in values.values()}
        if len(levels) != 1:
            raise ValueError(
                f"Inconsistent levels for trajectory {(dataset, model, category_id)}: "
                f"{sorted(levels)}"
            )
        level = next(iter(levels))
        spec = SPEC_BY_DATASET_LEVEL[(dataset, level)]
        best_f1 = max(test_f1 for _, test_f1 in values.values())
        for width in WIDTHS:
            test_f1 = values[width][1]
            observations.append(
                RegretObservation(
                    domain=spec.domain,
                    level=level,
                    model=model,
                    category_key=(dataset, category_id),
                    width=width,
                    test_f1=test_f1,
                    regret=max(0.0, best_f1 - test_f1),
                )
            )
    return observations


def median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def summarize_regrets(
    observations: Iterable[RegretObservation],
    *,
    tolerance: float,
) -> dict[tuple[str, int], CellSummary]:
    if tolerance < 0.0:
        raise ValueError("Tolerance must be non-negative")

    grouped: dict[tuple[str, int], list[RegretObservation]] = defaultdict(list)
    for observation in observations:
        grouped[(observation.level, observation.width)].append(observation)

    summaries: dict[tuple[str, int], CellSummary] = {}
    for key, group in grouped.items():
        regrets = [observation.regret for observation in group]
        summaries[key] = CellSummary(
            count=len(group),
            mean_regret=sum(regrets) / len(regrets),
            median_regret=median(regrets),
            within_tolerance=sum(regret <= tolerance + 1e-12 for regret in regrets),
            mean_test_f1=sum(observation.test_f1 for observation in group) / len(group),
        )
    return summaries


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _render_latex(
    summaries: dict[tuple[str, int], CellSummary],
    *,
    tolerance: float | None,
    caption: str,
    label: str,
    mean_decimals: int,
    percent_decimals: int,
    include_within_fraction: bool,
    metric: str,
) -> str:
    if not re.fullmatch(r"[A-Za-z0-9:._-]+", label):
        raise ValueError(
            "LaTeX label may contain only letters, digits, ':', '.', '_', and '-'"
        )
    if mean_decimals < 0 or percent_decimals < 0:
        raise ValueError("Decimal counts must be non-negative")
    if metric not in {"mean_regret", "mean_test_f1"}:
        raise ValueError(f"Unknown table metric: {metric!r}")
    if include_within_fraction and metric != "mean_regret":
        raise ValueError("Within-tolerance percentages are defined only for regret")

    def metric_value(cell: CellSummary) -> float:
        value = getattr(cell, metric)
        if value is None:
            raise ValueError(f"Missing {metric} in a table cell")
        return value

    higher_is_better = metric == "mean_test_f1"

    available_specs = [
        spec
        for spec in LEVEL_SPECS
        if any((spec.level, width) in summaries for width in WIDTHS)
    ]
    if not available_specs:
        raise ValueError("No hierarchy levels are available to render")

    lines = [
        "% Generated by scripts/hierarchy/export_hierarchy_regret_latex.py.",
        "% Requires \\usepackage{booktabs}, \\usepackage{graphicx}, and \\usepackage{xcolor}.",
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{llr{'c' * len(WIDTHS)}}}",
        r"\toprule",
        "Domain & Level & $N$ & "
        + " & ".join(WIDTH_LABELS[width] for width in WIDTHS)
        + r" \\",
        r"\midrule",
    ]

    previous_domain: str | None = None
    for spec in available_specs:
        if previous_domain is not None and spec.domain != previous_domain:
            lines.append(r"\midrule")
        previous_domain = spec.domain

        cells = [summaries.get((spec.level, width)) for width in WIDTHS]
        if any(cell is None for cell in cells):
            missing = [width for width, cell in zip(WIDTHS, cells) if cell is None]
            raise ValueError(
                f"Missing summary cells for {spec.domain}/{spec.label}: {missing}"
            )
        complete_cells = [cell for cell in cells if cell is not None]
        counts = {cell.count for cell in complete_cells}
        if len(counts) != 1:
            raise ValueError(
                f"Cell counts vary across widths for {spec.domain}/{spec.label}"
            )

        distinct_values: list[float] = []
        cell_values = [metric_value(cell) for cell in complete_cells]
        for candidate in sorted(cell_values, reverse=higher_is_better):
            if not any(
                math.isclose(candidate, ranked_value, rel_tol=0.0, abs_tol=1e-12)
                for ranked_value in distinct_values
            ):
                distinct_values.append(candidate)
        best_value = distinct_values[0]
        second_value = distinct_values[1] if len(distinct_values) > 1 else None

        rendered_cells = []
        for cell, numeric_value in zip(complete_cells, cell_values):
            value = f"{numeric_value:.{mean_decimals}f}"
            if include_within_fraction:
                value += f" / {100.0 * cell.within_fraction:.{percent_decimals}f}\\%"
            if math.isclose(numeric_value, best_value, rel_tol=0.0, abs_tol=1e-12):
                value = rf"\textcolor{{blue}}{{\textbf{{{value}}}}}"
            elif second_value is not None and math.isclose(
                numeric_value, second_value, rel_tol=0.0, abs_tol=1e-12
            ):
                value = rf"\textcolor{{red}}{{\textbf{{{value}}}}}"
            rendered_cells.append(value)

        lines.append(
            f"{spec.domain} & {spec.label} & {complete_cells[0].count} & "
            + " & ".join(rendered_cells)
            + r" \\"
        )

    if metric == "mean_test_f1":
        metric_note = r"Each cell reports mean held-out test F1."
        rank_note = (
            r"Bold blue marks the highest unrounded mean F1 in each row; bold red marks "
            r"the second-highest distinct unrounded mean F1."
        )
    else:
        metric_note = (
            r"Regret is the maximum held-out test F1 across the nine widths minus the test F1 "
            r"at the displayed width, computed within each dataset, model, and category. "
        )
        if include_within_fraction:
            if tolerance is None:
                raise ValueError(
                    "Tolerance is required when rendering within-tolerance percentages"
                )
            metric_note += (
                r"Each cell reports mean regret / percentage of probes with regret "
                rf"at most {tolerance:g}."
            )
        else:
            metric_note += r"Each cell reports mean regret."
        rank_note = (
            r"Bold blue marks the lowest unrounded mean regret in each row; bold red marks "
            r"the second-lowest distinct unrounded mean."
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\vspace{0.4em}",
            r"\begin{minipage}{\textwidth}",
            r"\footnotesize",
            r"\emph{Notes:} "
            + metric_note
            + " "
            + rank_note
            + r" Ties share a rank. Models are pooled; $N$ counts category--model trajectories.",
            r"\end{minipage}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def render_latex(
    summaries: dict[tuple[str, int], CellSummary],
    *,
    tolerance: float,
    caption: str,
    label: str,
    mean_decimals: int = 3,
    percent_decimals: int = 1,
) -> str:
    """Render mean regret and the within-tolerance percentage in each cell."""
    return _render_latex(
        summaries,
        tolerance=tolerance,
        caption=caption,
        label=label,
        mean_decimals=mean_decimals,
        percent_decimals=percent_decimals,
        include_within_fraction=True,
        metric="mean_regret",
    )


def render_mean_only_latex(
    summaries: dict[tuple[str, int], CellSummary],
    *,
    caption: str,
    label: str,
    mean_decimals: int = 4,
) -> str:
    """Render a companion table containing only mean regret in each cell."""
    return _render_latex(
        summaries,
        tolerance=None,
        caption=caption,
        label=label,
        mean_decimals=mean_decimals,
        percent_decimals=0,
        include_within_fraction=False,
        metric="mean_regret",
    )


def render_mean_f1_latex(
    summaries: dict[tuple[str, int], CellSummary],
    *,
    caption: str,
    label: str,
    mean_decimals: int = 5,
) -> str:
    """Render a companion table containing mean held-out test F1 in each cell."""
    return _render_latex(
        summaries,
        tolerance=None,
        caption=caption,
        label=label,
        mean_decimals=mean_decimals,
        percent_decimals=0,
        include_within_fraction=False,
        metric="mean_test_f1",
    )


def write_latex(path: Path, latex: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(latex, encoding="utf-8")
