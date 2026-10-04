#!/usr/bin/env python3
"""Build the hand-selected compact version of the feature-3290 mix table."""
from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import bisect
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(_paper_path(__file__).resolve().parents[1]))
from scripts.analyze_target_3290_activating_texts import (
    DEFAULT_NUMERIC_CACHE,
    DEFAULT_SPARSE_CACHE,
    cross_activations,
    DEFAULT_RESULT_DIR,
    DEFAULT_TEXT_CATALOG,
    load_family,
)
from scripts.transform_text import (
    ATOM_IDS,
    latex_display_text,
    normalize_text,
    opacity_percent,
)
from scripts.sample_io import load_numeric_samples, load_texts


RESULT_DIR = _paper_path(
    "full_experiments/results/top_feature_examples/"
    "gemini_target_3290_expanded_17atom_sse_0p05"
)
SOURCE_CSV = RESULT_DIR / "target_3290_transform_substring_mix_100.csv"
OPTIMUS_CSVS = [
    RESULT_DIR / "target_3290_exclusive_atom_examples_50.csv",
    RESULT_DIR / "target_3290_thresholded_coactivation_100.csv",
]
OUTPUT_CSV = RESULT_DIR / "target_3290_transform_mix_selected_12.csv"
OUTPUT_TEX = _paper_path(
    "full_experiments/plots/target_3290_transform_mix_selected_12.tex"
)
SELECTED_RANKS = [1, 3, 7, 9, 14, 16, 28, 29, 93, 76, 4, 62]
EXAMPLE_ORDER = [1, 3, 4, 5, 6, 11, 13, 7, 9, 2, 8, 10, 12]
ACTIVATION_ATOM_ORDER = [8574, 68236, 114331, 72509, 97729]
OPACITY_MAX = 0.30
EXCERPT_CHARACTERS = 180
BINNED_EXCERPT_CHARACTERS = 100
BINNED_ROWS_BY_ATOM = {
    8574: [
        43049258, 68836748, 41232611, 84745234, 754521, 81175822,
        69134610, 1222765, 60904002, 3267051, 68321457, 4505558,
    ],
    114331: [
        52098920, 84829800, 55254115, 85183682, 15020092, 76561892,
        69618643, 78680825, 6742627, 52098832, 22915693, 56663279,
    ],
    68236: [
        19313245, 51137184, 50455789, 82417917, 20888456, 83787120,
        1933199, 83519815, 67250734, 17532305, 78162057, 36202301,
    ],
    72509: [
        15007818, 14997951, 69278538, 4648282, 2171584, 4354326,
        4418790, 83124858, 65211459, 67177812, 66220465, 4490300,
    ],
    97729: [
        336031, 930012, 1649582, 2121072, 1765817, 861628,
        1009109, 1962899, 4632659, 3983292, 914439, 1776483,
    ],
}


# One existing Table 3 example per atom, in Table 2 column order. Prefer
# clear, selective texts; the general-transform and electrical examples have
# small nonzero cross-activations, which are displayed rather than suppressed.
SELECTIVE_ROWS_BY_ATOM = {
    8574: 68321457,
    68236: 20888456,
    114331: 84829800,
    72509: 4418790,
    97729: 861628,
}


def load_selective_atom_rows() -> list[dict[str, str]]:
    rows = set(SELECTIVE_ROWS_BY_ATOM.values())
    activations = cross_activations(DEFAULT_SPARSE_CACHE, sorted(rows), ATOM_IDS)
    _, texts = load_texts(
        DEFAULT_TEXT_CATALOG, DEFAULT_NUMERIC_CACHE / "manifest.json", rows
    )
    records = json.loads((DEFAULT_SPARSE_CACHE / "manifest.json").read_text())["shards"]
    starts = [int(record["global_row_start"]) for record in records]
    selected = []
    for atom, global_row in SELECTIVE_ROWS_BY_ATOM.items():
        if global_row not in BINNED_ROWS_BY_ATOM[atom]:
            raise ValueError(f"Selective example {global_row} is absent from Table 3")
        values = activations[global_row]
        if values[atom] < 0.09 or max(v for a, v in values.items() if a != atom) > 0.05:
            raise ValueError(f"Example {global_row} no longer meets the selectivity criteria")
        record = records[bisect.bisect_right(starts, global_row) - 1]
        if not int(record["global_row_start"]) <= global_row < int(record["global_row_stop"]):
            raise ValueError(f"Example {global_row} is outside the target activation cache")
        directory = DEFAULT_SPARSE_CACHE / "shards" / record["relative_shard"] / "gemini_m4096_k32"
        indices = np.load(_paper_location(directory / "indices.npy"), mmap_mode="r", allow_pickle=False)
        data = np.load(_paper_location(directory / "data.npy"), mmap_mode="r", allow_pickle=False)
        start = (global_row - int(record["global_row_start"])) * 32
        target = float(data[start:start + 32][indices[start:start + 32] == 3290].sum())
        row = {
            "source_row": f"Table 3 atom {atom}",
            "selective_atom": str(atom),
            "contains_transform": str("transform" in texts[global_row].casefold()),
            "global_row": str(global_row),
            "relative_shard": str(record["relative_shard"]),
            "text": texts[global_row],
            "target_activation": repr(target),
        }
        row.update({f"activation_{a}": repr(v) for a, v in values.items()})
        row.update({f"above_0.05_{a}": str(v > 0.05) for a, v in values.items()})
        selected.append(row)
    return selected


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def select_rows() -> list[dict[str, str]]:
    source = {int(row["rank"]): row for row in read_csv(SOURCE_CSV)}
    if set(SELECTED_RANKS) - source.keys():
        raise ValueError("A requested rank is absent from the 100-row source table")
    selected: list[dict[str, str]] = []
    for rank in SELECTED_RANKS:
        row = dict(source[rank])
        row["source_row"] = str(rank)
        selected.append(row)

    optimus_candidates: dict[int, dict[str, str]] = {}
    for path in OPTIMUS_CSVS:
        for original in read_csv(path):
            if "optimus prime" not in original["text"].casefold():
                continue
            row = dict(original)
            row["contains_transform"] = str(
                "transform" in original["text"].casefold()
            )
            optimus_candidates[int(row["global_row"])] = row
    if not optimus_candidates:
        raise RuntimeError("No Optimus Prime example found")
    optimus = max(
        optimus_candidates.values(), key=lambda row: float(row["target_activation"])
    )
    optimus["source_row"] = "Optimus"
    selected.append(optimus)
    if len({int(row["global_row"]) for row in selected}) != len(selected):
        raise ValueError("Compact selection contains duplicate corpus rows")
    selected.sort(key=lambda row: -float(row["target_activation"]))
    return [selected[index - 1] for index in EXAMPLE_ORDER]


def focused_excerpt(text: str, phrase: str | None = None) -> str:
    text = normalize_text(text)
    if not phrase or phrase.casefold() not in text.casefold():
        return text
    position = text.casefold().index(phrase.casefold())
    start = max(0, position - EXCERPT_CHARACTERS // 2)
    stop = min(len(text), start + EXCERPT_CHARACTERS)
    start = max(0, stop - EXCERPT_CHARACTERS)
    value = text[start:stop]
    if start:
        value = "..." + value[3:]
    if stop < len(text):
        value = value[:-3] + "..."
    return value


def unicode_safe_text(raw_text: str, escaped_text: str) -> str:
    """Select an explicit Unicode font for excerpts outside basic ASCII."""
    codepoints = [ord(character) for character in raw_text]
    if any(0xAC00 <= value <= 0xD7AF for value in codepoints):
        return rf"\AtomKoreanText{{{escaped_text}}}"
    if any(
        0x0600 <= value <= 0x06FF
        or 0x0750 <= value <= 0x077F
        or 0x08A0 <= value <= 0x08FF
        for value in codepoints
    ):
        return rf"\AtomArabicText{{{escaped_text}}}"
    if any(value > 0x7F for value in codepoints):
        return rf"\AtomUnicodeText{{{escaped_text}}}"
    return escaped_text


def shaded_activation_cell(value: float) -> str:
    """Render a regular-weight activation with opacity-scaled shading."""
    percentage = opacity_percent(value, OPACITY_MAX)
    return rf"\cellcolor{{orange!{percentage}}}{value:.3f}"


def load_curated_binned_examples() -> dict[int, list[tuple[float, str]]]:
    family, atom_ids, _ = load_family(DEFAULT_RESULT_DIR, "omp")
    if atom_ids != ATOM_IDS:
        raise ValueError(f"Unexpected first five OMP atoms: {atom_ids}")
    _, samples = load_numeric_samples(DEFAULT_NUMERIC_CACHE, [family])
    requested_rows = {
        row for rows in BINNED_ROWS_BY_ATOM.values() for row in rows
    }
    _, texts = load_texts(
        DEFAULT_TEXT_CATALOG,
        DEFAULT_NUMERIC_CACHE / "manifest.json",
        requested_rows,
    )
    result: dict[int, list[tuple[float, str]]] = {}
    for atom in ATOM_IDS:
        activation_by_row = {
            row: activation
            for activation, row in samples[("gemini_m131072_k128", atom)]
        }
        missing = set(BINNED_ROWS_BY_ATOM[atom]) - activation_by_row.keys()
        if missing:
            raise ValueError(f"Atom {atom} selections are not binned samples: {missing}")
        result[atom] = sorted(
            [
            (activation_by_row[row], normalize_text(texts[row]))
            for row in BINNED_ROWS_BY_ATOM[atom]
            ],
            key=lambda item: -item[0],
        )
    return result


def render_tex(
    rows: list[dict[str, str]],
    binned_examples: dict[int, list[tuple[float, str]]],
) -> str:
    lines = [
        "% Generated by scripts/build_target_3290_transform_mix_small_table.py.",
        "% Compile with LuaLaTeX or XeLaTeX so all Unicode text is preserved.",
        "% Requires booktabs, array, xcolor[table], fontspec, DejaVu Sans, and Noto Sans CJK KR.",
        r"\providecommand{\AtomUnicodeText}[1]{{\fontspec{DejaVu Sans}#1}}",
        r"\providecommand{\AtomArabicText}[1]{{\fontspec[Script=Arabic,Language=Persian]{DejaVu Sans}#1}}",
        r"\providecommand{\AtomKoreanText}[1]{{\fontspec[Script=Hangul]{Noto Sans CJK KR}#1}}",
        r"\providecolor{TargetFeature}{HTML}{D9E2F3}",
        r"\providecolor{AtomGenericTransform}{HTML}{65C2DB}",
        r"\providecolor{AtomTransformers}{HTML}{F3AD55}",
        r"\providecolor{AtomChange}{HTML}{86C875}",
        r"\providecolor{AtomElectrical}{HTML}{B18AD8}",
        r"\providecolor{AtomGeometry}{HTML}{E889A0}",
        r"\begingroup",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.0}",
        r"\begin{table}",
        r"\centering",
        (
            r"\caption{Selected feature-3290 activating texts and their "
            r"activations on the five 131K atoms. Cell opacity is proportional "
            rf"to activation on a shared linear scale capped at {OPACITY_MAX:g}.}}"
        ),
        r"\label{tab:target-3290-transform-mix-selected}",
        # Booktabs rule padding otherwise leaves unshaded strips above/below cells.
        r"\setlength{\aboverulesep}{0pt}",
        r"\setlength{\belowrulesep}{0pt}",
        r"\begin{tabular}{r p{9.0cm} c c c c c}",
        r"\toprule",
        (
            r"\vphantom{\raisebox{2pt}{$z^{131K}_{114331}$}\raisebox{-2pt}{$z^{131K}_{114331}$}}$z^{4K}_{3290}$ & Text "
            r"& \cellcolor{AtomGenericTransform}$z^{131K}_{8574}$ "
            r"& \cellcolor{AtomChange}$z^{131K}_{68236}$ "
            r"& \cellcolor{AtomTransformers}$z^{131K}_{114331}$ "
            r"& \cellcolor{AtomElectrical}$z^{131K}_{72509}$ "
            r"& \cellcolor{AtomGeometry}$z^{131K}_{97729}$ \\"
        ),
        r"\midrule",
    ]
    for row_index, row in enumerate(rows):
        if row.get("selective_atom") and (row_index == 0 or not rows[row_index - 1].get("selective_atom")):
            lines.extend([
                r"\multicolumn{7}{l}{\textit{Selective examples for individual atoms}} \\",
                r"\midrule",
            ])
        raw_excerpt = focused_excerpt(
            row["text"], "Optimus Prime" if row["source_row"] == "Optimus" else None
        )
        excerpt = unicode_safe_text(
            raw_excerpt,
            latex_display_text(raw_excerpt, EXCERPT_CHARACTERS + 1),
        )
        cells = [
            shaded_activation_cell(
                float(row[f"activation_{atom}"])
            )
            for atom in ACTIVATION_ATOM_ORDER
        ]
        target_cell = shaded_activation_cell(
            float(row["target_activation"])
        )
        lines.append(
            f"{target_cell} & {excerpt} & "
            + " & ".join(cells)
            + r" \\"
        )
        if row_index != len(rows) - 1:
            lines.append(r"\hline")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            r"\par\medskip",
            r"\begin{table}",
            r"\centering",
            r"\caption{Representative precomputed binned-example excerpts for the five 131K atoms. Each column contains twelve examples curated across activation bins and semantic subtypes, sorted by decreasing cached atom activation, shown in gray.}",
            r"\label{tab:target-3290-five-atom-binned-examples}",
            r"\tiny",
            r"\setlength{\tabcolsep}{2pt}",
            r"\renewcommand{\arraystretch}{1.15}",
            (
                r"\begin{tabular}{"
                r">{\raggedright\arraybackslash}p{0.19\textwidth} "
                r">{\raggedright\arraybackslash}p{0.19\textwidth} "
                r">{\raggedright\arraybackslash}p{0.19\textwidth} "
                r">{\raggedright\arraybackslash}p{0.19\textwidth} "
                r">{\raggedright\arraybackslash}p{0.19\textwidth}}"
            ),
            r"\toprule",
            (
                r"\cellcolor{AtomGenericTransform}\textbf{8574} & "
                r"\cellcolor{AtomTransformers}\textbf{114331} & "
                r"\cellcolor{AtomChange}\textbf{68236} & "
                r"\cellcolor{AtomElectrical}\textbf{72509} & "
                r"\cellcolor{AtomGeometry}\textbf{97729} \\"
            ),
            r"\midrule",
        ]
    )
    for example_index in range(12):
        cells = []
        for atom in ATOM_IDS:
            activation, text = binned_examples[atom][example_index]
            excerpt = unicode_safe_text(
                text,
                latex_display_text(text, BINNED_EXCERPT_CHARACTERS),
            )
            cells.append(
                rf"{excerpt} "
                rf"\textcolor{{gray}}{{({activation:.3f})}}"
            )
        lines.append(" & ".join(cells) + r" \\")
        if example_index != 11:
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            r"\endgroup",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "compact_rank",
        "source_row",
        "selective_atom",
        "contains_transform",
        "global_row",
        "relative_shard",
        "text",
        "target_activation",
    ]
    fields += [f"activation_{atom}" for atom in ATOM_IDS]
    fields += [f"above_0.05_{atom}" for atom in ATOM_IDS]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for compact_rank, row in enumerate(rows, start=1):
            output = {field: row.get(field, "") for field in fields}
            output["compact_rank"] = compact_rank
            writer.writerow(output)


def main() -> None:
    rows = select_rows() + load_selective_atom_rows()
    binned_examples = load_curated_binned_examples()
    OUTPUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_TEX.write_text(render_tex(rows, binned_examples), encoding="utf-8")
    write_csv(OUTPUT_CSV, rows)
    print(f"Wrote {OUTPUT_TEX} and {OUTPUT_CSV} ({len(rows)} examples)")


if __name__ == "__main__":
    main()
