"""Text normalization and formatting for the target-3290 examples."""

from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import sys

from pathlib import Path

sys.path.insert(0, str(_paper_path(__file__).resolve().parents[1]))

from scripts.analyze_target_3290_activating_texts import DEFAULT_RESULT_DIR

DEFAULT_TEX_OUTPUT = _paper_path(
    "full_experiments/plots/target_3290_thresholded_coactivation_100.tex"
)

DEFAULT_OPACITY_TEX_OUTPUT = _paper_path(
    "full_experiments/plots/target_3290_opacity_coactivation_100.tex"
)

DEFAULT_CSV_OUTPUT = (
    DEFAULT_RESULT_DIR / "target_3290_thresholded_coactivation_100.csv"
)

ATOM_IDS = [8574, 114331, 68236, 72509, 97729]

ATOM_COLORS = {
    8574: "AtomGenericTransform",
    114331: "AtomTransformers",
    68236: "AtomChange",
    72509: "AtomElectrical",
    97729: "AtomGeometry",
}

def normalize_text(value: str) -> str:
    return " ".join(str(value).split()) or "[EMPTY]"

def excerpt(value: str, limit: int) -> str:
    value = normalize_text(value)
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."

def latex_escape(value: str) -> str:
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
    return "".join(replacements.get(character, character) for character in value)

def latex_display_text(value: str, limit: int) -> str:
    """Make a compact text excerpt readable in a LaTeX table cell.

    The companion CSV retains the original text.  Here, corpus-side TeX and
    Markdown markup is flattened so a truncation cannot leave a half-open math
    expression and long formula-heavy examples remain legible.
    """

    value = excerpt(value, limit)
    value = value.replace("\\", "")
    value = value.replace("$", "")
    value = value.replace("{", "(").replace("}", ")")
    value = value.replace("**", "")
    return latex_escape(value)

def opacity_percent(value: float, maximum: float) -> int:
    if value <= 0:
        return 0
    return max(1, min(100, round(100.0 * value / maximum)))
