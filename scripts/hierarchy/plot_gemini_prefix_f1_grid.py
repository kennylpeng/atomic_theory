"""Single-page 5x5 prefix F1 analogs of the main a-prefix panel, excluding x."""
from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import argparse
import csv
import hashlib
import json
import string
import shlex
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import numpy as np

from plot_wordfreq_a_prefix_figures import CHILD_COLOR, PARENT_COLOR, SUMMARY, WIDTHS

DEFAULT_OUTPUT = _paper_path(__file__).resolve().parents[2] / "full_experiments/plots/gemini_letter_prefix_f1"
LETTERS = string.ascii_lowercase.replace("x", "")


def read_rows(path: Path, model: str = "gemini") -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream, delimiter="\t") if row["dataset"] == "wordfreq" and row["model"] == model]
    keys = {(row["category_id"], int(row["width"])) for row in rows}
    if len(keys) != len(rows):
        raise ValueError("Duplicate category/width records in the original figure summary")
    for row in rows:
        tp, fp, fn = (int(row["test_" + key]) for key in ("tp", "fp", "fn"))
        if abs(float(row["test_f1"]) - 2 * tp / (2 * tp + fp + fn)) > 1e-12:
            raise ValueError(f"Inconsistent test F1: {row['category_id']}")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", choices=["gemini", "nemotron"], default="gemini")
    parser.add_argument("--widths", type=int, nargs="+", default=WIDTHS)
    parser.add_argument("--top-k", type=int, help="Require a fixed Top-K value in every source row.")
    args = parser.parse_args()
    widths = sorted(args.widths)
    if not widths or min(widths) <= 0 or len(set(widths)) != len(widths):
        raise ValueError("Widths must be distinct positive integers")
    rows = read_rows(args.summary, args.model)
    if args.top_k is not None and any(int(row["top_k"]) != args.top_k for row in rows):
        raise ValueError("Source rows do not all use the requested fixed Top-K")
    by_category = defaultdict(dict)
    for row in rows:
        by_category[row["category"]][int(row["width"])] = row
    expected = set(widths)
    if {category for category in by_category if len(category) == 1} != set(string.ascii_lowercase):
        raise ValueError("Expected all 26 parent-prefix categories")
    for category, records in by_category.items():
        if set(records) != expected:
            raise ValueError(f"Incomplete widths for prefix {category}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(5, 5, figsize=(11.6, 14.2), sharex=True, sharey=True)
    curve_rows, source_rows, child_counts = [], [], {}
    for index, (letter, ax) in enumerate(zip(LETTERS, axes.ravel())):
        children = sorted(category for category in by_category if len(category) == 2 and category.startswith(letter))
        child_counts[letter] = len(children)
        parent_values = [float(by_category[letter][width]["test_f1"]) for width in widths]
        for child in children:
            values = [float(by_category[child][width]["test_f1"]) for width in widths]
            ax.plot(widths, values, color=CHILD_COLOR, linewidth=0.65, alpha=0.23, zorder=1)
            ax.scatter(widths, values, color=CHILD_COLOR, s=7, alpha=0.35, edgecolors="none", zorder=2)
        if children:
            child_means = [float(np.mean([float(by_category[child][width]["test_f1"]) for child in children])) for width in widths]
            ax.plot(widths, child_means, color=CHILD_COLOR, linewidth=1.9, marker="o", markersize=3.2, zorder=3)
        else:
            child_means = [None] * len(widths)
        ax.plot(widths, parent_values, color=PARENT_COLOR, linewidth=2.1, marker="o", markersize=3.6, zorder=4)
        for width, parent_f1, child_mean in zip(widths, parent_values, child_means):
            curve_rows.append({"prefix": letter, "width": width, "parent_test_f1": parent_f1, "eligible_children": len(children), "mean_child_test_f1": "" if child_mean is None else child_mean})
            for category in [letter, *children]:
                original = by_category[category][width]
                source_rows.append({"prefix": letter, **original})
        ax.set_title(f"{letter}-prefix", fontsize=16, pad=6)
        ax.set_xscale("log", base=2)
        ax.set_xlim(min(widths) / 2**0.25, max(widths) * 2**0.25)
        tick_widths = widths if len(widths) <= 5 else [widths[0], widths[len(widths) // 2], widths[-1]]
        ax.set_xticks(tick_widths, [f"{width:,}" for width in tick_widths], rotation=55, ha="right")
        ax.tick_params(axis="x", labelsize=10 if len(tick_widths) > 3 else 13, labelbottom=index >= 20)
        ax.set_ylim(-0.02, 1.02)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.tick_params(axis="y", labelsize=13, labelleft=index % 5 == 0)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    handles = [
        Line2D([0], [0], color=PARENT_COLOR, linewidth=2.1, marker="o", markersize=4, label="Parent prefix"),
        Line2D([0], [0], color=CHILD_COLOR, linewidth=1.9, marker="o", markersize=4, label="Mean child F1"),
        Line2D([0], [0], color=CHILD_COLOR, linewidth=0.8, marker="o", markersize=3, alpha=0.4, label="Individual children"),
    ]
    title = f"{args.model.capitalize()} letter-prefix recovery"
    if args.top_k is not None:
        title += f" (k={args.top_k})"
    fig.suptitle(title, fontsize=19, y=1.027)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 0.995), ncol=3, fontsize=14, frameon=False, columnspacing=1.2)
    fig.supxlabel("SAE width", fontsize=18, y=0.007)
    fig.supylabel("Held-out test F1", fontsize=18, x=0.005)
    fig.subplots_adjust(left=0.072, right=0.994, top=0.946, bottom=0.078, wspace=0.13, hspace=0.26)
    stem = args.output_dir / f"{args.model}_prefix_f1_a_to_z_excluding_x_grid"
    for extension in ["pdf", "png"]:
        fig.savefig(stem.with_suffix("." + extension), dpi=220, bbox_inches="tight")
    plt.close(fig)
    write_csv(args.output_dir / "prefix_f1_curves.csv", curve_rows)
    write_csv(args.output_dir / "prefix_f1_source_rows.csv", source_rows)
    with args.summary.open("rb") as stream:
        digest = hashlib.sha256(stream.read()).hexdigest()
    provenance = {"summary": str(args.summary), "summary_sha256": digest, "model": args.model, "top_k": args.top_k, "prefixes": list(LETTERS), "widths": widths, "child_counts": child_counts, "source_rows": len(source_rows), "curve_rows": len(curve_rows), "protocol": "Same saved wordfreq detector selection and held-out test F1 as Figure 3a's a-prefix panel. Each category has its own selection-chosen feature and threshold. Parent curves show its detector F1; child means are unweighted means over eligible two-letter categories. All requested widths and all eligible children are included, with no F1 filtering. The grid includes a and excludes x, which has no eligible two-letter children."}
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    command = ["python", "scripts/hierarchy/plot_gemini_prefix_f1_grid.py", "--summary", str(args.summary),
               "--output-dir", str(args.output_dir), "--model", args.model, "--widths", *map(str, widths)]
    if args.top_k is not None:
        command += ["--top-k", str(args.top_k)]
    (args.output_dir / "README.md").write_text(
        f"# {title}: a–z grid excluding x\n\n"
        "One 5×5 grid containing all 25 letter-prefix panels with eligible children, with the same colors and curve definitions as the a-prefix panel. "
        "Dark lines show parent F1; green bold lines show the unweighted mean child F1; faint green lines and points show individual two-letter children. "
        "Each category uses its own train-selected feature and threshold, evaluated unchanged on the held-out test split. "
        "All eligible children are included without F1 filtering. The a panel is included; x is omitted because it has no eligible two-letter children.\n\n"
        f"Widths: {', '.join(map(str, widths))}.\n\n"
        f"Reproduce:\n```bash\n{shlex.join(command)}\n```\n"
    )
    print(f"Saved one 5x5 grid with {len(LETTERS)} prefixes, {len(curve_rows)} parent/width rows, and {len(source_rows)} category/width rows to {args.output_dir}")
    print("Child counts:", child_counts)


if __name__ == "__main__":
    main()
