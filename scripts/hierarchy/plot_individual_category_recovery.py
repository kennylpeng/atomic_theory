"""Plot counts and proportions of categories recovered by frozen SAE detectors."""
from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import argparse
import json
import shlex
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from plot_parent_child_pair_recovery import meets_cutoff, sha256
from recovery_data import (
    DATASETS, MODELS, RANKS, TITLES, WIDTHS, PROBE_RESULTS_DIR,
    load_probes, write_table,
)
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from recovery_plot_axes import add_count_axis

DEFAULT_OUTPUT = _paper_path(__file__).resolve().parents[2] / "full_experiments/plots/individual_category_recovery_f1_gt_0p6_separate_prompts"


def evaluate_categories(probes: dict, cutoff: Fraction, *, strict: bool) -> tuple[list[dict], list[dict]]:
    details = []
    for dataset in DATASETS:
        for model in MODELS:
            for width in WIDTHS:
                for category_id, probe in sorted(probes[dataset, model, width].items()):
                    row = {
                        "dataset": dataset, "probe_dataset": probe["dataset"],
                        "model": model, "width": width, "top_k": int(probe["top_k"]),
                        "level": probe["level"], "category_id": category_id, "category": probe["category"],
                        "feature": int(probe["best_feature"]), "threshold": float(probe["threshold"]),
                        "selection_f1": float(probe["train_f1"]), "test_f1": float(probe["test_f1"]),
                        "test_tp": int(probe["test_tp"]), "test_fp": int(probe["test_fp"]), "test_fn": int(probe["test_fn"]),
                        "recovered": int(meets_cutoff(probe, cutoff, strict=strict)),
                    }
                    details.append(row)
    groups = defaultdict(list)
    for row in details:
        for level in ["all", row["level"]]:
            groups[row["dataset"], row["model"], row["width"], level].append(row)
    summary = []
    for (dataset, model, width, level), rows in sorted(groups.items()):
        recovered = [row for row in rows if row["recovered"]]
        summary.append({
            "dataset": dataset, "model": model, "width": width, "top_k": rows[0]["top_k"],
            "level": level, "eligible_categories": len(rows), "recovered_categories": len(recovered),
            "recovered_proportion": len(recovered) / len(rows),
            "recovered_percent": 100 * len(recovered) / len(rows),
            "unique_features_for_recovered_categories": len({row["feature"] for row in recovered}),
        })
    return summary, details


def plot_categories(summary: list[dict], output: Path, cutoff: Fraction, *, strict: bool, by_rank: bool = False, proportions: bool = False, paper_layout: bool = False) -> None:
    if by_rank:
        panels = [("gbif", rank, rank.title()) for rank in RANKS]
        fig, grid = plt.subplots(2, 3, figsize=(22, 11.5))
    else:
        panels = [(dataset, "all", TITLES[dataset]) for dataset in DATASETS]
        fig, grid = plt.subplots(1, 3, figsize=(12.6, 5.2)) if paper_layout else plt.subplots(1, 3, figsize=(22, 6.5))
    axes = grid.ravel()
    comparison = ">" if strict else "≥"
    for panel_index, (ax, (dataset, level, title)) in enumerate(zip(axes, panels)):
        maximum = 0
        cohort_sizes = set()
        for model, color, marker in [("gemini", "#0072B2", "o"), ("nemotron", "#D55E00", "s")]:
            rows = sorted((row for row in summary if (row["dataset"], row["level"], row["model"]) == (dataset, level, model)), key=lambda row: row["width"])
            if len(rows) != len(WIDTHS):
                raise ValueError(f"Missing widths: {dataset}, {level}, {model}")
            cohort_sizes.update(row["eligible_categories"] for row in rows)
            ys = [row["recovered_proportion" if proportions else "recovered_categories"] for row in rows]
            maximum = max(maximum, max(ys))
            ax.plot(WIDTHS, ys, color=color, marker=marker, linewidth=2.8, markersize=7, label=model.title())
        if len(cohort_sizes) != 1:
            raise ValueError(f"Category cohort changes: {dataset}, {level}")
        ax.set_xscale("log", base=2)
        tick_widths = WIDTHS[::4] if paper_layout else WIDTHS
        ax.set_xticks(tick_widths, [f"{width:,}" for width in tick_widths], rotation=45, ha="right", fontsize=16 if paper_layout else 14)
        ax.set_xlabel("SAE width", fontsize=20)
        ax.set_title(title, fontsize=22, pad=13)
        ax.tick_params(axis="y", labelsize=16)
        if proportions:
            ax.set_ylim(-0.025, 1.025)
            ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        else:
            ax.set_ylim(-0.025 * max(maximum, 1), 1.10 * max(maximum, 1))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, min_n_ticks=2))
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        if proportions:
            count_axis = add_count_axis(ax, next(iter(cohort_sizes)), tick_fontsize=16 if paper_layout else 14)
            if paper_layout and panel_index < len(panels) - 1:
                count_axis.set_ylabel("")
        if panel_index % 3 == 0:
            ylabel = ("Proportion recovered" if paper_layout else "Proportion of categories recovered") if proportions else "Recovered categories"
            ax.set_ylabel(f"{ylabel}\n(F1 {comparison} {float(cutoff):g})", fontsize=20)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=21)
    fig.tight_layout(rect=(0, 0, 1, 0.84 if paper_layout else (0.94 if by_rank else 0.90)), w_pad=4.5 if proportions else 2.5, h_pad=2.5)
    stem = "individual_category_recovery_" + ("proportions" if proportions else "counts") + ("_gbif_by_rank" if by_rank else "") + ("_appendix" if paper_layout else "")
    for extension in ["pdf", "png"]:
        fig.savefig(output / f"{stem}.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-root", type=Path, default=PROBE_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--f1-cutoff", default="0.6")
    parser.add_argument("--comparison", choices=["gt", "ge"], default="gt", help="Strict > (gt) or inclusive ≥ (ge).")
    parser.add_argument("--geography-prompts", choices=["shared", "separate"], default="separate")
    args = parser.parse_args()
    cutoff = Fraction(args.f1_cutoff)
    if not 0 < cutoff <= 1:
        parser.error("F1 cutoff must be in (0, 1]")
    strict = args.comparison == "gt"
    comparison = ">" if strict else "≥"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    probes, categories, sources = load_probes(args.probe_root, args.geography_prompts)
    summary, details = evaluate_categories(probes, cutoff, strict=strict)
    write_table(args.output_dir / "category_recovery_summary.csv", summary)
    write_table(args.output_dir / "category_recovery_details.csv", details)
    for proportions in [False, True]:
        for by_rank in [False, True]:
            plot_categories(summary, args.output_dir, cutoff, strict=strict, by_rank=by_rank, proportions=proportions)
    plot_categories(summary, args.output_dir, cutoff, strict=strict, proportions=True, paper_layout=True)
    geography_description = (
        "Geography combines continent detectors evaluated on `what continent is [city] in` and country detectors evaluated on `what country is [city] in`."
        if args.geography_prompts == "separate" else
        "Geography evaluates both continent and country detectors on `Where is [city]?` inputs."
    )
    protocol = "Count each category once when its frozen selection-split detector passes the held-out F1 cutoff. Recovery does not depend on parent or child recovery. Different categories may use the same SAE coordinate; the plotted metric counts categories, not distinct features."
    levels = {dataset: dict(Counter(row["level"] for row in probes[dataset, MODELS[0], WIDTHS[0]].values())) for dataset in DATASETS}
    provenance = {
        "f1_cutoff": str(cutoff), "f1_comparison": ">" if strict else ">=",
        "models": MODELS, "widths": WIDTHS, "category_protocol": protocol,
        "geography_prompts": args.geography_prompts, "geography_prompt_description": geography_description,
        "eligible_categories": {dataset: len(ids) for dataset, ids in categories.items()},
        "eligible_categories_by_level": levels,
        "proportion_protocol": "Recovered categories divided by all eligible categories in that domain or rank. The domain aggregate pools categories across levels; it is not an unweighted average of rank proportions.",
        "category_ids": {dataset: sorted(ids) for dataset, ids in categories.items()},
        "source_sha256": {str(path): sha256(path) for path in sources},
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Counts and proportions of recovered individual categories", "",
        f"Cutoff: held-out F1 {comparison} {float(cutoff):g}, checked from exact TP/FP/FN counts.", "",
        protocol, "",
        "All categories in the existing probes are included, without filtering on parent relationships or the availability of eligible children. Each category uses its original feature and activation threshold selected on the selection split. Labels are defined by category ID, including taxonomic rank, and counted once per model and width. Eligibility is constant across both models and all nine widths. Prefix/geography eligibility requires at least 100 rows; GBIF eligibility requires at least 100 species.", "",
        "The main plot pools the two prefix lengths, continent and country categories, and all six GBIF ranks, respectively. It sums the category counts across levels. The additional taxonomy plots separate the six ranks. Proportions divide recovered counts by all eligible categories in the relevant domain or rank, including categories that fail the cutoff. Domain denominators are 242 (prefixes), 93 (geography), and 499 (taxonomy); the six taxonomy denominators are 4, 12, 33, 131, 276, and 43. Pooled proportions weight every category equally, rather than averaging rank proportions equally. Summary CSVs contain counts, proportions, and percentages by level, plus the number of distinct features used by recovered categories as a separate diagnostic.", "",
        geography_description, "",
        "GBIF retains the existing name-row test split and its documented ambiguous-name limitation. Plots show the original width-dependent sparsity sweep with one trained SAE per model/width. Count plots use raw integer counts with separate scales by panel. Proportion plots use a common 0–1 scale on the left and a linked right-hand count scale (count = proportion × eligible categories in that panel). The `_appendix` figure places the three domains in one row, with linked count axes, so the individual-category and adjacent-pair figures fit together on one paper page. No confidence bands are shown.", "",
        "| Domain | Eligible categories | Gemini at 131,072 | Nemotron at 131,072 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for dataset in DATASETS:
        rows = [next(row for row in summary if (row["dataset"], row["model"], row["width"], row["level"]) == (dataset, model, WIDTHS[-1], "all")) for model in MODELS]
        lines.append(f"| {TITLES[dataset]} | {rows[0]['eligible_categories']} | {rows[0]['recovered_categories']} ({rows[0]['recovered_percent']:.1f}%) | {rows[1]['recovered_categories']} ({rows[1]['recovered_percent']:.1f}%) |")
    command = ["python3", "scripts/hierarchy/plot_individual_category_recovery.py", "--f1-cutoff", args.f1_cutoff, "--comparison", args.comparison, "--geography-prompts", args.geography_prompts, "--output-dir", str(args.output_dir)]
    if args.probe_root != PROBE_RESULTS_DIR:
        command.extend(["--probe-root", str(args.probe_root)])
    lines += ["", "Reproduce from the repository root:", "", "```bash", shlex.join(command), "```", ""]
    (args.output_dir / "README.md").write_text("\n".join(lines))
    print("\n".join(lines[-12:]))
    print(f"Saved figures and audit tables to {args.output_dir}")


if __name__ == "__main__":
    main()
