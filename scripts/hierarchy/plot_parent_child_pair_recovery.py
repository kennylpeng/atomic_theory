"""Plot counts and proportions of recovered parent-child pairs using frozen SAE probes."""
from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import argparse
import hashlib
import json
import shlex
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

from recovery_data import (
    DATASETS, MODELS, RANKS, TITLES, WIDTHS,
    GBIF_METADATA_DIR, HIERARCHY_DATA_DIR, PROBE_RESULTS_DIR,
    build_families, load_probes, passes, write_table,
)
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from recovery_plot_axes import add_count_axis

DEFAULT_OUTPUT = _paper_path(__file__).resolve().parents[2] / "full_experiments/plots/parent_child_pair_recovery_f1_0p7_separate_prompts"


def meets_cutoff(row: dict, cutoff: Fraction, *, strict: bool = False) -> bool:
    inclusive_pass = passes(row, cutoff)  # Also validates F1 against exact counts.
    if not strict:
        return inclusive_pass
    tp, fp, fn = (int(row[f"test_{key}"]) for key in ("tp", "fp", "fn"))
    return inclusive_pass and 2 * tp * cutoff.denominator > cutoff.numerator * (2 * tp + fp + fn)


def evaluate_pairs(probes: dict, parents: dict, cutoff: Fraction, *, strict: bool = False, original_parents: dict | None = None) -> tuple[list[dict], list[dict]]:
    details = []
    for dataset in DATASETS:
        for model in MODELS:
            for width in WIDTHS:
                probes_at_width = probes[dataset, model, width]
                for parent, children in parents[dataset].items():
                    reference_children = (parents if original_parents is None else original_parents)[dataset].get(parent, [])
                    p = probes_at_width[parent]
                    parent_pass = meets_cutoff(p, cutoff, strict=strict)
                    for child in children:
                        c = probes_at_width[child]
                        child_pass = meets_cutoff(c, cutoff, strict=strict)
                        distinct = int(p["best_feature"]) != int(c["best_feature"])
                        row = {
                            "dataset": dataset, "model": model, "width": width, "top_k": int(p["top_k"]),
                            "parent_level": p["level"], "child_level": c["level"],
                            "parent": parent, "child": child,
                            "parent_eligible_children": len(children),
                            "in_original_family_cohort": int(len(reference_children) >= 2 and child in reference_children),
                            "parent_probe_dataset": p["dataset"], "child_probe_dataset": c["dataset"],
                            "parent_feature": int(p["best_feature"]), "child_feature": int(c["best_feature"]),
                            "parent_threshold": float(p["threshold"]), "child_threshold": float(c["threshold"]),
                            "parent_test_f1": float(p["test_f1"]), "child_test_f1": float(c["test_f1"]),
                            "parent_pass": int(parent_pass), "child_pass": int(child_pass),
                            "distinct_features": int(distinct),
                            "recovered": int(parent_pass and child_pass and distinct),
                            "both_pass_without_distinctness": int(parent_pass and child_pass),
                        }
                        for prefix, probe in [("parent", p), ("child", c)]:
                            for metric in ["tp", "fp", "fn"]:
                                row[f"{prefix}_test_{metric}"] = int(probe[f"test_{metric}"])
                        details.append(row)
    groups = defaultdict(list)
    for row in details:
        for level in ["all", row["parent_level"]]:
            groups[row["dataset"], row["model"], row["width"], level].append(row)
    summary = []
    for (dataset, model, width, level), rows in sorted(groups.items()):
        recovered = sum(row["recovered"] for row in rows)
        summary.append({
            "dataset": dataset, "model": model, "width": width, "top_k": rows[0]["top_k"],
            "parent_level": level, "eligible_pairs": len(rows),
            "eligible_parents": len({row["parent"] for row in rows}),
            "recovered_pairs": recovered,
            "recovered_proportion": recovered / len(rows),
            "recovered_percent": 100 * recovered / len(rows),
            "pairs_excluded_by_distinctness": sum(row["both_pass_without_distinctness"] - row["recovered"] for row in rows),
            "eligible_pairs_in_original_family_cohort": sum(row["in_original_family_cohort"] for row in rows),
            "recovered_pairs_in_original_family_cohort": sum(row["recovered"] * row["in_original_family_cohort"] for row in rows),
        })
    return summary, details


def plot_pairs(summary: list[dict], output: Path, cutoff: Fraction, *, by_rank: bool = False, strict: bool = False, proportions: bool = False, pair_label: str = "parent–child", paper_layout: bool = False) -> None:
    panels = [("gbif", rank, rank.title() + " → " + RANKS[i + 1]) for i, rank in enumerate(RANKS[:-1])] if by_rank else [(dataset, "all", TITLES[dataset]) for dataset in DATASETS]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 5.2)) if paper_layout else plt.subplots(1, len(panels), figsize=(24, 5.8) if by_rank else (22, 6.5))
    for panel_index, (ax, (dataset, level, title)) in enumerate(zip(axes, panels)):
        maximum = 0
        for model, color, marker in [("gemini", "#0072B2", "o"), ("nemotron", "#D55E00", "s")]:
            rows = sorted((row for row in summary if (row["dataset"], row["parent_level"], row["model"]) == (dataset, level, model)), key=lambda row: row["width"])
            if len(rows) != len(WIDTHS) or len({row["eligible_pairs"] for row in rows}) != 1:
                raise ValueError(f"Missing rows or changing pair cohort: {dataset}, {level}, {model}")
            ys = [row["recovered_proportion" if proportions else "recovered_pairs"] for row in rows]
            maximum = max(maximum, max(ys))
            ax.plot(WIDTHS, ys, color=color, marker=marker, linewidth=2.8, markersize=7, label=model.title())
        ax.set_xscale("log", base=2)
        tick_widths = WIDTHS[::4] if paper_layout else (WIDTHS[::2] if by_rank else WIDTHS)
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
            count_axis = add_count_axis(ax, rows[0]["eligible_pairs"], fontsize=16 if by_rank else 18, tick_fontsize=16 if paper_layout else (12 if by_rank else 14))
            if paper_layout and panel_index < len(panels) - 1:
                count_axis.set_ylabel("")
    comparison = ">" if strict else "≥"
    ylabel = ("Proportion recovered" if paper_layout else "Proportion of pairs recovered") if proportions else f"Recovered {pair_label} pairs"
    for ax in axes[:1]:
        ax.set_ylabel(f"{ylabel}\n(both F1 {comparison} {float(cutoff):g})", fontsize=20)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=21)
    fig.tight_layout(rect=(0, 0, 1, 0.84 if paper_layout else 0.90), w_pad=4.5 if proportions else 2.5, h_pad=2.5 if paper_layout else 1.08)
    stem = "parent_child_pair_recovery_" + ("proportions" if proportions else "counts") + ("_gbif_by_rank" if by_rank else "") + ("_appendix" if paper_layout else "")
    for extension in ["pdf", "png"]:
        fig.savefig(output / f"{stem}.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-root", type=Path, default=PROBE_RESULTS_DIR)
    parser.add_argument("--data-root", type=Path, default=HIERARCHY_DATA_DIR)
    parser.add_argument("--gbif-root", type=Path, default=GBIF_METADATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--f1-cutoff", default="0.7")
    parser.add_argument("--strict", action="store_true", help="Require F1 strictly greater than the cutoff; default is inclusive.")
    parser.add_argument("--geography-prompts", choices=["shared", "separate"], default="separate")
    args = parser.parse_args()
    cutoff = Fraction(args.f1_cutoff)
    if not 0 < cutoff <= 1:
        parser.error("F1 cutoff must be in (0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    probes, categories, sources = load_probes(args.probe_root, args.geography_prompts)
    parents, excluded, metadata_sources = build_families(categories, args.data_root, args.gbif_root, args.geography_prompts, min_children=1)
    summary, details = evaluate_pairs(probes, parents, cutoff, strict=args.strict)
    write_table(args.output_dir / "pair_recovery_summary.csv", summary)
    write_table(args.output_dir / "pair_recovery_details.csv", details)
    for proportions in [False, True]:
        for by_rank in [False, True]:
            plot_pairs(summary, args.output_dir, cutoff, by_rank=by_rank, strict=args.strict, proportions=proportions)
    plot_pairs(summary, args.output_dir, cutoff, strict=args.strict, proportions=True, paper_layout=True)
    geography_description = (
        "Geography uses `what continent is [city] in` for parents and `what country is [city] in` for children, on identical city records and selection/test splits. The two levels use different input embeddings."
        if args.geography_prompts == "separate" else
        "Geography uses the same `Where is [city]?` embeddings for parents and children."
    )
    protocol = "Count each eligible immediate parent-child relationship once when both frozen detectors pass held-out test F1 and use distinct SAE coordinates. Coordinates may be reused across different pairs. Parents with one eligible child are included."
    comparison = ">" if args.strict else "≥"
    provenance = {
        "f1_cutoff": str(cutoff), "f1_comparison": ">" if args.strict else ">=", "models": MODELS, "widths": WIDTHS,
        "pair_protocol": protocol, "geography_prompts": args.geography_prompts,
        "proportion_protocol": "Recovered pairs divided by all eligible pairs in the domain or rank transition. Each eligible relationship has equal weight; pairs failing F1 or coordinate distinctness remain in the denominator.",
        "geography_prompt_description": geography_description,
        "eligible_pairs": {dataset: sum(map(len, parent_map.values())) for dataset, parent_map in parents.items()},
        "eligible_parents": {dataset: len(parent_map) for dataset, parent_map in parents.items()},
        "parent_to_children": parents, "excluded_relationships": excluded,
        "source_sha256": {str(path): sha256(path) for path in sources + metadata_sources},
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Counts and proportions of recovered parent-child pairs", "",
        f"Both parent and child must satisfy held-out F1 {comparison} {float(cutoff):g}, checked using exact TP/FP/FN counts.", "",
        protocol, "",
        "Each category's detector coordinate and activation threshold were selected on the original selection split and are reused unchanged. A parent with five passing children contributes five recovered pairs. These are category-relationship counts, not counts of unique SAE coordinate pairs or a globally injective assignment. The metric does not require two recovered children.", "",
        "All immediate parent-child relationships with both categories present in the saved probes and an unambiguous parent are eligible, including parents that have only one eligible child. Prefix/geography categories require at least 100 rows; GBIF categories require at least 100 species, as in the existing analyses. Missing or ambiguous immediate-parent relationships are excluded. Cohorts stay fixed across models and widths. Proportions divide recovered pairs by all eligible pairs in the domain or rank transition, including pairs failing F1 or coordinate distinctness. Domain denominators are 216 (prefixes), 87 (geography), and 453 (taxonomy). Each relationship receives equal weight; the pooled taxonomy proportion is not an unweighted average of rank-transition proportions. The CSV files contain counts, proportions, and percentages, and also report counts for the subset belonging to the subset of parents with at least two eligible children.", "",
        geography_description, "",
        "Taxonomy pools kingdom→phylum, phylum→class, class→order, order→family, and family→genus relationships. The additional rank-specific plot shows each transition separately. GBIF uses the existing name-row test split with its documented ambiguous-name limitation. Plots use the original sparsity schedule and one trained SAE per model/width. Count plots use raw integer counts with separate scales by panel. Proportion plots use a common 0–1 scale on the left and a linked right-hand count scale (count = proportion × eligible pairs in that panel). The `_appendix` figure places the three domains in one row, with linked count axes, so the individual-category and adjacent-pair figures fit together on one paper page. There are no confidence bands.", "",
        "| Domain | Eligible pairs | Gemini at 131,072 | Nemotron at 131,072 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for dataset in DATASETS:
        rows = [next(row for row in summary if (row["dataset"], row["model"], row["width"], row["parent_level"]) == (dataset, model, WIDTHS[-1], "all")) for model in MODELS]
        lines.append(f"| {TITLES[dataset]} | {rows[0]['eligible_pairs']} | {rows[0]['recovered_pairs']} ({rows[0]['recovered_percent']:.1f}%) | {rows[1]['recovered_pairs']} ({rows[1]['recovered_percent']:.1f}%) |")
    command = ["python3", "scripts/hierarchy/plot_parent_child_pair_recovery.py", "--f1-cutoff", args.f1_cutoff, "--geography-prompts", args.geography_prompts, "--output-dir", str(args.output_dir)]
    if args.strict:
        command.append("--strict")
    for flag, value, default in [("--probe-root", args.probe_root, PROBE_RESULTS_DIR), ("--data-root", args.data_root, HIERARCHY_DATA_DIR), ("--gbif-root", args.gbif_root, GBIF_METADATA_DIR)]:
        if value != default:
            command.extend([flag, str(value)])
    lines += ["", "Reproduce from the repository root:", "", "```bash", shlex.join(command), "```", ""]
    (args.output_dir / "README.md").write_text("\n".join(lines))
    print("\n".join(lines[-12:]))
    print(f"Saved figures and audit tables to {args.output_dir}")


if __name__ == "__main__":
    main()
