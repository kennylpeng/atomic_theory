"""Render median-only appendix plots from the validated prevalence bin CSVs."""
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = _paper_path(__file__).resolve().parents[2]
SOURCE = ROOT / "full_experiments/results/sparsity_seed_controls/prevalence_median"
OUTPUT = SOURCE / "paper"
FAMILIES = ("gemini", "nemotron")
WIDTHS = (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)
CONTRASTS = (
    ("wikipedia", "no_wikipedia", "Wiki → Non-Wiki"),
    ("no_wikipedia", "wikipedia", "Non-Wiki → Wiki"),
    ("random1", "random2", "Random 1 → Random 2"),
    ("random2", "random1", "Random 2 → Random 1"),
)


def read_bins(name):
    with (SOURCE / name).open() as stream:
        return [
            row for row in csv.DictReader(stream)
            if row["threshold"] == "median_positive" and row["membership"] == "witness"
        ]


def curve(ax, rows, label, color):
    rows = sorted(rows, key=lambda row: int(row["bin"]))
    assert rows
    assert all(np.isclose(float(row["selection_rate"]), int(row["selected"]) / int(row["n"])) for row in rows)
    ax.plot(
        [float(row["prevalence_median"]) for row in rows],
        [float(row["selection_rate"]) for row in rows],
        "o-", markersize=3, linewidth=1.2, label=label, color=color,
    )


def style(ax, family, cross_distribution):
    ax.set_title(family.capitalize())
    ax.set_xscale("symlog", linthresh=1e-8, linscale=0.5)
    ax.set_xlim(0, 0.01 if cross_distribution else 0.1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks(np.linspace(0, 1, 6))
    ticks = [0, 1e-7, 1e-5, 1e-3]
    if not cross_distribution:
        ticks.append(1e-1)
    ax.set_xticks(ticks, ["0", r"$10^{-7}$", r"$10^{-5}$", r"$10^{-3}$"] + ([] if cross_distribution else [r"$10^{-1}$"]))
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Minimum prevalence in both distributions" if cross_distribution else "Feature prevalence")


def save(fig, name):
    for extension in ("pdf", "png"):
        fig.savefig(OUTPUT / f"{name}.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "legend.fontsize": 8, "pdf.fonttype": 42})
    widths = read_bins("width_bins.csv")
    cross = read_bins("cross_distribution_bins.csv")
    for is_cross in (False, True):
        fig, axes = plt.subplots(1, 2, figsize=(7, 3.45), sharey=True)
        for ax, family in zip(axes, FAMILIES):
            if is_cross:
                for color, (source, target, label) in zip(plt.get_cmap("tab10").colors, CONTRASTS):
                    rows = [r for r in cross if r["family"] == family and r["source_split"] == source and r["target_split"] == target and r["prevalence"] == "minimum_both"]
                    assert sum(int(r["n"]) for r in rows) == 16384
                    curve(ax, rows, label, color)
            else:
                for color, width in zip(plt.get_cmap("viridis")(np.linspace(0.05, 0.9, len(WIDTHS))), WIDTHS):
                    rows = [r for r in widths if r["family"] == family and int(r["width"]) == width]
                    assert sum(int(r["n"]) for r in rows) == width
                    curve(ax, rows, f"{width:,}", color)
            style(ax, family, is_cross)
        axes[0].set_ylabel("Fraction selected in direction matching" if is_cross else "Fraction selected in persistent matching")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2 if is_cross else 4, frameon=False, title="Source → target" if is_cross else "SAE width")
        fig.tight_layout(rect=(0, 0.21, 1, 1), w_pad=1.5)
        save(fig, "cross_distribution_prevalence_median" if is_cross else "width_prevalence_median")
    paths = [_paper_path(__file__), SOURCE / "width_bins.csv", SOURCE / "cross_distribution_bins.csv"]
    provenance = {
        "description": "Median-only paper figures; all saved matching witnesses and bins unchanged.",
        "sources": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    }
    (OUTPUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
