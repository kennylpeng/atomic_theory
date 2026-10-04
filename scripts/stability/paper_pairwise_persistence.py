"""Persistence from independently selected pairwise maximum matchings at t=0.7.

Run from the research repository: python -m scripts.stability.paper_pairwise_persistence
No source filtering or coordination across target dictionaries is performed.
"""
from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import scipy
from scipy.sparse import csr_matrix, load_npz
from scipy.sparse.csgraph import maximum_bipartite_matching

ROOT = _paper_path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.stability.persistent_matching import validate_witness
from scripts.stability.validate_split_cardinality_outputs import certify_maximum

WIDTHS = (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
MODELS = ("gemini", "nemotron")
METRICS = ("cosine", "pearson")


def independent_persistence(graphs):
    from scripts.stability.persistent_matching import independent_assignments
    return independent_assignments(graphs)


def source_directory(results, model, metric):
    if metric == "cosine":
        return results / "persistent_stability_signed_0.7" / model
    prefix = "" if model == "gemini" else "nemotron_"
    return results / f"{prefix}persistent_stability_pearson_0.7"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def compute(results, output):
    output.mkdir(parents=True, exist_ok=True)
    rows, pairwise, inputs = [], [], {}
    for metric in METRICS:
        for model in MODELS:
            directory = source_directory(results, model, metric)
            for name in ("summary.json", "manifest.json"):
                path = directory / name
                inputs[str(path)] = digest(path)
            baseline_path = directory / "summary.json"
            baseline = {r["width"]: r for r in json.loads(baseline_path.read_text())}
            destination = output / metric / model
            destination.mkdir(parents=True, exist_ok=True)
            for i, width in enumerate(WIDTHS[:-1]):
                targets = WIDTHS[i + 1:]
                previous = baseline[width]
                assert previous["target_widths"] == list(targets)
                assert previous["threshold"] == .7
                graphs = []
                for target in targets:
                    path = directory / f"graph_{width}_to_{target}.npz"
                    graph = load_npz(_paper_location(path))
                    assert graph.shape == (width, target)
                    assert np.all(graph.data == 1), "Expected cached boolean threshold graph"
                    graphs.append(graph)
                    inputs[str(path)] = digest(path)
                selected, assignments = independent_persistence(graphs)
                count = len(selected)
                for target, graph, assignment in zip(targets, graphs, assignments):
                    pairwise.append(dict(model=model, metric=metric, width=width,
                                         target_width=target, edges=graph.nnz,
                                         matched_count=int(np.count_nonzero(assignment >= 0))))
                np.savez_compressed(_paper_location(destination / f"assignments_{width}.npz"),
                                    persistent_source=selected,
                                    **{f"target_{t}": a for t, a in zip(targets, assignments)})
                row = dict(model=model, metric=metric, width=width,
                           persistent_count=count, proportion=count / width)
                rows.append(row)
                print(f"{model:8} {metric:7} {width:6}: {count:6}", flush=True)
    for name, records in (("counts.csv", rows), ("pairwise_counts.csv", pairwise)):
        with (output / name).open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    metadata = dict(
        threshold=.7, comparison="similarity >= float32(0.7), using cached threshold graphs",
        method="Intersection of source IDs matched by independent full-graph maximum-cardinality matchings",
        tie_breaking="Ascending original source/target IDs; sorted CSR; scipy.sparse.csgraph.maximum_bipartite_matching(perm_type='column')",
        versions=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__),
        source_widths=list(WIDTHS[:-1]), target_only_width=WIDTHS[-1],
        denominator="All source features, including inactive features",
        certification="Every pairwise maximum certified with an equal-size vertex cover; intersection witnesses validated",
        cosine_graphs="Exhaustive signed decoder cosine threshold graphs",
        pearson_graphs="Retained bidirectional top-five full-corpus Pearson candidate edges",
        limitations=["Maximum matchings can be nonunique; intersections depend on tie-breaking.",
                     "Pearson intersections are candidate-restricted, with no guaranteed lower-bound relation to exhaustive independently matched intersections."],
        source_files_sha256=inputs,
        implementation_sha256={str(p): digest(p) for p in (
            _paper_path(__file__), ROOT / "scripts/stability/validate_split_cardinality_outputs.py",
            ROOT / "scripts/stability/persistent_matching.py")},
    )
    write_json(output / "manifest.json", metadata)
    write_json(output / "plotted_values.json", dict(threshold=.7, comparisons=rows))
    return rows


def plot(rows, output, *, main_only=False):
    from scripts.stability.stability_plotting import apply_stability_grid, apply_stability_style
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, PercentFormatter

    apply_stability_style()
    styles = {"gemini": ("#0072B2", "o"), "nemotron": ("#D55E00", "s")}
    widths = WIDTHS[:-1]

    def format_axis(ax, proportion, *, main_figure=False):
        ax.set_xscale("log", base=2)
        tick_labels = [str(w) if w < 1024 else f"{w // 1024}K" for w in widths]
        if main_figure:
            tick_labels[-1] = "65K"
        ax.set_xticks(widths, tick_labels)
        ax.set_xlabel("SAE width" if main_figure else "Source SAE width")
        if main_figure:
            ax.set_ylabel("% persistent directions" if proportion else "# persistent directions")
        else:
            ax.set_ylabel("Persistent feature proportion" if proportion else "Persistent feature count")
        ax.set_ylim(0, 1 if proportion else None)
        ax.yaxis.set_major_formatter(PercentFormatter(1) if proportion else FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.spines[["top", "right"]].set_visible(False)
        apply_stability_grid(ax, proportion=proportion)

    def save(fig, name):
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{name}.{extension}", dpi=220)
        plt.close(fig)

    for metric in (("cosine",) if main_only else METRICS):
        label = "Decoder cosine" if metric == "cosine" else "Activation Pearson (retained candidates)"
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), layout="constrained")
        for ax, key in zip(axes, ("persistent_count", "proportion")):
            for model, (color, marker) in styles.items():
                group = [r for r in rows if r["metric"] == metric and r["model"] == model]
                ax.plot(widths, [r[key] for r in group], color=color, marker=marker,
                        linewidth=2.5, label=model if metric == "cosine" else model.title())
            format_axis(ax, key == "proportion", main_figure=metric == "cosine")
        axes[0].legend(frameon=False)
        if metric != "cosine":
            fig.suptitle(f"{label} ≥ 0.7: independent pairwise persistence", fontsize=18)
        save(fig, f"persistent_stability_{metric}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "full_experiments/results")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.results / "persistent_stability_pairwise_0.7"
    rows = compute(args.results, output)
    plot(rows, output)
    print(output / "persistent_stability_cosine.pdf")


if __name__ == "__main__":
    main()
