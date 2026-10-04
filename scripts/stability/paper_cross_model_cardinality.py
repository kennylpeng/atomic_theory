"""Same-width Gemini–Nemotron maximum-cardinality activation matching."""
from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import argparse
from collections import deque
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, save_npz
from scipy.sparse.csgraph import maximum_bipartite_matching

from scripts.stability.paper_pearson_persistent_stability import threshold_graph

ROOT = _paper_path(__file__).resolve().parents[2]
SOURCE = ROOT / "full_experiments/results/gemini_nemotron_all_widths_sketch512_candidates32_top5"


def certified_matching(graph):
    """Return a matching and an equally sized vertex cover proving optimality."""
    graph = csr_matrix(graph, dtype=bool)
    graph.eliminate_zeros()
    graph.sum_duplicates()
    graph.sort_indices()
    assignment = maximum_bipartite_matching(graph, perm_type="column")
    source = np.flatnonzero(assignment >= 0)
    target = assignment[source]
    assert len(np.unique(target)) == len(source)
    if len(source):
        assert np.all(np.asarray(graph[source, target]))

    # Alternating paths from unmatched left vertices give a minimum vertex cover:
    # unvisited left vertices together with visited right vertices (Konig).
    reverse = np.full(graph.shape[1], -1, dtype=np.int64)
    reverse[target] = source
    seen_left = assignment < 0
    seen_right = np.zeros(graph.shape[1], dtype=bool)
    queue = deque(np.flatnonzero(seen_left & (np.diff(graph.indptr) > 0)))
    while queue:
        left = queue.popleft()
        for right in graph.indices[graph.indptr[left]:graph.indptr[left + 1]]:
            if right == assignment[left] or seen_right[right]:
                continue
            seen_right[right] = True
            partner = reverse[right]
            assert partner >= 0, "An augmenting path contradicts maximal cardinality"
            if not seen_left[partner]:
                seen_left[partner] = True
                queue.append(partner)
    cover_left = np.flatnonzero(~seen_left)
    cover_right = np.flatnonzero(seen_right)
    edges = graph.tocoo()
    assert np.all((~seen_left[edges.row]) | seen_right[edges.col])
    assert len(cover_left) + len(cover_right) == len(source)
    return source, target, cover_left, cover_right


def plot(rows, output, threshold):
    plot_series([dict(threshold=threshold, comparisons=rows)], output)


def plot_series(series, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, PercentFormatter
    from scripts.stability.stability_plotting import apply_stability_grid, apply_stability_style

    apply_stability_style()
    series = sorted(series, key=lambda run: run["threshold"])
    widths = [r["width"] for r in series[0]["comparisons"]]
    for run in series:
        assert [r["width"] for r in run["comparisons"]] == widths
    styles = {0.5: ("#009E73", "s"), 0.6: ("#E69F00", "^"),
              0.7: ("#0072B2", "o"), 0.8: ("#CC79A7", "D")}
    labels = [str(w) if w < 1024 else f"{w // 1024}K" for w in widths]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for ax, key in zip(axes, ("matched_count", "proportion")):
        for index, run in enumerate(series):
            threshold = run["threshold"]
            color, marker = styles.get(threshold, (f"C{index % 10}", "o"))
            ax.plot(widths, [r[key] for r in run["comparisons"]], color=color, marker=marker,
                    linewidth=2.5, markersize=6, label=f"Pearson ≥ {threshold:g}")
        ax.set_xscale("log", base=2)
        ax.set_xticks(widths, labels)
        ax.set_xlabel("SAE width")
        ax.tick_params(axis="x", labelsize=15)
        ax.spines[["top", "right"]].set_visible(False)
        apply_stability_grid(ax, proportion=key == "proportion")
        ax.legend(frameon=False, fontsize=15)
        if key == "matched_count":
            ax.set_ylabel("Matched feature count")
            ax.set_ylim(bottom=0)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:,.0f}"))
        else:
            ax.set_ylabel("Matched feature proportion")
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle("Gemini–Nemotron activation matching at equal SAE widths")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output / f"same_width_matching.{ext}", dpi=220)
    plt.close(fig)


def compute(source, output, threshold):
    if not 0 < threshold <= 1:
        raise ValueError("Threshold must lie in (0, 1].")
    manifest = json.loads((source / "manifest.json").read_text())
    summary = json.loads((source / "summary.json").read_text())
    assert summary["complete"] and summary["manifest_id"] == manifest["id"]
    widths = sorted(m["width"] for m in manifest["models"] if m["family"] == "gemini")
    assert widths == sorted(m["width"] for m in manifest["models"] if m["family"] == "nemotron")
    output.mkdir(parents=True, exist_ok=True)
    rows, hashes = [], {}
    for width in widths:
        directions = []
        for family, other in (("gemini", "nemotron"), ("nemotron", "gemini")):
            path = source / f"{family}_{width}_to_{other}_{width}_top5.npz"
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(_paper_location(path)) as data:
                assert str(data["manifest_id"]) == manifest["id"]
                assert int(data["rows"]) == manifest["rows"]
                assert int(data["source_width"]) == int(data["target_width"]) == width
                assert str(data["source_family"]) == family
                assert str(data["target_family"]) == other
                np.testing.assert_array_equal(data["source_feature_ids"], np.arange(width))
                directions.append((data["target_feature_ids"], data["pearson"]))
        graph, diagnostics = threshold_graph(*directions[0], *directions[1], threshold=threshold)
        left, right, cover_left, cover_right = certified_matching(graph)
        save_npz(_paper_location(output / f"graph_{width}.npz"), graph)
        np.savez_compressed(_paper_location(output / f"matching_{width}.npz"),
                            gemini_feature_ids=left, nemotron_feature_ids=right,
                            cover_gemini_feature_ids=cover_left, cover_nemotron_feature_ids=cover_right,
                            width=width, threshold=threshold, rows=manifest["rows"],
                            source_manifest_id=manifest["id"], candidate_limited=True)
        row = dict(width=width, matched_count=len(left), proportion=len(left) / width,
                   eligible_gemini_features=int(np.count_nonzero(np.diff(graph.indptr))),
                   eligible_nemotron_features=len(np.unique(graph.indices)),
                   minimum_vertex_cover_size=len(cover_left) + len(cover_right),
                   certificate="matching_and_equal_size_vertex_cover", **diagnostics)
        rows.append(row)
        print(json.dumps(row), flush=True)
    payload = dict(complete=True, threshold=threshold,
                   comparison=f"signed continuous activation Pearson >= float32({threshold:g})",
                   objective="maximum cardinality of a one-to-one same-width Gemini–Nemotron matching",
                   denominator="total SAE width, including dead or unmatched features",
                   edge_source="union of threshold-qualified saved top-five neighbors in both directions",
                   duplicate_edge_rule="included if either saved directional score meets the threshold",
                   rows=manifest["rows"], source_directory=str(source.resolve()),
                   source_manifest_id=manifest["id"], source_files_sha256=hashes,
                   source_search={k: manifest[k] for k in ("sketch_dim", "candidate_k", "save_k")},
                   implementation_sha256=hashlib.sha256(_paper_path(__file__).read_bytes()).hexdigest(),
                   threshold_implementation_sha256=hashlib.sha256(
                       (ROOT / "scripts/stability/paper_pearson_persistent_stability.py").read_bytes()).hexdigest(),
                   candidate_limited=True, optimal_on_retained_graph=True,
                   interpretation="lower bound on exhaustive above-threshold matching cardinality, subject to floating-point precision",
                   comparisons=rows)
    (output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (output / "counts.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, output, threshold)
    caption = (
        f"Maximum-cardinality feature matching between Gemini and Nemotron SAEs of the same width, "
        f"using signed Pearson correlation at least ${threshold:g}$ between continuous post-TopK "
        f"activations on {manifest['rows']:,} aligned examples. Left: matched-pair count. Right: count "
        "divided by the common SAE width, including dead and unmatched features in the denominator. "
        "Eligible pairs are the union of retained neighbors from both search directions; counts are "
        "certified optimal on this candidate graph and lower bounds on exhaustive matching counts. "
        "The 131K pair is included because this comparison does not require a larger dictionary."
    )
    (output / "caption.tex").write_text("\\caption{" + caption + "}\n")
    (output / "README.md").write_text(
        "# Same-width cross-model activation matching\n\n" + caption.replace("$", "") + "\n\n"
        "The figure follows the count/proportion layout and styling of Figure 2(a). Each width has "
        "one bipartite matching, so its count and proportion are identical from either model's side. "
        "This uses all features, without a persistence or reciprocal-nearest-neighbor filter. "
        "The objective counts qualifying pairs; it does not maximize the sum of correlations.\n\n"
        f"The input search used {manifest['sketch_dim']}-dimensional sketches, "
        f"{manifest['candidate_k']} candidates per source feature, and {manifest['save_k']} retained "
        "neighbors per direction. The Pearson scores use all aligned rows. Missing candidate edges "
        "can reduce the maximum cardinality. The inclusive threshold is compared in float32, "
        "matching the existing Pearson persistence analysis. If directional scores differ, a pair "
        "qualifies when either score meets the threshold. Sparsity k varies across these main SAEs: "
        "32 at widths 512–4096, 64 at 8192–32768, and 128 at 65536–131072.\n\n"
        "`counts.csv` and `summary.json` contain the plotted values and input provenance. "
        "`matching_WIDTH.npz` stores the one-to-one pairs and a vertex cover of the same size; "
        "the cover certifies optimality on `graph_WIDTH.npz`. The witness matching need not be unique. "
        "`caption.tex` supplies a manuscript caption.\n\n"
        f"Reproduce from the repository root: `python -m scripts.stability.paper_cross_model_cardinality "
        f"--threshold {threshold:g}`. Use `--source` and `--output` for alternate locations.\n"
    )
    return payload


def compute_sweep(source, output, thresholds):
    thresholds = sorted(set(thresholds))
    if not thresholds or any(not 0 < threshold <= 1 for threshold in thresholds):
        raise ValueError("Thresholds must lie in (0, 1].")
    runs = [compute(source, output / f"threshold_{threshold:g}", threshold)
            for threshold in thresholds]
    reference = runs[0]
    for run in runs:
        assert run["source_manifest_id"] == reference["source_manifest_id"]
        assert run["source_files_sha256"] == reference["source_files_sha256"]
    # Lower thresholds add edges, so their maximum matching cannot be smaller.
    for lower, higher in zip(runs, runs[1:]):
        for a, b in zip(lower["comparisons"], higher["comparisons"]):
            assert a["width"] == b["width"]
            assert a["matched_count"] >= b["matched_count"]
    rows = [dict(threshold=run["threshold"], **row)
            for run in runs for row in run["comparisons"]]
    with (output / "threshold_sweep_counts.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = dict(complete=True, thresholds=thresholds, rows=reference["rows"],
                   source_manifest_id=reference["source_manifest_id"],
                   source_files_sha256=reference["source_files_sha256"],
                   candidate_limited=True, optimal_on_retained_graph=True,
                   threshold_monotonicity_verified=True,
                   implementation_sha256=hashlib.sha256(_paper_path(__file__).read_bytes()).hexdigest(),
                   comparisons=rows,
                   run_summaries=[f"threshold_{t:g}/summary.json" for t in thresholds])
    (output / "threshold_sweep_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    plot_series(runs, output)
    values = ", ".join(f"{t:g}" for t in thresholds)
    caption = (
        "Maximum-cardinality feature matching between Gemini and Nemotron SAEs of the same width, "
        f"at signed activation Pearson thresholds {values}, on {reference['rows']:,} aligned examples. "
        "Left: matched-pair count. Right: count divided by the common SAE width, including dead and "
        "unmatched features in the denominator. Each curve uses the union of threshold-qualified "
        "retained neighbors from both directions. Counts are certified optimal on the retained "
        "candidate graphs and lower bounds on exhaustive matching counts."
    )
    (output / "caption.tex").write_text("\\caption{" + caption + "}\n")
    command = "python -m scripts.stability.paper_cross_model_cardinality --thresholds "
    command += " ".join(f"{t:g}" for t in thresholds)
    command += " --output " + str(output.relative_to(ROOT) if output.is_relative_to(ROOT) else output)
    (output / "README.md").write_text(
        "# Same-width cross-model activation matching across thresholds\n\n" + caption + "\n\n"
        "`same_width_matching.pdf`, `.png`, and `.svg` contain all threshold curves in the "
        "count/proportion layout of Figure 2(a). `threshold_sweep_counts.csv` and "
        "`threshold_sweep_summary.json` record the plotted values. Each `threshold_T/` directory "
        "contains its own graphs, matched feature IDs, minimum vertex cover certificates, input "
        "hashes, and methodological details. Lowering the threshold was verified never to decrease "
        "the matching count.\n\n"
        "Reproduce from the repository root: `" + command + "`.\n"
    )
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--threshold", type=float, default=.7)
    group.add_argument("--thresholds", type=float, nargs="+")
    args = parser.parse_args()
    if args.thresholds:
        output = args.output or ROOT / "full_experiments/results/cross_model_same_width_cardinality_pearson_thresholds"
        compute_sweep(args.source, output, args.thresholds)
    else:
        output = args.output or ROOT / f"full_experiments/results/cross_model_same_width_cardinality_pearson_{args.threshold:g}"
        compute(args.source, output, args.threshold)
