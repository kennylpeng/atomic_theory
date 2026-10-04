"""Separate threshold sweeps for Figure 2(a) and activation persistence."""
from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz

from scripts.stability.paper_pearson_persistent_stability import threshold_graph
from scripts.stability.persistent_matching import persistent_matching, validate_witness

ROOT = _paper_path(__file__).resolve().parents[2]
RESULTS = ROOT / "full_experiments/results"
OUTPUT = RESULTS / "persistent_stability_threshold_sweep"
THRESHOLDS = (.5, .6, .7, .8)
FAMILIES = ("gemini", "nemotron")
WIDTHS = (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def original_dir(metric, family):
    if metric == "cosine":
        return RESULTS / "persistent_stability_signed_0.7" / family
    prefix = "" if family == "gemini" else "nemotron_"
    return RESULTS / f"{prefix}persistent_stability_pearson_0.7"


def initialize():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "original_graphs_sha256.json"
    if not path.exists():
        folders = [original_dir(metric, family) for metric in ("cosine", "pearson") for family in FAMILIES]
        write_json(path, {str(p.relative_to(ROOT)): digest(p) for folder in folders for p in sorted(folder.glob("graph_*.npz"))})
    return verify_originals()


def verify_originals():
    path = OUTPUT / "original_graphs_sha256.json"
    if not path.exists():
        return initialize()
    originals = json.loads(path.read_text())
    for name, expected in originals.items():
        assert digest(ROOT / name) == expected, f"Original graph changed: {name}"
    return len(originals)


def signed_score_graph(source, target, device="cuda", batch=512):
    """Exhaustively retain signed float32 decoder cosines at least 0.5."""
    import torch
    with torch.inference_mode():
        target = target.to(device)
        indices, values, counts = [], [], []
        for start in range(0, len(source), batch):
            scores = (source[start:start + batch].to(device) @ target.T).clamp_(-1, 1)
            mask = scores >= np.float32(min(THRESHOLDS))
            r, c = mask.nonzero(as_tuple=True)
            indices.append(c.cpu().numpy())
            values.append(scores[r, c].cpu().numpy())
            counts.append(mask.sum(dim=1).cpu().numpy())
    indptr = np.r_[0, np.cumsum(np.concatenate(counts), dtype=np.int64)]
    return csr_matrix((np.concatenate(values), np.concatenate(indices), indptr),
                      shape=(len(source), len(target)))


def filter_scores(scores, threshold):
    graph = scores.copy()
    graph.data = graph.data >= np.float32(threshold)
    graph.eliminate_zeros()
    return graph.astype(bool)


def prepare_inputs(metric, family, output):
    source = original_dir(metric, family)
    if metric == "cosine":
        import torch
        from scripts.stability.utils import load_dictionary
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        original = json.loads((source / "manifest.json").read_text())
        for item in original["source_files"]:
            stat = _paper_path(item["path"]).stat()
            assert stat.st_size == item["size"] and stat.st_mtime_ns == item["mtime_ns"]
        vectors = [load_dictionary(_paper_path(item["path"]), "sae") for item in original["source_files"]]
        assert [len(v) for v in vectors] == list(WIDTHS)
        metadata = dict(metric="signed decoder cosine", exhaustive=True,
                        source_files=original["source_files"], float32=True, tf32=False,
                        threshold_07="Original exhaustive signed graphs, preserving threshold decisions")
        extra = vectors
    else:
        neighbors = RESULTS / f"{family}_cross_width_sketch512_candidates32_top5"
        manifest = json.loads((neighbors / "manifest.json").read_text())
        summary = json.loads((neighbors / "summary.json").read_text())
        assert summary["complete"] and summary["manifest_id"] == manifest["id"]
        assert [m["width"] for m in manifest["models"]] == list(WIDTHS)
        metadata = dict(metric="signed continuous post-TopK activation Pearson", exhaustive=False,
                        source_directory=str(neighbors), source_manifest_id=manifest["id"],
                        rows=manifest["rows"], sketch_dim=manifest["sketch_dim"],
                        candidate_k=manifest["candidate_k"], save_k=manifest["save_k"],
                        interpretation="Exact on retained graphs; no guaranteed bound on exhaustive independent intersections")
        extra = (neighbors, manifest)
    metadata.update(family=family, thresholds=THRESHOLDS, widths=WIDTHS,
                    objective="Intersection of independent full-source pairwise maximum matchings into every larger SAE",
                    input_files_sha256={}, implementation_sha256=digest(_paper_path(__file__)),
                    solver_sha256=digest(ROOT / "scripts/stability/persistent_matching.py"))
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        for key in ("family", "metric", "implementation_sha256", "solver_sha256"):
            assert prior[key] == metadata[key], f"Changed cached run: {key}"
    write_json(manifest_path, metadata)
    return metadata, extra


def compute(metric, family, device):
    verify_originals()
    output = OUTPUT / metric / family
    output.mkdir(parents=True, exist_ok=True)
    metadata, extra = prepare_inputs(metric, family, output)
    previous = {r["width"]: r["persistent_count"] for r in
                json.loads((original_dir(metric, family) / "summary.json").read_text())}
    results = []
    for i, width in enumerate(WIDTHS[:-1]):
        targets = WIDTHS[i + 1:]
        graphs = {threshold: [] for threshold in THRESHOLDS}
        for j, target in enumerate(targets, start=i + 1):
            old_path = original_dir(metric, family) / f"graph_{width}_to_{target}.npz"
            metadata["input_files_sha256"][str(old_path)] = digest(old_path)
            old = load_npz(_paper_location(old_path)).astype(bool)
            if metric == "cosine":
                scores_path = output / f"scores_{width}_to_{target}.npz"
                if scores_path.exists():
                    scores = load_npz(_paper_location(scores_path))
                else:
                    scores = signed_score_graph(extra[i], extra[j], device=device)
                    save_npz(_paper_location(scores_path), scores)
                pair = {t: (old if t == .7 else filter_scores(scores, t)) for t in THRESHOLDS}
            else:
                neighbors, manifest = extra
                arrays = []
                for sw, tw in ((width, target), (target, width)):
                    path = neighbors / f"{family}_{sw}_to_{tw}_top5.npz"
                    metadata["input_files_sha256"][str(path)] = digest(path)
                    with np.load(_paper_location(path)) as data:
                        assert str(data["manifest_id"]) == manifest["id"]
                        assert int(data["rows"]) == manifest["rows"]
                        assert int(data["source_width"]) == sw and int(data["target_width"]) == tw
                        np.testing.assert_array_equal(data["source_feature_ids"], np.arange(sw))
                        arrays.append((data["target_feature_ids"], data["pearson"]))
                pair = {t: threshold_graph(*arrays[0], *arrays[1], threshold=t)[0] for t in THRESHOLDS}
                assert (pair[.7] != old).nnz == 0
            for lo, hi in zip(THRESHOLDS, THRESHOLDS[1:]):
                assert pair[hi].multiply(pair[lo]).nnz == pair[hi].nnz
            for threshold in THRESHOLDS:
                folder = output / f"threshold_{threshold:g}"
                folder.mkdir(exist_ok=True)
                save_npz(_paper_location(folder / f"graph_{width}_to_{target}.npz"), pair[threshold])
                graphs[threshold].append(pair[threshold])
            print(f"{metric} {family}: graphs {width} -> {target}", flush=True)
        for threshold in reversed(THRESHOLDS):
            started = time.monotonic()
            selected, destinations, stats = persistent_matching(graphs[threshold])
            validate_witness(graphs[threshold], selected, destinations)
            if threshold == .7:
                assert len(selected) == previous[width]
            folder = output / f"threshold_{threshold:g}"
            np.savez_compressed(_paper_location(folder / f"witness_{width}.npz"), source=selected,
                                **{f"target_{w}": d for w, d in zip(targets, destinations)})
            row = dict(model=family, metric=metric, width=width, threshold=threshold,
                       persistent_count=len(selected), proportion=len(selected) / width,
                       target_widths=targets, elapsed_seconds=time.monotonic() - started, **stats)
            write_json(folder / f"result_{width}.json", row)
            results.append(row)
            print(json.dumps(row), flush=True)
        counts = {r["threshold"]: r["persistent_count"] for r in results if r["width"] == width}
        # Nested graphs need not yield monotone source-set intersections.
        write_json(output / "manifest.json", metadata)
    results.sort(key=lambda r: (r["threshold"], r["width"]))
    write_json(output / "summary.json", dict(complete=True, original_07_counts_verified=True,
                                             threshold_monotonicity_required=False, comparisons=results))
    verify_originals()


def plot(rows, metric):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, PercentFormatter
    from scripts.stability.stability_plotting import apply_stability_grid, apply_stability_style
    apply_stability_style()
    colors = {0.5: "#009E73", 0.6: "#E69F00", 0.7: "#0072B2", 0.8: "#CC79A7"}
    styles = {"gemini": dict(linestyle="-", marker="o"), "nemotron": dict(linestyle="--", marker="s")}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    fig.subplots_adjust(top=.77, bottom=.15, left=.10, right=.98, wspace=.28)
    for ax, key in zip(axes, ("persistent_count", "proportion")):
        for family in FAMILIES:
            for threshold in THRESHOLDS:
                group = sorted((r for r in rows if r["model"] == family and r["threshold"] == threshold),
                               key=lambda r: r["width"])
                assert [r["width"] for r in group] == list(WIDTHS[:-1])
                ax.plot(WIDTHS[:-1], [r[key] for r in group], color=colors[threshold],
                        linewidth=2.4, markersize=5.5, **styles[family])
        ax.set_xscale("log", base=2)
        ax.set_xticks(WIDTHS[:-1], ["512", "1K", "2K", "4K", "8K", "16K", "32K", "64K"])
        ax.tick_params(axis="x", labelsize=15)
        ax.set_xlabel("SAE width")
        ax.spines[["top", "right"]].set_visible(False)
        apply_stability_grid(ax, proportion=key == "proportion")
        if key == "persistent_count":
            ax.set_ylim(bottom=0)
            ax.set_ylabel("Persistent feature count")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:,.0f}"))
        else:
            ax.set_ylim(0, 1)
            ax.set_ylabel("Persistent feature proportion")
            ax.yaxis.set_major_formatter(PercentFormatter(1))
    metric_label = "Signed decoder cosine" if metric == "cosine" else "Activation Pearson"
    fig.suptitle(f"Persistence across SAE widths · {metric_label}", y=.99, fontsize=20)
    handles = [Line2D([], [], color=colors[t], linewidth=2.5, label=f"t = {t:g}") for t in THRESHOLDS]
    handles.extend(Line2D([], [], color="#444444", linewidth=2.5, label=f.capitalize(), **styles[f]) for f in FAMILIES)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .94), ncol=6,
               fontsize=14, frameon=False, handlelength=2.4, columnspacing=1.3)
    stem = "figure_2a_cosine_thresholds" if metric == "cosine" else "activation_pearson_thresholds"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUTPUT / f"{stem}.{ext}", dpi=220)
    plt.close(fig)


def package():
    untouched = verify_originals()
    all_rows = []
    for metric in ("cosine", "pearson"):
        rows = []
        for family in FAMILIES:
            data = json.loads((OUTPUT / metric / family / "summary.json").read_text())
            assert data["complete"] and data["original_07_counts_verified"]
            rows.extend(data["comparisons"])
        assert len(rows) == 64
        plot(rows, metric)
        all_rows.extend(rows)
    fields = ["metric", "model", "threshold", "width", "persistent_count", "proportion"]
    with (OUTPUT / "counts.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    write_json(OUTPUT / "summary.json", dict(complete=True, thresholds=THRESHOLDS,
               original_files_verified=untouched, original_07_counts_verified=True,
               threshold_monotonicity_required=False, comparisons=all_rows))
    (OUTPUT / "README.md").write_text(
        "# Persistent-stability threshold sweeps\n\n"
        "Threshold sweeps for Figure 2(a) and its activation counterpart. "
        "Each curve intersects the source sets of independently selected maximum matchings "
        "into every larger trained SAE, independently for each model family and threshold. "
        "Thresholds are inclusive 0.5, 0.6, 0.7, and 0.8. The largest source width is omitted because "
        "it has no larger target. All widths remain in proportion denominators.\n\n"
        "`figure_2a_cosine_thresholds.pdf` uses exhaustive signed decoder cosines. Thresholds other than 0.7 "
        "are computed in float32 with TF32 disabled; 0.7 uses the saved exhaustive signed graphs "
        "to preserve their threshold decisions. `activation_pearson_thresholds.pdf` uses signed "
        "continuous activation Pearson and the union of saved top-five neighbors in both directions. "
        "The Pearson input used 512-dimensional sketches and 32 candidates per source feature; "
        "intersections use the retained graphs and have no guaranteed bound on exhaustive Pearson intersections. "
        "Pearson corpora are 89,827,558 examples for Gemini and 89,227,558 for Nemotron.\n\n"
        "Both figures use threshold colors and model-specific line styles. PNG and SVG copies are "
        "provided. `counts.csv` contains all 128 plotted points. Per-model directories retain input "
        "provenance, graphs, witnesses, and pairwise maximum certificates. Counts at threshold 0.7 are checked against the reference; "
        "threshold changes can alter matching choices and intersection counts. Reference threshold graphs "
        "are protected by `original_graphs_sha256.json`.\n\n"
        "Reproduce using `python -m scripts.stability.paper_persistent_threshold_sweep init`, "
        "then `compute --metric cosine|pearson --family gemini|nemotron` for each combination "
        "(GPU required for the default cosine computation), followed by `package`.\n"
    )
    print(f"Packaged 128 points; {untouched} original files unchanged.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "compute", "package"))
    parser.add_argument("--metric", choices=("cosine", "pearson"))
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.action == "init":
        initialize()
    elif args.action == "package":
        package()
    else:
        if args.metric is None or args.family is None:
            parser.error("compute requires --metric and --family")
        compute(args.metric, args.family, args.device)
