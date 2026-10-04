"""Recompute the principal empirical results from the compact release bundle."""

import csv
import json
from pathlib import Path
import re
import numpy as np
from scipy.sparse import load_npz
from .bundles import verify, sha256, write_json
from .matching import assignment, persistent_matching, validate_witness, candidate_graph
from .hierarchy import (
    WIDTHS,
    LEVEL_SPECS,
    read_summary,
    compute_regrets,
    summarize_regrets,
    render_mean_f1_latex,
)


from .plotting import write_plots


def _csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def reproduce(bundle, output, candidates=None):
    """Verify bundles, recompute P1/P2/P3, compare paper values, and write outputs.

    This is the public orchestration entry point used by the CLI. Matching is
    re-solved from graphs; hierarchy statistics use the saved per-category fits.
    The returned report records precisely which computations were checked.
    """
    bundle, output = Path(bundle), Path(output)
    manifest = verify(bundle)
    if manifest["kind"] != "results":
        raise ValueError("Expected a results bundle")
    if output.exists():
        raise FileExistsError("Use a new output directory")
    if candidates:
        candidates = Path(candidates)
        if verify(candidates)["kind"] != "candidates":
            raise ValueError("Expected a candidates bundle")
    output.mkdir(parents=True)
    persistence, evidence, candidate_checks = reproduce_persistence(
        bundle, output, candidates
    )
    split_rows = reproduce_splits(bundle, output)
    hierarchy_rows = reproduce_hierarchy(bundle, output)
    write_plots(output, persistence, split_rows, hierarchy_rows)
    report = dict(
        status="passed",
        results_manifest_sha256=sha256(bundle / "manifest.json"),
        persistent_counts=32,
        split_graph_optima=60,
        directional_split_counts=120,
        hierarchy_cells=90,
        candidate_graphs_reconstructed=candidate_checks,
        hierarchy_tolerance=0.0000051,
        persistence_certificates=evidence,
        scope="Recomputed independent persistence intersections and pairwise graph optima and pooled saved-probe statistics; no training or embedding regeneration.",
    )
    if candidates:
        report["candidates_manifest_sha256"] = sha256(candidates / "manifest.json")
    write_json(output / "report.json", report)
    return report


def reproduce_persistence(bundle, output, candidates=None):
    """Solve P1 graphs and check saved witnesses and published counts (Fig. 2a).

    Called after bundle verification by reproduce(). When candidate arrays are
    supplied, rebuild every Pearson graph before solving it. Returns plot rows,
    optimality certificates, and the number of rebuilt candidate graphs.
    """
    reference = json.loads((bundle / "reference/persistence.json").read_text())[
        "comparisons"
    ]
    expected = {(r["model"], r["metric"], r["width"]): r for r in reference}
    if len(expected) != 32:
        raise ValueError("Expected 32 unique persistence results")
    persistence, evidence = [], []
    candidate_checks = 0
    for model in ("gemini", "nemotron"):
        for metric in ("decoder_cosine", "activation_pearson"):
            base = bundle / "persistence" / model / metric
            for width in WIDTHS[:-1]:
                targets = [w for w in WIDTHS if w > width]
                graphs = [load_npz(base / f"graph_{width}_to_{w}.npz") for w in targets]
                if any(g.shape != (width, w) for g, w in zip(graphs, targets)):
                    raise ValueError("Graph shape does not match dictionary widths")
                with np.load(
                    base / f"witness_{width}.npz", allow_pickle=False
                ) as saved:
                    validate_witness(
                        graphs, saved["source"], [saved[f"target_{w}"] for w in targets]
                    )
                    saved_source = saved["source"].copy()
                    saved_destinations = [saved[f"target_{w}"].copy() for w in targets]
                    saved_count = len(saved_source)
                if candidates and metric == "activation_pearson":
                    for target, graph in zip(targets, graphs):
                        with np.load(
                            candidates
                            / model
                            / f"{model}_{width}_to_{target}_top5.npz",
                            allow_pickle=False,
                        ) as f, np.load(
                            candidates
                            / model
                            / f"{model}_{target}_to_{width}_top5.npz",
                            allow_pickle=False,
                        ) as r:
                            for a, n, m in [(f, width, target), (r, target, width)]:
                                if (
                                    not np.array_equal(
                                        a["source_feature_ids"], np.arange(n)
                                    )
                                    or int(a["source_width"]) != n
                                    or int(a["target_width"]) != m
                                ):
                                    raise ValueError(
                                        "Candidate feature ordering mismatch"
                                    )
                            rebuilt = candidate_graph(
                                f["target_feature_ids"],
                                f["pearson"],
                                r["target_feature_ids"],
                                r["pearson"],
                            )
                        if (rebuilt != graph).nnz:
                            raise ValueError(
                                "Candidate reconstruction disagrees with saved graph"
                            )
                        candidate_checks += 1
                chosen, destinations, stats = persistent_matching(graphs)
                if not np.array_equal(chosen, saved_source) or any(not np.array_equal(a,b) for a,b in zip(destinations,saved_destinations)):
                    raise ValueError("Saved persistence does not match the fixed pairwise tie-breaking")
                ref = expected[(model, metric, width)]
                if (
                    len(chosen) != ref["persistent_count"]
                    or saved_count != len(chosen)
                    or abs(ref["proportion"] - len(chosen) / width) > 1e-12
                ):
                    raise ValueError(f"Persistence mismatch: {model}/{metric}/{width}")
                persistence.append(
                    dict(
                        model=model,
                        metric=metric,
                        width=width,
                        persistent_count=len(chosen),
                        proportion=len(chosen) / width,
                    )
                )
                evidence.append(dict(model=model, metric=metric, width=width, **stats))
    _csv(output / "persistence.csv", persistence)
    return persistence, evidence, candidate_checks


def reproduce_splits(bundle, output):
    """Re-solve 60 P2 graphs and check 120 directional proportions (Fig. 2b)."""
    with (bundle / "reference/splits.csv").open() as f:
        split_rows = list(csv.DictReader(f))
    if len(split_rows) != 120:
        raise ValueError("Expected 120 directional split comparisons")
    pair_counts = {}
    for pair in sorted((bundle / "splits").iterdir()):
        graph = load_npz(pair / "graph.npz")
        with np.load(pair / "assignment.npz", allow_pickle=False) as a:
            validate_witness([graph], a["source"], [a["destination"]])
            saved_count = len(a["source"])
        actual = int(np.count_nonzero(assignment(graph) >= 0))
        if saved_count != actual:
            raise ValueError(f"Nonoptimal split witness: {pair.name}")
        pair_counts[pair.name] = (actual, graph.shape)
    if len(pair_counts) != 60:
        raise ValueError("Expected 60 split graphs")
    for row in split_rows:
        model, rep, a, b = [
            row[k]
            for k in ("model", "representation", "source_split", "comparison_split")
        ]
        key = f"{model}_{rep}_{a}_to_{b}"
        reverse = key not in pair_counts
        if reverse:
            key = f"{model}_{rep}_{b}_to_{a}"
        count, shape = pair_counts[key]
        width = shape[1 if reverse else 0]
        if (
            count != int(row["matches"])
            or width != int(row["source_features"])
            or abs(count / width - float(row["proportion"])) > 1e-12
        ):
            raise ValueError(f"Split comparison mismatch: {key}")
    _csv(output / "split_matches.csv", split_rows)
    return split_rows


def reproduce_hierarchy(bundle, output):
    """Pool saved probe fits and check 90 paper table cells (Fig. 2c).

    Category-model trajectories are pooled before taking the mean. This stage
    does not refit probes or choose features using held-out labels.
    """
    rows = []
    for path in sorted((bundle / "hierarchy").glob("*/*/summary.tsv")):
        rows.extend(read_summary(path))
    summaries = summarize_regrets(compute_regrets(rows), tolerance=0.02)
    reference_tex = (bundle / "reference/hierarchy_mean_f1.tex").read_text()
    hierarchy_rows = []
    for spec in LEVEL_SPECS:
        lines = [
            line
            for line in reference_tex.splitlines()
            if line.startswith(f"{spec.domain} & {spec.label} &")
        ]
        if len(lines) != 1:
            raise ValueError(f"Missing reference row: {spec}")
        fields = lines[0].split(" & ")
        expected_n = int(fields[2])
        for width, text in zip(WIDTHS, fields[3:]):
            value = float(re.search(r"\d+\.\d+", text)[0])
            cell = summaries[(spec.level, width)]
            if cell.count != expected_n or abs(cell.mean_test_f1 - value) > 0.0000051:
                raise ValueError(f"Hierarchy table mismatch: {spec.level}/{width}")
            hierarchy_rows.append(
                dict(
                    dataset=spec.dataset,
                    level=spec.level,
                    width=width,
                    count=cell.count,
                    mean_test_f1=cell.mean_test_f1,
                    mean_regret=cell.mean_regret,
                )
            )
    if len(hierarchy_rows) != 90:
        raise ValueError("Expected 90 hierarchy cells")
    _csv(output / "hierarchy.csv", hierarchy_rows)
    (output / "hierarchy_mean_f1.tex").write_text(
        render_mean_f1_latex(
            summaries,
            caption="Mean held-out test F1 by hierarchy level and SAE width.",
            label="tab:hierarchy-mean-f1",
        )
    )
    return hierarchy_rows
