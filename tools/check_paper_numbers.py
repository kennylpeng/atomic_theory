#!/usr/bin/env python3
"""Audit paper scalars against saved numerical outputs, without training.

This complements graph/probe recomputation; it does not certify input provenance.
The output records input hashes and checked quantities; only numerical result files are read.
"""
from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics


def audit(root):
    sources, checks = {}, []
    def contents(name):
        data = (root / name).read_bytes()
        sources[name] = hashlib.sha256(data).hexdigest()
        return data.decode()
    def rows(name):
        return list(csv.DictReader(contents(name).splitlines()))
    def document(name):
        return json.loads(contents(name))
    def check(label, actual, expected, tolerance=0):
        ok = abs(actual - expected) <= tolerance if isinstance(actual, (float, int)) else actual == expected
        checks.append(dict(quantity=label, actual=actual, expected=expected, tolerance=tolerance, passed=ok))
    base = "full_experiments/results/"
    persistence = rows(base + "persistent_stability_pairwise_0.7/counts.csv")
    for family, metric, percent in [("gemini", "cosine",72.6),("nemotron","cosine",69.4),
                                     ("gemini","pearson",59.7),("nemotron","pearson",66.4)]:
        row = next(r for r in persistence if r["model"] == family and r["metric"] == metric and int(r["width"]) == 4096)
        check(f"P1 {family} {metric}, width 4096 (%)", round(int(row['persistent_count']) / 4096 * 100, 1), percent)
    ranges = {}
    for mode in ["maximum_cardinality", "exclusive"]:
        table = rows(base + f"split_matches/paper_sae_pca_0.7/{mode}_counts.csv")
        check(f"{mode} directed SAE/PCA heatmap cells", len(table), 80)
        for rep in ["sae", "pca"]:
            values = [100 * int(r["matches"]) / int(r["source_features"]) for r in table if r["representation"] == rep]
            ranges[f"{mode}_{rep}"] = dict(min=min(values), max=max(values), median=statistics.median(values))
    table = rows(base + "split_sae_activation_sketch512_candidates32_top5/counts.csv")
    for family, expected in [("gemini", (22.0,48.9)),("nemotron",(29.3,60.7))]:
        values = [100 * int(r["matches"]) / 16384 for r in table if r["family"] == family]
        check(f"P2 {family} activation min (%)",round(min(values),1),expected[0])
        check(f"P2 {family} activation max (%)",round(max(values),1),expected[1])
    effects = rows(base + "sparsity_seed_controls/prevalence_median/width_effects.csv")
    for family, expected in [("gemini",[.814,.801,.822]),("nemotron",[.818,.905,.917])]:
        for width, value in zip([16384,32768,65536], expected):
            row = next(r for r in effects if r["family"] == family and int(r["width"]) == width and
                       r["threshold"] == "median_positive" and r["membership"] == "witness" and r["population"] == "all")
            check(f"P1 {family} width {width} prevalence AUC", round(float(row["auc"]),3), value)
    effects = rows(base + "sparsity_seed_controls/prevalence_median/cross_distribution_effects.csv")
    for family, population, expected in [("gemini","all",(.864,.923)),("nemotron","all",(.839,.927)),
                                          ("gemini","positive_both",(.844,.905)),("nemotron","positive_both",(.828,.921))]:
        values = [float(r["auc"]) for r in effects if r["family"]==family and r["population"]==population and
                  r["threshold"]=="median_positive" and r["membership"]=="witness" and
                  r["prevalence"]=="minimum_both" and r["disjoint_pair"]=="True"]
        check(f"P2 {family} {population} AUC min",round(min(values),3),expected[0])
        check(f"P2 {family} {population} AUC max",round(max(values),3),expected[1])
    synthetic = rows("experiments/recovery_principle/scaling/billion/results/persistence_1024M_t0.8/counts.csv")
    check("Synthetic persistence source/metric/seed cells",len(synthetic),780)
    for kind, expected in [("flat",(93.6,96.1)),("hierarchical",(91.9,99.1))]:
        groups = defaultdict(list)
        for r in synthetic:
            if r["kind"]==kind and float(r["alpha"])==1.6 and int(r["k"])==8 and r["metric"]=="activation_pearson":
                groups[int(r["width"])].append(int(r["persistent_count"])/int(r["width"])*100)
        values = [statistics.mean(v) for v in groups.values()]
        check(f"Synthetic {kind} activation mean min (%)",round(min(values),1),expected[0])
        check(f"Synthetic {kind} activation mean max (%)",round(max(values),1),expected[1])
    recovery = rows("experiments/recovery_principle/scaling/billion/results/checkpoints.csv")
    final = [r for r in recovery if int(r["examples"])==1024000000]
    check("Synthetic final trajectories",len(final),468)
    check("Synthetic evaluated milestones",len(recovery),5148)
    cosine = document(base + "a_prefix_probe_decoder_cosines/summary.json")
    for r, expected in zip(cosine["results"],[.1438,.0084]):
        check(f"Prefix cosine median at {r['width']}",round(r["median_cosine"],4),expected)
        check(f"Prefix unordered pairs at {r['width']}",r["unique_pairs"],231)
    molecule = document(base + "top_feature_examples/gemini_target_3290_expanded_17atom_sse_0p05/family_spec.json")
    check("OMP terms",molecule["atom_count"],17)
    fit = molecule["families"][0]
    check("OMP squared error",round(fit["squared_l2_error"],6),.049297)
    check("OMP reconstruction cosine",round(fit["reconstruction_cosine"],6),.975040)
    for i, value in enumerate([.6763,.1955,.1665,.1615,.0593]):
        check(f"OMP displayed coefficient {i+1}",round(fit["coefficients"][i],4),value)
    return dict(schema_version=2, checks=checks, all_checks_passed=all(c["passed"] for c in checks),
                split_percentage_ranges=ranges, input_sha256=sources,
                scope="Scalar consistency with saved results; graph/probe/checkpoint recomputation is reported separately.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_paper_path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(f"{len(report['checks'])} scalar checks; passed={report['all_checks_passed']}")
    if not report["all_checks_passed"]:
        for check in report["checks"]:
            if not check["passed"]:
                print(check)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
