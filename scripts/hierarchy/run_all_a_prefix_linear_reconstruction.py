#!/usr/bin/env python3
"""Reconstruct all selected a-prefix probe vectors from a 131K probe-vector basis."""
from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import argparse
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(_paper_path(__file__).resolve().parents[2]))
from project_paths import get_path
from scripts.hierarchy.run_a_prefix_linear_reconstruction import (
    WIDTH_TOPK, a_prefix_rows, project, read_summary, state_dict, vectors, write_csv,
)


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_root = args.probe_results_dir / "wordfreq"
    anchor_width, anchor_top_k = WIDTH_TOPK[-1]
    anchor_rows = a_prefix_rows(read_summary(summary_root / f"{args.model}_m{anchor_width}_k{anchor_top_k}" / "summary.tsv"))
    if args.basis_min_test_f1 is not None:
        anchor_rows = [row for row in anchor_rows if float(row["test_f1"]) >= args.basis_min_test_f1]
    if not anchor_rows:
        raise ValueError("No 131K a-prefix probes satisfy the basis F1 threshold")
    basis_categories = [row["category"] for row in anchor_rows]
    basis_features = [int(row["best_feature"]) for row in anchor_rows]
    anchor_state = state_dict(args.models_dir / f"{args.model}_m{anchor_width}_k{anchor_top_k}")
    basis_encoder, basis_decoder = vectors(anchor_state, basis_features)
    del anchor_state
    basis = basis_encoder if args.basis_direction == "encoder" else basis_decoder
    basis64 = basis.astype(np.float64)

    result_rows = []
    coefficient_rows = []
    target_widths = []
    target_categories = []
    target_features = []
    target_directions = []
    target_vectors = []
    reconstructed_vectors = []

    for width, top_k in WIDTH_TOPK:
        selected = a_prefix_rows(read_summary(summary_root / f"{args.model}_m{width}_k{top_k}" / "summary.tsv"))
        by_category = {row["category"]: row for row in selected}
        missing = [category for category in basis_categories if category not in by_category]
        if missing:
            raise ValueError(f"Width {width} is missing categories: {missing}")
        features = [int(by_category[category]["best_feature"]) for category in basis_categories]
        state = state_dict(args.models_dir / f"{args.model}_m{width}_k{top_k}")
        encoders, decoders = vectors(state, features)
        del state
        for category, feature, encoder, decoder in zip(basis_categories, features, encoders, decoders):
            for direction, target in (("encoder", encoder), ("decoder", decoder)):
                coefficients, reconstruction, metrics = project(target.astype(np.float64), basis64)
                result_rows.append({"model": args.model, "width": width, "top_k": top_k, "category": category, "feature": feature, "direction": direction, "basis_direction": args.basis_direction, "basis_width": anchor_width, "basis_min_test_f1": args.basis_min_test_f1, "basis_categories": len(basis_categories), "basis_unique_features": len(set(basis_features)), **metrics})
                for basis_category, basis_feature, coefficient in zip(basis_categories, basis_features, coefficients):
                    coefficient_rows.append({"model": args.model, "target_width": width, "target_category": category, "target_feature": feature, "target_direction": direction, "basis_direction": args.basis_direction, "basis_category": basis_category, "basis_feature": basis_feature, "coefficient": float(coefficient)})
                target_widths.append(width)
                target_categories.append(category)
                target_features.append(feature)
                target_directions.append(direction)
                target_vectors.append(target)
                reconstructed_vectors.append(reconstruction.astype(np.float32))

    write_csv(args.out_dir / "reconstruction_summary.csv", result_rows)
    write_csv(args.out_dir / "reconstruction_coefficients.csv", coefficient_rows)
    np.savez_compressed(_paper_location(args.out_dir / "target_vectors_and_reconstructions.npz"), widths=np.asarray(target_widths), categories=np.asarray(target_categories), feature_ids=np.asarray(target_features), directions=np.asarray(target_directions), target_vectors=np.stack(target_vectors), reconstructed_vectors=np.stack(reconstructed_vectors))
    metadata = {"model": args.model, "basis_direction": args.basis_direction, "basis_width": anchor_width, "basis_min_test_f1": args.basis_min_test_f1, "basis_categories": basis_categories, "basis_feature_ids": basis_features, "basis_unique_feature_ids": sorted(set(basis_features)), "targets": "131K-basis categories at every standard width", "probe_results_dir": str(args.probe_results_dir), "models_dir": str(args.models_dir)}
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote results to {args.out_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    hierarchy_dir = _paper_path(get_path("hierarchy_data_dir"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("gemini", "nemotron"), default="gemini")
    parser.add_argument("--basis-direction", choices=("decoder", "encoder"), required=True)
    parser.add_argument("--basis-min-test-f1", type=float)
    parser.add_argument("--probe-results-dir", type=Path, default=hierarchy_dir / "probe_results")
    parser.add_argument("--models-dir", type=Path, default=_paper_path(get_path("models_dir")))
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
