#!/usr/bin/env python3
"""Project broad a-prefix probe vectors onto the 131K a-prefix decoder span."""
from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(_paper_path(__file__).resolve().parents[2]))
from project_paths import get_path
from scripts.plot_style import apply_plot_style

WIDTH_TOPK = [(512, 32), (1024, 32), (2048, 32), (4096, 32), (8192, 64), (16384, 64), (32768, 64), (65536, 128), (131072, 128)]


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def state_dict(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(_paper_location(path), map_location="cpu", mmap=True, weights_only=False)
    return payload.get("model_state_dict", payload.get("state_dict", payload))


def vectors(state: dict[str, torch.Tensor], feature_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    ids = torch.as_tensor(feature_ids, dtype=torch.long)
    encoder = state["W_enc"][:, ids].T.float().numpy().copy()
    decoder = state["W_dec"][ids].float().numpy().copy()
    return encoder, decoder


def a_prefix_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected = [row for row in rows if row["category"] == "a" or (len(row["category"]) == 2 and row["category"].startswith("a"))]
    return sorted(selected, key=lambda row: (row["category"] != "a", row["category"]))


def project(target: np.ndarray, basis: np.ndarray):
    coefficients, _, rank, singular_values = np.linalg.lstsq(basis.T, target, rcond=None)
    reconstruction = coefficients @ basis
    residual = target - reconstruction
    target_norm = float(np.linalg.norm(target))
    reconstruction_norm = float(np.linalg.norm(reconstruction))
    residual_norm = float(np.linalg.norm(residual))
    relative_error = residual_norm / target_norm
    explained_fraction = max(0.0, min(1.0, 1.0 - relative_error ** 2))
    cosine = float(target @ reconstruction / (target_norm * reconstruction_norm)) if reconstruction_norm else 0.0
    return coefficients, reconstruction, {
        "target_norm": target_norm,
        "reconstruction_norm": reconstruction_norm,
        "residual_norm": residual_norm,
        "relative_error": relative_error,
        "explained_fraction": explained_fraction,
        "cosine_similarity": cosine,
        "basis_rank": int(rank),
        "largest_singular_value": float(singular_values[0]),
        "smallest_nonzero_singular_value": float(singular_values[rank - 1]),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows: list[dict[str, object]], out_dir: Path, model: str, basis_direction: str) -> None:
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    styles = {"decoder": ("#006D77", "o"), "encoder": ("#B5651D", "s")}
    for direction in ("decoder", "encoder"):
        subset = [row for row in rows if row["direction"] == direction]
        color, marker = styles[direction]
        ax.plot([row["width"] for row in subset], [row["explained_fraction"] for row in subset], color=color, marker=marker, linewidth=2.7, markersize=8, label=direction.title())
    ax.set_xscale("log", base=2)
    ax.set_xticks([width for width, _ in WIDTH_TOPK], [str(width) for width, _ in WIDTH_TOPK], rotation=35, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("SAE width")
    ax.set_ylabel("fraction of squared norm reconstructed")
    ax.set_title(f"Broad a-prefix reconstruction from 131K {basis_direction} probes ({model})")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "explained_fraction_by_width.png", dpi=220)
    fig.savefig(out_dir / "explained_fraction_by_width.pdf")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_root = args.probe_results_dir / "wordfreq"
    anchor_width, anchor_top_k = WIDTH_TOPK[-1]
    anchor_rows = a_prefix_rows(read_summary(summary_root / f"{args.model}_m{anchor_width}_k{anchor_top_k}" / "summary.tsv"))
    basis_categories = [row["category"] for row in anchor_rows]
    basis_features = [int(row["best_feature"]) for row in anchor_rows]
    anchor_state = state_dict(args.models_dir / f"{args.model}_m{anchor_width}_k{anchor_top_k}")
    basis_encoder, basis_decoder = vectors(anchor_state, basis_features)
    del anchor_state
    basis = basis_encoder if args.basis_direction == "encoder" else basis_decoder

    target_encoders = []
    target_decoders = []
    target_features = []
    result_rows = []
    coefficient_rows = []
    reconstructions = {"encoder": [], "decoder": []}
    for width, top_k in WIDTH_TOPK:
        summary = read_summary(summary_root / f"{args.model}_m{width}_k{top_k}" / "summary.tsv")
        parent = [row for row in summary if row["category"] == "a"]
        if len(parent) != 1:
            raise ValueError(f"Expected one broad a probe at width {width}, found {len(parent)}")
        feature = int(parent[0]["best_feature"])
        state = state_dict(args.models_dir / f"{args.model}_m{width}_k{top_k}")
        encoder, decoder = vectors(state, [feature])
        del state
        target_features.append(feature)
        target_encoders.append(encoder[0])
        target_decoders.append(decoder[0])
        for direction, target in (("encoder", encoder[0]), ("decoder", decoder[0])):
            coefficients, reconstruction, metrics = project(target.astype(np.float64), basis.astype(np.float64))
            reconstructions[direction].append(reconstruction.astype(np.float32))
            result_rows.append({"model": args.model, "width": width, "top_k": top_k, "category": "a", "feature": feature, "direction": direction, "basis_direction": args.basis_direction, "basis_width": anchor_width, "basis_categories": len(basis_categories), "basis_unique_features": len(set(basis_features)), **metrics})
            for category, basis_feature, coefficient in zip(basis_categories, basis_features, coefficients):
                coefficient_rows.append({"model": args.model, "target_width": width, "target_feature": feature, "target_direction": direction, "basis_category": category, "basis_feature": basis_feature, "coefficient": float(coefficient)})

    np.savez_compressed(_paper_location(args.out_dir / "basis_131072_vectors.npz"), categories=np.asarray(basis_categories), feature_ids=np.asarray(basis_features), encoder_vectors=basis_encoder, decoder_vectors=basis_decoder)
    np.savez_compressed(_paper_location(args.out_dir / "broad_a_target_vectors.npz"), widths=np.asarray([width for width, _ in WIDTH_TOPK]), feature_ids=np.asarray(target_features), encoder_vectors=np.stack(target_encoders), decoder_vectors=np.stack(target_decoders), encoder_reconstructions=np.stack(reconstructions["encoder"]), decoder_reconstructions=np.stack(reconstructions["decoder"]))
    write_csv(args.out_dir / "reconstruction_summary.csv", result_rows)
    write_csv(args.out_dir / "reconstruction_coefficients.csv", coefficient_rows)
    metadata = {"model": args.model, "probe_results_dir": str(args.probe_results_dir), "models_dir": str(args.models_dir), "basis_direction": args.basis_direction, "basis_width": anchor_width, "basis_categories": basis_categories, "basis_feature_ids": basis_features, "basis_unique_feature_ids": sorted(set(basis_features)), "targets": "single-letter a best probe at every standard width"}
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plot_results(result_rows, args.out_dir, args.model, args.basis_direction)
    print(f"Wrote results to {args.out_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    hierarchy_dir = _paper_path(get_path("hierarchy_data_dir"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("gemini", "nemotron"), default="gemini")
    parser.add_argument("--basis-direction", choices=("decoder", "encoder"), default="decoder")
    parser.add_argument("--probe-results-dir", type=Path, default=hierarchy_dir / "probe_results")
    parser.add_argument("--models-dir", type=Path, default=_paper_path(get_path("models_dir")))
    parser.add_argument("--out-dir", type=Path, default=_paper_path("full_experiments/results/a_prefix_linear_reconstruction/gemini"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
