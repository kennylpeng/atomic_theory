from project_paths import resource_path as _paper_path, resource_location as _paper_location
#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(_paper_path(__file__).resolve().parents[2]))
from project_paths import get_path
from scripts.plot_style import apply_plot_style, family_style

"""Find exclusive matches across dictionary sizes and plot all-larger counts."""

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import torch

from utils import load_dictionary


# SAE dictionaries are identified by width m and TopK value k.
SIZES = ((512, 32), (1024, 32), (2048, 32), (4096, 32), (8192, 64),
         (16384, 64), (32768, 64), (65536, 128), (131072, 128))
KINDS = ("sae", "kmeans", "pca")
# PCA width is capped by each model embedding dimension.
PCA_WIDTHS = {"gemini": (512, 1024, 2048, 3072),
              "nemotron": (512, 1024, 2048, 4096)}


def dictionary_path(models_dir: Path, model: str, kind: str, width: int) -> Path:
    """Translate a representation and width into its saved filename."""
    top_k = dict(SIZES).get(width)
    if kind == "sae":
        return models_dir / f"{model}_m{width}_k{top_k}"
    if kind == "kmeans":
        return models_dir / f"{model}_kmeans_pca256_m{width}"
    return models_dir / f"{model}_pca_components"


@torch.inference_mode()
def exclusive_matches(u, v, t1, t2, src_batch, dst_batch, device, *, absolute=False):
    """Return pairs above t1 with no other neighbor above t2.

    Inputs are unit directions. Absolute cosine applies to both thresholds.
    """
    if u.shape[1] != v.shape[1]:
        raise ValueError(f"Dictionary dimensions differ: {u.shape[1]} and {v.shape[1]}")

    # Count neighbors above t2 for every row and column.
    row_counts = torch.zeros(len(u), dtype=torch.int32, device=device)
    col_counts = torch.zeros(len(v), dtype=torch.int32, device=device)
    # Pairs above t1 are candidates until the t2 counts prove exclusivity.
    candidates = []
    candidate_above_t2 = []
    v = v.to(device)

    # Block the product so the full similarity matrix is never stored.
    for u0 in range(0, len(u), src_batch):
        u1 = min(u0 + src_batch, len(u))
        u_block = u[u0:u1].to(device)
        for v0 in range(0, len(v), dst_batch):
            v1 = min(v0 + dst_batch, len(v))
            similarities = u_block @ v[v0:v1].T
            similarities.clamp_(-1.0, 1.0)
            if absolute:
                similarities.abs_()
            # Count every potential competitor above the lower threshold.
            above_t2 = similarities > t2
            row_counts[u0:u1] += above_t2.sum(dim=1)
            col_counts[v0:v1] += above_t2.sum(dim=0)

            # Record strong matches using global, rather than block-local, indices.
            ij = (similarities > t1).nonzero()
            if len(ij):
                candidate_above_t2.append(above_t2[ij[:, 0], ij[:, 1]].cpu())
                ij[:, 0] += u0
                ij[:, 1] += v0
                candidates.append(ij.cpu())

    if not candidates:
        return [], []
    candidates = torch.cat(candidates)
    candidate_above_t2 = torch.cat(candidate_above_t2).to(device=device, dtype=torch.int32)
    # An exclusive match has no other above-t2 value in its row or column.
    keep = ((row_counts[candidates[:, 0].to(device)] - candidate_above_t2 == 0)
            & (col_counts[candidates[:, 1].to(device)]
               - candidate_above_t2 == 0)).cpu()
    matches = candidates[keep]
    return matches[:, 0].tolist(), matches[:, 1].tolist()


def widths_for(model, kind):
    """Return available widths in increasing order."""
    return PCA_WIDTHS[model] if kind == "pca" else tuple(width for width, _ in SIZES)


def compute_rows(args, device):
    """Compute and save direct matches for every pair of dictionary widths."""
    rows = []
    pairs_dir = args.out_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    models = ("gemini", "nemotron") if args.model == "both" else (args.model,)
    kinds = KINDS if args.representation == "both" else (args.representation,)

    for model in models:
        for kind in kinds:
            widths = widths_for(model, kind)
            # Compare every dictionary directly with every larger dictionary.
            for left_width, right_width in combinations(widths, 2):
                left_path = dictionary_path(args.models_dir, model, kind, left_width)
                right_path = dictionary_path(args.models_dir, model, kind, right_width)
                print(f"{model} {kind}: {left_width} -> {right_width}", flush=True)
                u = load_dictionary(
                    left_path,
                    kind,
                    left_width,
                    sae_direction=args.sae_direction,
                )
                v = load_dictionary(
                    right_path,
                    kind,
                    right_width,
                    sae_direction=args.sae_direction,
                )
                matched_u, matched_v = exclusive_matches(
                    u, v, args.t1, args.t2, args.src_batch, args.dst_batch, device,
                    absolute=kind == "pca",
                )
                row = {
                    "model": model, "representation": kind,
                    "left_width": left_width, "right_width": right_width,
                    "similarity": "absolute cosine" if kind == "pca" else "signed cosine",
                    "matches": len(matched_u), "matched_u": matched_u,
                    "matched_v": matched_v, "t1": args.t1, "t2": args.t2,
                    "sae_direction": args.sae_direction if kind == "sae" else None,
                    "left_path": str(left_path), "right_path": str(right_path),
                }
                rows.append(row)
                # JSON retains source indices needed for all-larger intersection.
                detail = pairs_dir / f"{model}_{kind}_m{left_width}_to_m{right_width}.json"
                detail.write_text(json.dumps(row, indent=2) + "\n")
    return rows


def all_larger_counts(rows):
    """Count source features with a direct match in every larger dictionary."""
    counts = []
    for model in ("gemini", "nemotron"):
        for kind in KINDS:
            group = [r for r in rows if r["model"] == model and r["representation"] == kind]
            if not group:
                continue
            left_widths = sorted({r["left_width"] for r in group})
            for left_width in left_widths:
                comparisons = [r for r in group if r["left_width"] == left_width]

                # matched_u indexes the same source dictionary in every direct comparison.
                # Keep only source indices that match in every larger dictionary.
                active = set(comparisons[0]["matched_u"])
                for row in comparisons[1:]:
                    active &= set(row["matched_u"])

                # Build a fresh summary; destination indices differ by dictionary.
                largest = max(comparisons, key=lambda r: r["right_width"])
                counts.append({
                    "model": model,
                    "representation": kind,
                    "left_width": left_width,
                    "largest_compared_width": largest["right_width"],
                    "matches": len(active),
                    "matched_u": sorted(active),
                    "t1": largest["t1"],
                    "t2": largest["t2"],
                    "left_path": largest["left_path"],
                    "right_path": largest["right_path"],
                })
    return counts


def write_counts(rows, path):
    """Write a compact count summary without feature-index lists."""
    fieldnames = ("model", "representation", "left_width", "largest_compared_width", "matches", "t1", "t2",
                  "left_path", "right_path")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fieldnames} for row in rows)


def plot(rows, path, y_key, ylabel, *, sae_direction="decoder"):
    """Plot one metric for both models and all representations."""
    labels = {
        "sae": f"SAE {sae_direction}",
        "kmeans": "KMeans centroids",
        "pca": "PCA directions",
    }
    # PCA occupies the first three of the shared SAE/KMeans positions.
    positions = {width: i for i, (width, _) in enumerate(SIZES[:-1])}
    apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.5), sharey=True)
    for ax, model in zip(axes, ("gemini", "nemotron")):
        for kind in KINDS:
            group = [r for r in rows if r["model"] == model and r["representation"] == kind]
            if not group:
                continue
            x = [positions[r["left_width"]] for r in group]
            ax.plot(
                x,
                [r[y_key] for r in group],
                label=labels[kind],
                **family_style(kind),
            )
        ax.set(title=model, xlabel="dictionary size")
        ax.set_xticks(range(len(positions)), [str(w) for w in positions], rotation=35, ha="right")
        ax.grid(axis="y", alpha=.25)
        ax.yaxis.set_minor_locator(MultipleLocator(1000))
        ax.grid(axis="y", which="minor", alpha=.15)
        ax.legend(frameon=False)
    axes[0].set_ylabel(ylabel)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=_paper_path(get_path("models_dir")))
    parser.add_argument("--out-dir", type=Path, default=_paper_path("full_experiments/matches"))
    parser.add_argument("--plot-path", type=Path, default=_paper_path("full_experiments/plots/all_larger_dictionary_matches.png"))
    parser.add_argument("--t1", type=float, default=.8)
    parser.add_argument("--t2", type=float, default=.5)
    parser.add_argument("--src-batch", type=int, default=512)
    parser.add_argument("--dst-batch", type=int, default=131072)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", choices=("gemini", "nemotron", "both"), default="both")
    parser.add_argument("--representation", choices=(*KINDS, "both"), default="both")
    parser.add_argument(
        "--sae-direction",
        choices=("decoder", "encoder"),
        default="decoder",
        help="Use W_dec rows or transposed W_enc columns for SAE matching.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Fall back to CPU for local runs on machines without CUDA.
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    print(f"Using {device}", flush=True)
    # Intersect direct matches from each width to all larger widths.
    rows = compute_rows(args, device)
    counts = all_larger_counts(rows)

    # Write the summary and produce the single all-larger plot.
    write_counts(counts, args.out_dir / "all_larger_dictionary_match_counts.csv")
    plot(
        counts,
        args.plot_path,
        "matches",
        "matches in all larger dictionaries",
        sae_direction=args.sae_direction,
    )


if __name__ == "__main__":
    main()
