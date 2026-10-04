#!/usr/bin/env python3
"""Reconstruct a smaller SAE decoder from a larger SAE decoder with batched OMP.

The rows of the large decoder are treated as atoms.  For every row of the
smaller decoder, orthogonal matching pursuit (OMP) adds one atom at a time,
refits all coefficients by least squares, and stops at the first support whose
absolute L2 residual is at most the requested tolerance.

OMP is a greedy sparse-approximation algorithm.  The first successful support
is minimal along its greedy path, but is not a proof of the globally sparsest
support (that cardinality-constrained problem is NP-hard in general).
"""

from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import argparse
import csv
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(_paper_path(__file__).resolve().parents[1]))
from project_paths import get_path


@dataclass
class OMPResult:
    """Dense, padded representation of a batch of sparse decompositions."""

    atom_indices: torch.Tensor
    normalized_coefficients: torch.Tensor
    num_atoms: torch.Tensor
    l2_error: torch.Tensor
    l2_error_by_count: torch.Tensor


def load_decoder(path: Path) -> torch.Tensor:
    """Memory-map a checkpoint when possible and return its decoder on CPU."""
    try:
        checkpoint = torch.load(_paper_location(path), map_location="cpu", mmap=True, weights_only=False)
    except TypeError:  # Compatibility with older PyTorch versions.
        checkpoint = torch.load(_paper_location(path), map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    if "W_dec" not in state:
        raise KeyError(f"Checkpoint {path} has no W_dec tensor")
    decoder = state["W_dec"].detach()
    if decoder.ndim != 2:
        raise ValueError(f"Expected a decoder matrix in {path}, got {tuple(decoder.shape)}")
    if not decoder.dtype.is_floating_point:
        raise TypeError(f"Expected floating-point W_dec in {path}, got {decoder.dtype}")
    return decoder


def _least_squares_residual(
    selected_atoms: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve batched least squares accurately using small float64 Gram systems."""
    atoms64 = selected_atoms.to(dtype=torch.float64)
    targets64 = targets.to(dtype=torch.float64)
    gram = torch.bmm(atoms64, atoms64.transpose(1, 2))
    rhs = torch.bmm(atoms64, targets64.unsqueeze(2))

    # OMP normally produces a full-rank support, in which case Cholesky is both
    # fast and accurate.  A pseudoinverse handles duplicate/degenerate atoms.
    cholesky, info = torch.linalg.cholesky_ex(gram)
    coefficients = torch.empty_like(rhs)
    good = info.eq(0)
    if bool(good.any()):
        coefficients[good] = torch.cholesky_solve(rhs[good], cholesky[good])
    bad = ~good
    if bool(bad.any()):
        coefficients[bad] = torch.bmm(
            torch.linalg.pinv(gram[bad], hermitian=True), rhs[bad]
        )

    reconstruction = torch.bmm(
        coefficients.transpose(1, 2), atoms64
    ).squeeze(1)
    residual = targets64 - reconstruction
    return coefficients.squeeze(2), residual


@torch.inference_mode()
def orthogonal_matching_pursuit(
    atoms: torch.Tensor,
    targets: torch.Tensor,
    *,
    tolerance: float,
    max_atoms: int,
) -> OMPResult:
    """Run simultaneous greedy OMP for a target batch.

    ``atoms`` must have unit-norm rows.  ``targets`` are intentionally left at
    their stored scale, so the stopping criterion is absolute L2 error on the
    original target decoder rows.
    """
    if atoms.ndim != 2 or targets.ndim != 2:
        raise ValueError("atoms and targets must both be matrices")
    if atoms.shape[1] != targets.shape[1]:
        raise ValueError(
            f"Dimension mismatch: atoms {tuple(atoms.shape)}, targets {tuple(targets.shape)}"
        )
    if max_atoms < 1 or max_atoms > atoms.shape[0]:
        raise ValueError("max_atoms must be between 1 and the number of atoms")
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")

    device = atoms.device
    targets = targets.to(device=device, dtype=torch.float32)
    batch_size = targets.shape[0]
    atom_indices = torch.full(
        (batch_size, max_atoms), -1, dtype=torch.int64, device=device
    )
    coefficients = torch.zeros(
        (batch_size, max_atoms), dtype=torch.float64, device=device
    )
    num_atoms = torch.zeros(batch_size, dtype=torch.int64, device=device)
    residual = targets.clone()
    l2_error = torch.linalg.vector_norm(residual.to(torch.float64), dim=1)
    error_history = torch.full(
        (batch_size, max_atoms + 1),
        torch.nan,
        dtype=torch.float64,
        device=device,
    )
    error_history[:, 0] = l2_error
    unfinished = l2_error > tolerance

    for step in range(max_atoms):
        active = torch.nonzero(unfinished, as_tuple=False).squeeze(1)
        if active.numel() == 0:
            break

        # The expensive operation is a single GEMM per OMP round.  Scores are
        # discarded immediately after the best atom for each residual is found.
        scores = torch.mm(residual[active], atoms.T).abs_()
        if step:
            scores.scatter_(1, atom_indices[active, :step], -torch.inf)
        best_atoms = torch.argmax(scores, dim=1)
        del scores
        atom_indices[active, step] = best_atoms

        support = atom_indices[active, : step + 1]
        selected = atoms[support]
        fitted_coefficients, fitted_residual = _least_squares_residual(
            selected, targets[active]
        )
        coefficients[active, : step + 1] = fitted_coefficients
        residual[active] = fitted_residual.to(torch.float32)
        active_errors = torch.linalg.vector_norm(fitted_residual, dim=1)
        l2_error[active] = active_errors
        error_history[active, step + 1] = active_errors
        num_atoms[active] = step + 1
        unfinished[active] = active_errors > tolerance

    return OMPResult(
        atom_indices=atom_indices.cpu(),
        normalized_coefficients=coefficients.cpu(),
        num_atoms=num_atoms.cpu(),
        l2_error=l2_error.cpu(),
        l2_error_by_count=error_history.cpu(),
    )


@torch.inference_mode()
def refit_and_verify_raw(
    source_decoder: torch.Tensor,
    target_decoder: torch.Tensor,
    atom_indices: np.ndarray,
    num_atoms: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Refit selected supports against raw W_dec rows and recompute L2 errors."""
    target_count, max_atoms = atom_indices.shape
    coefficients = np.zeros((target_count, max_atoms), dtype=np.float64)
    l2_error = np.full(target_count, np.nan, dtype=np.float64)

    # Grouping targets by support size keeps each least-squares batch rectangular.
    for count in sorted(int(value) for value in np.unique(num_atoms)):
        target_ids = np.flatnonzero(num_atoms == count)
        if count == 0:
            raw_targets = target_decoder[target_ids].double()
            l2_error[target_ids] = torch.linalg.vector_norm(
                raw_targets, dim=1
            ).numpy()
            continue
        for offset in range(0, target_ids.size, batch_size):
            ids = target_ids[offset : offset + batch_size]
            support_np = atom_indices[ids, :count]
            if np.any(support_np < 0):
                raise ValueError("A selected support contains a negative atom index")
            if any(np.unique(row).size != count for row in support_np):
                raise ValueError("A selected support contains a duplicate atom")
            support = torch.from_numpy(support_np)
            selected_raw = source_decoder[support].to(
                device=device, dtype=torch.float64
            )
            raw_targets = target_decoder[torch.from_numpy(ids)].to(
                device=device, dtype=torch.float64
            )
            fitted, residual = _least_squares_residual(selected_raw, raw_targets)
            coefficients[ids, :count] = fitted.cpu().numpy()
            l2_error[ids] = torch.linalg.vector_norm(residual, dim=1).cpu().numpy()

    if not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(l2_error)):
        raise ValueError("Non-finite values produced during raw reconstruction verification")
    return coefficients, l2_error


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _json_float_list(values: np.ndarray) -> str:
    return json.dumps([float(value) for value in values], separators=(",", ":"))


def _json_int_list(values: np.ndarray) -> str:
    return json.dumps([int(value) for value in values], separators=(",", ":"))


def write_outputs(
    *,
    out_dir: Path,
    args: argparse.Namespace,
    source_path: Path,
    target_path: Path,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
    source_norms: np.ndarray,
    target_norms: np.ndarray,
    atom_indices: np.ndarray,
    raw_coefficients: np.ndarray,
    normalized_coefficients: np.ndarray,
    num_atoms: np.ndarray,
    l2_error: np.ndarray,
    l2_error_by_count: np.ndarray,
    elapsed_seconds: float,
) -> None:
    """Write machine-readable decompositions and per-direction reports."""
    out_dir.mkdir(parents=True, exist_ok=True)
    within_tolerance = l2_error <= args.tolerance

    np.savez_compressed(
        _paper_location(out_dir / "decompositions.npz"),
        atom_indices=atom_indices.astype(np.int32, copy=False),
        coefficients=raw_coefficients,
        normalized_atom_coefficients=normalized_coefficients,
        num_atoms=num_atoms.astype(np.int16, copy=False),
        l2_error=l2_error,
        l2_error_by_count=l2_error_by_count,
        within_tolerance=within_tolerance,
        target_norms=target_norms,
        source_atom_norms=source_norms,
    )

    summary_rows: list[dict[str, Any]] = []
    atom_rows: list[dict[str, Any]] = []
    for target_id in range(target_shape[0]):
        count = int(num_atoms[target_id])
        indices = atom_indices[target_id, :count]
        coefficients = raw_coefficients[target_id, :count]
        normalized = normalized_coefficients[target_id, :count]
        success = bool(within_tolerance[target_id])
        summary_rows.append(
            {
                "target_feature": target_id,
                "atoms_required": count if success else f">{args.max_atoms}",
                "num_atoms_selected": count,
                "l2_error": f"{l2_error[target_id]:.17g}",
                "within_tolerance": success,
                "target_norm": f"{target_norms[target_id]:.17g}",
                "relative_l2_error": f"{l2_error[target_id] / target_norms[target_id]:.17g}",
                "atom_indices": _json_int_list(indices),
                "coefficients": _json_float_list(coefficients),
            }
        )
        for order, (atom_id, coefficient, normalized_coefficient) in enumerate(
            zip(indices, coefficients, normalized), start=1
        ):
            atom_rows.append(
                {
                    "target_feature": target_id,
                    "atom_order": order,
                    "atom_feature": int(atom_id),
                    "coefficient": f"{coefficient:.17g}",
                    "normalized_atom_coefficient": f"{normalized_coefficient:.17g}",
                    "l2_error_after_atom": f"{l2_error_by_count[target_id, order]:.17g}",
                }
            )

    _write_csv(
        out_dir / "atom_counts.csv",
        [
            "target_feature",
            "atoms_required",
            "num_atoms_selected",
            "l2_error",
            "within_tolerance",
            "target_norm",
            "relative_l2_error",
            "atom_indices",
            "coefficients",
        ],
        summary_rows,
    )
    _write_csv(
        out_dir / "atoms_used.csv",
        [
            "target_feature",
            "atom_order",
            "atom_feature",
            "coefficient",
            "normalized_atom_coefficient",
            "l2_error_after_atom",
        ],
        atom_rows,
    )

    histogram_rows = []
    for count in range(1, args.max_atoms + 1):
        histogram_rows.append(
            {
                "atoms_required": count,
                "directions": int(np.sum(within_tolerance & (num_atoms == count))),
            }
        )
    histogram_rows.append(
        {
            "atoms_required": f">{args.max_atoms}",
            "directions": int(np.sum(~within_tolerance)),
        }
    )
    _write_csv(
        out_dir / "count_histogram.csv",
        ["atoms_required", "directions"],
        histogram_rows,
    )

    successful_counts = num_atoms[within_tolerance]
    result_summary = {
        "directions": int(target_shape[0]),
        "within_tolerance": int(np.sum(within_tolerance)),
        "not_within_tolerance_at_cap": int(np.sum(~within_tolerance)),
        "success_rate": float(np.mean(within_tolerance)),
        "count_histogram": {
            str(row["atoms_required"]): int(row["directions"]) for row in histogram_rows
        },
        "successful_count_min": int(successful_counts.min()) if successful_counts.size else None,
        "successful_count_median": (
            float(np.median(successful_counts)) if successful_counts.size else None
        ),
        "successful_count_mean": (
            float(np.mean(successful_counts)) if successful_counts.size else None
        ),
        "successful_count_max": int(successful_counts.max()) if successful_counts.size else None,
        "final_l2_error_min": float(l2_error.min()),
        "final_l2_error_median": float(np.median(l2_error)),
        "final_l2_error_mean": float(l2_error.mean()),
        "final_l2_error_max": float(l2_error.max()),
    }
    metadata = {
        "model": args.model,
        "source_checkpoint": str(source_path),
        "target_checkpoint": str(target_path),
        "source_width": args.source_width,
        "target_width": args.target_width,
        "source_top_k": args.source_top_k,
        "target_top_k": args.target_top_k,
        "source_decoder_shape": list(source_shape),
        "target_decoder_shape": list(target_shape),
        "tolerance": args.tolerance,
        "error_definition": "absolute L2 norm on the stored target W_dec row",
        "max_atoms": args.max_atoms,
        "algorithm": "batched orthogonal matching pursuit with signed coefficients",
        "minimality": (
            "first support meeting tolerance along the greedy OMP path; "
            "not a certificate of globally minimum cardinality"
        ),
        "atom_preprocessing": (
            "source decoder rows L2-normalized for selection and fitting; reported "
            "coefficients are converted back to multiply the stored raw W_dec rows"
        ),
        "target_preprocessing": "none",
        "batch_size": args.batch_size,
        "device": str(args.device),
        "matmul_precision": args.matmul_precision,
        "elapsed_seconds": elapsed_seconds,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "outputs": {
            "decompositions": "decompositions.npz",
            "per_direction_report": "atom_counts.csv",
            "atoms_used_long_table": "atoms_used.csv",
            "count_histogram": "count_histogram.csv",
        },
        "result_summary": result_summary,
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run(args: argparse.Namespace) -> None:
    started = time.monotonic()
    if args.device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    torch.set_float32_matmul_precision(args.matmul_precision)

    models_dir = args.models_dir.resolve()
    source_path = models_dir / f"{args.model}_m{args.source_width}_k{args.source_top_k}"
    target_path = models_dir / f"{args.model}_m{args.target_width}_k{args.target_top_k}"
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if not target_path.is_file():
        raise FileNotFoundError(target_path)
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {args.out_dir}; pass --overwrite to replace files"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading source decoder: {source_path}", flush=True)
    source_decoder = load_decoder(source_path)
    print(f"Loading target decoder: {target_path}", flush=True)
    target_decoder = load_decoder(target_path)
    source_shape = tuple(source_decoder.shape)
    target_shape = tuple(target_decoder.shape)
    if source_shape[0] != args.source_width:
        raise ValueError(f"Expected source width {args.source_width}, got {source_shape[0]}")
    if target_shape[0] != args.target_width:
        raise ValueError(f"Expected target width {args.target_width}, got {target_shape[0]}")
    if source_shape[1] != target_shape[1]:
        raise ValueError(f"Decoder dimensions differ: {source_shape[1]} != {target_shape[1]}")
    if not bool(torch.isfinite(source_decoder).all()):
        raise ValueError(f"Non-finite values in {source_path}")
    if not bool(torch.isfinite(target_decoder).all()):
        raise ValueError(f"Non-finite values in {target_path}")

    source_norms_tensor = torch.linalg.vector_norm(source_decoder.float(), dim=1)
    target_norms_tensor = torch.linalg.vector_norm(target_decoder.float(), dim=1)
    if bool((source_norms_tensor == 0).any()):
        raise ValueError("Source decoder contains a zero-norm atom")
    source_norms = source_norms_tensor.numpy().astype(np.float64, copy=True)
    target_norms = target_norms_tensor.numpy().astype(np.float64, copy=True)

    print(
        f"Moving and normalizing {source_shape[0]} x {source_shape[1]} atoms on {args.device}",
        flush=True,
    )
    atoms = F.normalize(
        source_decoder.to(device=args.device, dtype=torch.float32), dim=1
    ).contiguous()

    atom_indices = np.full(
        (target_shape[0], args.max_atoms), -1, dtype=np.int64
    )
    normalized_coefficients = np.zeros(
        (target_shape[0], args.max_atoms), dtype=np.float64
    )
    num_atoms = np.zeros(target_shape[0], dtype=np.int64)
    l2_error = np.full(target_shape[0], np.nan, dtype=np.float64)
    l2_error_by_count = np.full(
        (target_shape[0], args.max_atoms + 1), np.nan, dtype=np.float64
    )

    total_batches = (target_shape[0] + args.batch_size - 1) // args.batch_size
    for batch_number, start in enumerate(
        range(0, target_shape[0], args.batch_size), start=1
    ):
        stop = min(start + args.batch_size, target_shape[0])
        batch_started = time.monotonic()
        batch_targets = target_decoder[start:stop].to(dtype=torch.float32)
        result = orthogonal_matching_pursuit(
            atoms,
            batch_targets,
            tolerance=args.tolerance,
            max_atoms=args.max_atoms,
        )
        atom_indices[start:stop] = result.atom_indices.numpy()
        normalized_coefficients[start:stop] = result.normalized_coefficients.numpy()
        num_atoms[start:stop] = result.num_atoms.numpy()
        l2_error[start:stop] = result.l2_error.numpy()
        l2_error_by_count[start:stop] = result.l2_error_by_count.numpy()
        successes = int(np.sum(l2_error[start:stop] <= args.tolerance))
        print(
            f"Batch {batch_number}/{total_batches} targets {start}:{stop}: "
            f"{successes}/{stop - start} met tolerance in "
            f"{time.monotonic() - batch_started:.1f}s",
            flush=True,
        )

    print("Refitting and verifying selected supports against raw W_dec rows", flush=True)
    raw_coefficients, verified_l2_error = refit_and_verify_raw(
        source_decoder,
        target_decoder,
        atom_indices,
        num_atoms,
        device=args.device,
        batch_size=args.batch_size,
    )
    max_error_change = float(np.max(np.abs(verified_l2_error - l2_error)))
    l2_error = verified_l2_error
    selected = atom_indices >= 0
    normalized_coefficients.fill(0.0)
    normalized_coefficients[selected] = (
        raw_coefficients[selected] * source_norms[atom_indices[selected]]
    )
    for target_id, count in enumerate(num_atoms):
        l2_error_by_count[target_id, count] = l2_error[target_id]
    print(f"Raw verification max error change: {max_error_change:.3e}", flush=True)

    elapsed_seconds = time.monotonic() - started
    write_outputs(
        out_dir=args.out_dir,
        args=args,
        source_path=source_path,
        target_path=target_path,
        source_shape=source_shape,
        target_shape=target_shape,
        source_norms=source_norms,
        target_norms=target_norms,
        atom_indices=atom_indices,
        raw_coefficients=raw_coefficients,
        normalized_coefficients=normalized_coefficients,
        num_atoms=num_atoms,
        l2_error=l2_error,
        l2_error_by_count=l2_error_by_count,
        elapsed_seconds=elapsed_seconds,
    )
    successes = int(np.sum(l2_error <= args.tolerance))
    print(
        f"Finished in {elapsed_seconds:.1f}s: {successes}/{target_shape[0]} directions "
        f"met L2 <= {args.tolerance}; results in {args.out_dir}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("gemini", "nemotron"), default="gemini")
    parser.add_argument("--source-width", type=int, default=131072)
    parser.add_argument("--source-top-k", type=int, default=128)
    parser.add_argument("--target-width", type=int, default=4096)
    parser.add_argument("--target-top-k", type=int, default=32)
    parser.add_argument("--tolerance", type=float, default=0.1)
    parser.add_argument("--max-atoms", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--models-dir", type=Path, default=_paper_path(get_path("models_dir"))
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_paper_path("full_experiments/results/sparse_decoder_reconstruction/gemini_m4096_from_m131072"),
    )
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda"))
    parser.add_argument(
        "--matmul-precision", choices=("highest", "high", "medium"), default="highest"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.max_atoms < 1:
        parser.error("--max-atoms must be positive")
    if args.tolerance < 0:
        parser.error("--tolerance must be nonnegative")
    return args


if __name__ == "__main__":
    run(parse_args())
