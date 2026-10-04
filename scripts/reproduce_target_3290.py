#!/usr/bin/env python3
"""Recompute the 17-term molecule with the paper's absolute squared-error rule."""
from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(_paper_path(__file__).resolve().parents[1]))
from project_paths import get_path
from scripts.sparse_decoder_reconstruction import load_decoder, orthogonal_matching_pursuit, refit_and_verify_raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
    models = args.models_dir or _paper_path(get_path("models_dir"))
    source = load_decoder(models / "gemini_m131072_k128")
    target = load_decoder(models / "gemini_m4096_k32")[3290:3291]
    device = torch.device(args.device)
    atoms = source.to(device=device, dtype=torch.float32)
    atoms = atoms / torch.linalg.vector_norm(atoms, dim=1)[:, None]
    result = orthogonal_matching_pursuit(atoms, target.to(device), tolerance=np.sqrt(.05), max_atoms=128)
    count = int(result.num_atoms[0])
    coefficients, errors = refit_and_verify_raw(source, target, result.atom_indices.numpy(), result.num_atoms.numpy(), device=device, batch_size=1)
    support = result.atom_indices[0, :count].numpy()
    reconstruction = coefficients[0, :count] @ source[support].double().numpy()
    d = target[0].double().numpy()
    history = result.l2_error_by_count[0].numpy()
    if errors[0] ** 2 > .05 + 1e-9 or (count > 0 and history[count - 1] ** 2 <= .05):
        raise ValueError("OMP did not stop at the first passing support")
    payload = dict(target_feature=3290, atom_count=count, atom_features=support.tolist(),
        coefficients=coefficients[0, :count].tolist(), squared_l2_error=float(errors[0] ** 2),
        reconstruction_cosine=float(d @ reconstruction / (np.linalg.norm(d) * np.linalg.norm(reconstruction))),
        criterion="first OMP support with absolute squared error <= 0.05; raw stored decoder scale",
        device=str(device), torch_version=torch.__version__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
