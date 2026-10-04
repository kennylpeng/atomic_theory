"""Logical checkpoint names and seeds for sparsity and independent-seed controls."""
from pathlib import Path
import json

from project_paths import resource_path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = json.loads((ROOT / "release/provenance/control-checkpoints.json").read_text())["checkpoints"]
FIXED = {
    family: {r["width"]: r["seed"] for r in CHECKPOINTS
             if r["family"] == family and r["control"] == "fixed-k"}
    for family in ("gemini", "nemotron")
}
ALTERNATE = {r["width"]: r["seed"] for r in CHECKPOINTS
             if r["family"] == "gemini" and r["control"] == "alternate seed"}


def checkpoint_path(family, width, top_k, seed):
    """Resolve a checkpoint by model settings, independent of scheduler layout."""
    return resource_path(f"/resources/control_models_dir/{family}_m{width}_k{top_k}_seed{seed}.pt")
