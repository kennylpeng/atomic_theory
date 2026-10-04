"""Shared synthetic experiment imports and local output paths."""
from pathlib import Path
from experiments.recovery_principle.code.distributions import Distribution, Spec
from experiments.recovery_principle.code.metrics import activation_recovery, prefix, distinct_triple
from experiments.recovery_principle.code.core import TopK, collect, atomic_json, digest
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent.parent
