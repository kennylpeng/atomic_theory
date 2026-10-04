#!/usr/bin/env python3
"""16K configuration for the full sparse Gemini activation-cache producer."""

from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import sys
from pathlib import Path

sys.path.insert(0, str(_paper_path(__file__).resolve().parents[1]))
from scripts import cache_full_sparse_activations as core  # noqa: E402


CACHE_ROOT = _paper_path(
    "/resources/activation_cache_dir/"
    "gemini_m16384_all_corpus_post_topk"
)
MODEL_NAME = "gemini_m16384_k64"

core.DEFAULT_CACHE_ROOT = CACHE_ROOT
core.SAE_SPECS = (
    {"name": MODEL_NAME, "width": 16384, "top_k": 64},
)
# One audit probe keeps the existing exact top-example certification path active.
core.TRACKED_FEATURES = (
    {
        "key": "gemini_m16384_feature_0",
        "role": "audit_probe",
        "width": 16384,
        "feature_id": 0,
    },
)


if __name__ == "__main__":
    arguments = core.parse_args()
    arguments.func(arguments)
