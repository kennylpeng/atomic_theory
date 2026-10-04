#!/usr/bin/env python3
"""Profiles for complete sparse activations of every experiment SAE."""

from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import sys
from pathlib import Path

sys.path.insert(0, str(_paper_path(__file__).resolve().parents[1]))
from scripts import cache_full_sparse_activations as core  # noqa: E402


STANDARD_WIDTH_TOPK = (
    (512, 32),
    (1024, 32),
    (2048, 32),
    (4096, 32),
    (8192, 64),
    (16384, 64),
    (32768, 64),
    (65536, 128),
    (131072, 128),
)
SPLIT_NAMES = ("wikipedia", "no_wikipedia", "random1", "random2")
CACHE_BASE = _paper_path("/resources/activation_cache_dir")
PROFILES = {
    "gemini_remaining": {
        "family": "gemini",
        "input_dim": 3072,
        "expected_shards": 929,
        "expected_rows": 89_827_558,
        "cache_root": CACHE_BASE / "gemini_remaining_experiment_saes_post_topk",
        "main_widths": {512, 1024, 2048, 8192, 32768, 65536},
        "include_splits": True,
    },
    "nemotron_all": {
        "family": "nemotron",
        "input_dim": 4096,
        "expected_shards": 923,
        "expected_rows": 89_227_558,
        "cache_root": CACHE_BASE / "nemotron_all_experiment_saes_post_topk",
        "main_widths": {width for width, _ in STANDARD_WIDTH_TOPK},
        "include_splits": True,
    },
}


def sae_specs(profile: dict[str, object]) -> tuple[dict[str, object], ...]:
    family = str(profile["family"])
    selected_widths = profile["main_widths"]
    specs = [
        {"name": f"{family}_m{width}_k{top_k}", "width": width, "top_k": top_k}
        for width, top_k in STANDARD_WIDTH_TOPK
        if width in selected_widths
    ]
    if profile["include_splits"]:
        specs.extend(
            {
                "name": f"{family}_{split}_m16384_k64",
                "width": 16384,
                "top_k": 64,
            }
            for split in SPLIT_NAMES
        )
    return tuple(specs)


def configure(profile_name: str) -> None:
    if profile_name not in PROFILES:
        choices = ", ".join(sorted(PROFILES))
        raise SystemExit(f"Unknown profile {profile_name!r}; choose one of: {choices}")
    profile = PROFILES[profile_name]
    core.DEFAULT_CACHE_ROOT = _paper_path(profile["cache_root"])
    core.EXPECTED_INPUT_DIM = int(profile["input_dim"])
    core.EXPECTED_SHARDS = int(profile["expected_shards"])
    core.EXPECTED_ROWS = int(profile["expected_rows"])
    core.SAE_SPECS = sae_specs(profile)
    core.TRACKED_FEATURES = ()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(
            f"Usage: {_paper_path(sys.argv[0]).name} "
            f"<{'|'.join(sorted(PROFILES))}> <prepare|compute|audit> ..."
        )
    selected_profile = sys.argv.pop(1)
    configure(selected_profile)
    arguments = core.parse_args()
    arguments.func(arguments)
