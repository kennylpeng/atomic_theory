#!/usr/bin/env python3
"""Nemotron 131K configuration for the sampled-example text catalog."""

from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import sys
from pathlib import Path

sys.path.insert(0, str(_paper_path(__file__).resolve().parents[1]))
from scripts import cache_payload as feature_reader  # noqa: E402
from scripts import sparse_activation_cache as sparse_reader  # noqa: E402

MODEL_NAME = "nemotron_m131072_k128"
SOURCE_ROOT = _paper_path(
    "/resources/activation_cache_dir/"
    "nemotron_m131072_post_topk_view_v1"
)
NUMERIC_CACHE = _paper_path(
    "/resources/activation_cache_dir/"
    "nemotron_m131072_binned_random_examples_v1/final"
)
OUTPUT_ROOT = NUMERIC_CACHE.parent / "text_catalog"
MODEL_SPEC = {"width": 131072, "top_k": 128}
MODEL_SALT = 0x4E454D4F131072

feature_reader.MODEL_SPECS = {MODEL_NAME: MODEL_SPEC}
feature_reader.WIDTH_TO_MODEL = {131072: MODEL_NAME}
sparse_reader.SUPPORTED_MODELS = frozenset((MODEL_NAME,))
from scripts import cache_binned_random_examples as numeric_core  # noqa: E402

numeric_core.ARTIFACT_KIND = "nemotron_post_topk_binned_random_examples"
numeric_core.DEFAULT_CACHE_ROOT = SOURCE_ROOT
numeric_core.MODEL_SPECS = {
    MODEL_NAME: {
        **MODEL_SPEC,
        "salt": MODEL_SALT,
        "expected_positive_slots": 0,
    }
}
from scripts import cache_binned_example_text_catalog as core  # noqa: E402

core.ARTIFACT_KIND = "nemotron_binned_random_example_text_catalog"
core.MODEL_SPECS = numeric_core.MODEL_SPECS
core.DEFAULT_NUMERIC_CACHE = NUMERIC_CACHE
core.DEFAULT_OUTPUT_ROOT = OUTPUT_ROOT


if __name__ == "__main__":
    core.main()
