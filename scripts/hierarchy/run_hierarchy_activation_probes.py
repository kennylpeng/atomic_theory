#!/usr/bin/env python3
"""Run one hierarchy SAE activation probe."""
from __future__ import annotations

import argparse
from pathlib import Path

from hierarchy_probe_computation import run_probe
from hierarchy_probe_config import PROBE_RESULTS_DIR

DATASETS = (
    "geonames", "gbif", "wordfreq", "geonames_country_questions",
    "geonames_continent_questions", "geonames_us_state_questions",
    "geonames_ca_county_questions", "geonames_where_questions",
    "geonames_us_state_where_questions", "geonames_ca_county_where_questions",
)

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--model", choices=("gemini", "nemotron"), required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--out-base", type=Path, default=PROBE_RESULTS_DIR)
    parser.add_argument("--min-count", type=int, default=100)
    parser.add_argument("--min-train-tp", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)

def main() -> None:
    run_probe(parse_args())

if __name__ == "__main__":
    main()
