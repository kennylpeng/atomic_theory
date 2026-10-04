#!/usr/bin/env python3
"""Aggregate hierarchy SAE probe summaries."""
from __future__ import annotations

import argparse
from pathlib import Path

from hierarchy_probe_config import PROBE_RESULTS_DIR
from aggregate_probes import aggregate

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-base", type=Path, default=PROBE_RESULTS_DIR)
    return parser.parse_args(argv)

def main() -> None:
    aggregate(parse_args())

if __name__ == "__main__":
    main()
