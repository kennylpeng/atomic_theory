"""Aggregate frozen probe summaries for downstream paper figures."""
from __future__ import annotations
import csv
from pathlib import Path
from typing import Any

def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

def read_all_summaries(out_base: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    # Collect every completed probe summary below the result root.
    for path in sorted(out_base.glob("*/*/summary.tsv")):
        with path.open(newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f, delimiter="\t"))
    return rows

def aggregate(args):
    """Aggregate frozen probe summaries for paper producers (no diagnostic plots)."""
    rows = read_all_summaries(args.out_base)
    if not rows:
        raise ValueError(f"No summary.tsv files found under {args.out_base}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    write_tsv(args.out_base / "all_summaries.tsv", rows, fields)
