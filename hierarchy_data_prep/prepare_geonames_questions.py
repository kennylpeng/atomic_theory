#!/usr/bin/env python3
"""Create GeoNames country- and continent-question row tables."""
from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location
import argparse, csv
from pathlib import Path
from common import write_meta, write_rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=_paper_path("data/hierarchy"))
    args = parser.parse_args()
    source = args.output_root / "geonames" / "rows.csv"
    with source.open(newline="", encoding="utf-8") as file:
        source_rows = list(csv.DictReader(file))
    specs = [("geonames_country_questions", "country", "what country is {city} in"),
             ("geonames_continent_questions", "continent", "what continent is {city} in")]
    for dataset, kind, template in specs:
        rows = []
        for source_row in source_rows:
            row = dict(source_row)
            row.update(text=template.format(city=row["name"]), source_text=row["name"], question_type=kind)
            rows.append(row)
        out = args.output_root / dataset
        write_rows(out / "rows.csv", rows, list(rows[0]))
        write_meta(out / "source_meta.json", {"source": str(source), "rows": len(rows),
            "template": template, "split": "inherited unchanged from GeoNames"})

if __name__ == "__main__": main()
