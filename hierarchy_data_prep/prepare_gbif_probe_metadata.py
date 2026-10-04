#!/usr/bin/env python3
"""Build the auxiliary GBIF taxonomy tables consumed by hierarchy probes."""
from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location
import argparse, csv
from collections import defaultdict
from pathlib import Path

LEVELS = ["kingdom", "phylum", "class", "order", "family", "genus"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=_paper_path("data/hierarchy"))
    parser.add_argument("--min-species", type=int, default=100)
    args = parser.parse_args()
    directory = args.output_root / "gbif"
    with (directory / "rows.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    common_index = {row["text"]: int(row["row_idx"]) for row in rows if row["nameType"] == "common"}
    with (directory / "species_common_name_pairs.csv").open(newline="", encoding="utf-8") as file:
        pairs = list(csv.DictReader(file))
    for row in pairs:
        row["common_row_idx"] = common_index[row["vernacularName"]]
    pair_fields = ["taxonID", "scientificName", "vernacularName", "common_row_idx", *LEVELS, "split"]
    with (directory / "species_common_name_pairs.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=pair_fields)
        writer.writeheader(); writer.writerows({key: row.get(key, "") for key in pair_fields} for row in pairs)

    categories = []
    for level in LEVELS:
        members = defaultdict(set)
        for row in pairs:
            if row.get(level): members[row[level]].add(row["taxonID"])
        for category, ids in sorted(members.items()):
            if len(ids) >= args.min_species:
                categories.append({"category_index": len(categories), "level": level,
                    "category": category, "category_id": f"{level}:{category}", "n_species": len(ids)})
    fields = ["category_index", "level", "category", "category_id", "n_species"]
    with (directory / "categories_min100_species.tsv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, delimiter="\t")
        writer.writeheader(); writer.writerows(categories)

if __name__ == "__main__": main()
