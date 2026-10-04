#!/usr/bin/env python3
"""Download the GBIF backbone and create hierarchy/gbif/rows.csv."""
from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location
import argparse, csv, zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np
from common import download, stable_id, write_meta, write_rows

BACKBONE_URL = "https://hosted-datasets.gbif.org/datasets/backbone/current/backbone.zip"
LEVELS = ["kingdom", "phylum", "class", "order", "family", "genus"]

def read_vernacular(bundle):
    records, ids = [], set()
    with bundle.open("VernacularName.tsv") as raw:
        for row in csv.DictReader((x.decode("utf-8") for x in raw), delimiter="\t"):
            if row.get("language", "").lower() == "en" and row.get("vernacularName", "").strip():
                records.append(row); ids.add(row["taxonID"])
    return records, ids

def read_species(bundle, wanted):
    species = {}
    with bundle.open("Taxon.tsv") as raw:
        for row in csv.DictReader((x.decode("utf-8") for x in raw), delimiter="\t"):
            if (row["taxonID"] in wanted and row.get("taxonRank", "").lower() == "species"
                    and row.get("taxonomicStatus", "").lower() == "accepted"):
                species[row["taxonID"]] = row
    return species

def species_splits(taxon_ids, seed):
    """Use the GBIF two-shuffle species partition procedure."""
    ids = np.asarray(sorted(taxon_ids), dtype=object)
    shuffled = ids[np.random.RandomState(seed).permutation(len(ids))]
    labels = np.where(np.arange(len(ids)) < len(ids) / 2, "train", "test")
    order = np.random.RandomState(seed + 1).permutation(len(ids))
    return {str(shuffled[i]): str(labels[i]) for i in order}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=_paper_path("data/hierarchy"))
    parser.add_argument("--raw-dir", type=Path, default=_paper_path("data/raw/gbif"))
    parser.add_argument("--backbone-url", default=BACKBONE_URL)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    archive = download(args.backbone_url, args.raw_dir / "backbone.zip")
    with zipfile.ZipFile(archive) as bundle:
        vernacular, wanted = read_vernacular(bundle)
        species = read_species(bundle, wanted)
    by_species = defaultdict(set)
    for row in vernacular:
        if row["taxonID"] in species:
            by_species[row["taxonID"]].add(row["vernacularName"].strip())
    splits = species_splits(by_species, args.seed)
    names = [("scientific", species[t]["scientificName"], t) for t in sorted(by_species)]
    common_species = defaultdict(list)
    for taxon_id, common_names in by_species.items():
        for name in common_names: common_species[name].append(taxon_id)
    names += [("common", name, sorted(ids)[0]) for name, ids in sorted(common_species.items())]
    rows = [{"row_idx": i, "nameID": stable_id(kind, text), "nameType": kind,
             "text": text, "model": "gemini-embedding-2", "split": splits[taxon_id]}
            for i, (kind, text, taxon_id) in enumerate(names)]
    out = args.output_root / "gbif"
    write_rows(out / "rows.csv", rows, ["row_idx", "nameID", "nameType", "text", "model", "split"])
    memberships = []
    for taxon_id in sorted(by_species):
        taxon = species[taxon_id]
        for name in sorted(by_species[taxon_id]):
            memberships.append({"taxonID": taxon_id, "scientificName": taxon["scientificName"],
                "vernacularName": name, **{level: taxon.get(level, "") for level in LEVELS},
                "split": splits[taxon_id]})
    write_rows(out / "species_common_name_pairs.csv", memberships,
               ["taxonID", "scientificName", "vernacularName", *LEVELS, "split"])
    write_rows(out / "species_split.csv",
               [{"taxonID": key, "split": value} for key, value in sorted(splits.items())],
               ["taxonID", "split"])
    write_meta(out / "source_meta.json", {"backbone_url": args.backbone_url,
        "accepted_species": len(by_species), "embedding_rows": len(rows),
        "split": "two-shuffle species split", "split_seed": args.seed,
        "note": "Exact paper comparisons require the reference 2023 backbone snapshot."})

if __name__ == "__main__": main()
