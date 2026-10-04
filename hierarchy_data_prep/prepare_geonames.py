#!/usr/bin/env python3
"""Download GeoNames cities5000 and create hierarchy/geonames/rows.csv."""

from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import argparse
import csv
import zipfile
from pathlib import Path

from common import download, stratified_half_split, write_meta, write_rows

CITIES_URL = "https://download.geonames.org/export/dump/cities5000.zip"
COUNTRIES_URL = "https://download.geonames.org/export/dump/countryInfo.txt"
RAW_FIELDS = [
    "geoname_id", "name", "ascii_name", "alternate_names", "latitude", "longitude",
    "feature_class", "feature_code", "country_code", "cc2", "admin1_code",
    "admin2_code", "admin3_code", "admin4_code", "population", "elevation",
    "dem", "timezone", "modification_date",
]
OUT_FIELDS = [
    "row_idx", "geoname_id", "name", "ascii_name", "latitude", "longitude",
    "feature_class", "feature_code", "country_code", "admin1_code", "admin2_code",
    "population", "elevation", "timezone", "modification_date", "split", "continent",
]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=_paper_path("data/hierarchy"))
    parser.add_argument("--raw-dir", type=Path, default=_paper_path("data/raw/geonames"))
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--cities-url", default=CITIES_URL)
    parser.add_argument("--countries-url", default=COUNTRIES_URL)
    return parser.parse_args()


def country_continents(path: Path) -> dict[str, str]:
    result = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) > 8:
                result[fields[0]] = fields[8]
    names = {"AF": "Africa", "AS": "Asia", "EU": "Europe", "NA": "North America",
             "OC": "Oceania", "SA": "South America", "AN": "Antarctica"}
    return {country: names.get(code, code) for country, code in result.items()}


def main() -> None:
    args = arguments()
    archive = download(args.cities_url, args.raw_dir / "cities5000.zip")
    countries_path = download(args.countries_url, args.raw_dir / "countryInfo.txt")
    continents = country_continents(countries_path)
    rows = []
    with zipfile.ZipFile(archive) as bundle, bundle.open("cities5000.txt") as raw:
        lines = (line.decode("utf-8") for line in raw)
        for row_idx, values in enumerate(csv.reader(lines, delimiter="\t")):
            source = dict(zip(RAW_FIELDS, values))
            row = {field: source.get(field, "") for field in OUT_FIELDS}
            row["row_idx"] = row_idx
            row["continent"] = continents.get(source["country_code"], "")
            rows.append(row)
    splits = stratified_half_split((row["country_code"] for row in rows), args.seed)
    for row, split in zip(rows, splits):
        row["split"] = split
    out = args.output_root / "geonames"
    write_rows(out / "rows.csv", rows, OUT_FIELDS)
    write_meta(out / "source_meta.json", {
        "cities_url": args.cities_url, "country_info_url": args.countries_url,
        "rows": len(rows), "split": "stable country-stratified half split",
        "split_seed": args.seed,
    })


if __name__ == "__main__":
    main()
