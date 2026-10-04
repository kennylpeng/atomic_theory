"""Hierarchy target definitions, preserving source row order and taxonomy unions."""

from __future__ import annotations
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def split_masks(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    # Feature and threshold selection uses train; test remains held out.
    split = np.asarray([row.get("split", "") for row in rows])
    train = split == "train"
    test = split == "test"
    if not np.any(train) or not np.any(test):
        raise ValueError("rows.csv must contain train/test values in the split column")
    return train, test


def targets_from_column(
    rows: list[dict[str, str]],
    *,
    column: str,
    level: str,
    min_count: int,
) -> list[dict[str, Any]]:
    # Each target is a binary task represented by its positive row indices.
    counts = Counter(row[column] for row in rows if row.get(column))
    targets = []
    for value, count in sorted(counts.items()):
        if count >= min_count:
            idx = np.asarray(
                [i for i, row in enumerate(rows) if row[column] == value],
                dtype=np.int32,
            )
            targets.append(
                {
                    "level": level,
                    "category_id": f"{level}:{value}",
                    "category": value,
                    "positive_count": int(count),
                    "positive_rows": idx,
                }
            )
    return targets


def build_geonames_targets(
    rows: list[dict[str, str]], min_count: int
) -> list[dict[str, Any]]:
    return targets_from_column(
        rows, column="continent", level="continent", min_count=min_count
    ) + targets_from_column(
        rows, column="country_code", level="country", min_count=min_count
    )


def build_geonames_country_targets(
    rows: list[dict[str, str]], min_count: int
) -> list[dict[str, Any]]:
    return targets_from_column(
        rows, column="country_code", level="country", min_count=min_count
    )


def build_geonames_continent_targets(
    rows: list[dict[str, str]], min_count: int
) -> list[dict[str, Any]]:
    return targets_from_column(
        rows, column="continent", level="continent", min_count=min_count
    )


def build_geonames_state_targets(
    rows: list[dict[str, str]], min_count: int
) -> list[dict[str, Any]]:
    column = "state_code" if rows and "state_code" in rows[0] else "admin1_code"
    return targets_from_column(rows, column=column, level="state", min_count=min_count)


def build_geonames_county_targets(
    rows: list[dict[str, str]], min_count: int
) -> list[dict[str, Any]]:
    column = "county_code" if rows and "county_code" in rows[0] else "admin2_code"
    return targets_from_column(rows, column=column, level="county", min_count=min_count)


def build_wordfreq_targets(
    rows: list[dict[str, str]], min_count: int
) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    texts = [row["text"] for row in rows]
    # Build separate binary tasks for every one- and two-letter prefix.
    for n, level in [(1, "one_letter_prefix"), (2, "two_letter_prefix")]:
        counts = Counter(text[:n] for text in texts if len(text) >= n)
        for prefix, count in sorted(counts.items()):
            if count < min_count:
                continue
            idx = np.asarray(
                [
                    i
                    for i, text in enumerate(texts)
                    if len(text) >= n and text[:n] == prefix
                ],
                dtype=np.int32,
            )
            targets.append(
                {
                    "level": level,
                    "category_id": f"{level}:{prefix}",
                    "category": prefix,
                    "positive_count": int(count),
                    "positive_rows": idx,
                }
            )
    return targets


def build_gbif_targets(
    rows: list[dict[str, str]], min_count: int, metadata_dir: Path
) -> list[dict[str, Any]]:
    # External GBIF tables define eligible categories and species membership.
    categories_path = metadata_dir / "categories_min100_species.tsv"
    pairs_path = metadata_dir / "species_common_name_pairs.csv"

    category_meta: dict[str, dict[str, Any]] = {}
    with categories_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, dialect="excel-tab"):
            n_species = int(row["n_species"])
            if n_species < min_count:
                continue
            category_meta[row["category_id"]] = {
                "level": row["level"],
                "category": row["category"],
                "category_id": row["category_id"],
                "positive_count": n_species,
            }

    # Common names use source row IDs; scientific names use their text.
    common_row_to_categories: dict[int, set[str]] = defaultdict(set)
    scientific_to_categories: dict[str, set[str]] = defaultdict(set)
    levels = ["kingdom", "phylum", "class", "order", "family", "genus"]
    with pairs_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ids = [f"{level}:{row[level]}" for level in levels if row.get(level)]
            ids = [category_id for category_id in ids if category_id in category_meta]
            if not ids:
                continue
            common_row_to_categories[int(row["common_row_idx"])].update(ids)
            scientific_to_categories[row["scientificName"]].update(ids)

    # Translate external taxonomy membership into indices in this rows.csv.
    category_to_rows: dict[str, list[int]] = {
        category_id: [] for category_id in category_meta
    }
    for i, row in enumerate(rows):
        if row["nameType"] == "common":
            ids = common_row_to_categories.get(int(row["row_idx"]), set())
        else:
            ids = scientific_to_categories.get(row["text"], set())
        for category_id in ids:
            category_to_rows[category_id].append(i)

    targets = []
    for category_id, meta in sorted(
        category_meta.items(), key=lambda kv: (kv[1]["level"], kv[1]["category"])
    ):
        targets.append(
            {
                **meta,
                "positive_rows": np.asarray(
                    sorted(set(category_to_rows[category_id])), dtype=np.int32
                ),
            }
        )
    return targets


def build_targets(
    dataset: str,
    rows: list[dict[str, str]],
    min_count: int = 100,
    metadata_dir: Path | None = None,
) -> list[dict[str, Any]]:
    if dataset == "geonames":
        return build_geonames_targets(rows, min_count)
    if dataset == "geonames_country_questions":
        return build_geonames_country_targets(rows, min_count)
    if dataset == "geonames_continent_questions":
        return build_geonames_continent_targets(rows, min_count)
    if dataset == "geonames_us_state_questions":
        return build_geonames_state_targets(rows, min_count)
    if dataset == "geonames_ca_county_questions":
        return build_geonames_county_targets(rows, min_count)
    if dataset == "geonames_where_questions":
        return build_geonames_targets(rows, min_count)
    if dataset == "geonames_us_state_where_questions":
        return build_geonames_state_targets(rows, min_count)
    if dataset == "geonames_ca_county_where_questions":
        return build_geonames_county_targets(rows, min_count)
    if dataset == "wordfreq":
        return build_wordfreq_targets(rows, min_count)
    if dataset == "gbif":
        if metadata_dir is None:
            raise ValueError("GBIF requires an explicit metadata_dir")
        return build_gbif_targets(rows, min_count, Path(metadata_dir))
    raise ValueError(f"Unknown dataset: {dataset}")
