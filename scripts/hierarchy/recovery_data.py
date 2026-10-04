"""Load frozen probes and hierarchy memberships for category and pair recovery."""

from __future__ import annotations
from hierarchy_probe_config import GBIF_METADATA_DIR, HIERARCHY_DATA_DIR, PROBE_RESULTS_DIR
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import csv

from collections import defaultdict

from fractions import Fraction

from pathlib import Path



WIDTHS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

MODELS = ["gemini", "nemotron"]

DATASETS = ["wordfreq", "geonames", "gbif"]

TITLES = {"wordfreq": "Word prefixes", "geonames": "Geography", "gbif": "Organism taxonomy"}

RANKS = ["kingdom", "phylum", "class", "order", "family", "genus"]

DEFAULT_OUTPUT = _paper_path(__file__).resolve().parents[2] / "full_experiments/plots/simultaneous_family_recovery"

def read_table(path: Path, delimiter: str = ",") -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))

def write_table(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def passes(row: dict, cutoff: Fraction) -> bool:
    tp, fp, fn = (int(row[f"test_{key}"]) for key in ("tp", "fp", "fn"))
    denominator = 2 * tp + fp + fn
    recomputed = 2 * tp / denominator if denominator else 0.0
    if abs(recomputed - float(row["test_f1"])) > 1e-12:
        raise ValueError(f"Inconsistent saved test F1: {row['category_id']}")
    return denominator > 0 and 2 * tp * cutoff.denominator >= cutoff.numerator * denominator

def geography_sources(mode: str) -> list[str]:
    return ["geonames_continent_questions", "geonames_country_questions"] if mode == "separate" else ["geonames_where_questions"]

def load_probes(root: Path, geography_prompts: str = "shared") -> tuple[dict, dict, list[Path]]:
    probes, category_sets, sources = {}, {}, []
    for dataset in DATASETS:
        source_datasets = geography_sources(geography_prompts) if dataset == "geonames" else [dataset]
        for model in MODELS:
            for width in WIDTHS:
                rows = []
                for source in source_datasets:
                    paths = sorted((root / source).glob(f"{model}_m{width}_k*/summary.tsv"))
                    if len(paths) != 1:
                        raise ValueError(f"Expected exactly one original-sweep summary: {source}, {model}, {width}: {paths}")
                    path = paths[0]
                    source_rows = read_table(path, "\t")
                    for row in source_rows:
                        if (row["dataset"], row["model"], int(row["width"])) != (source, model, width):
                            raise ValueError(f"Incorrect summary metadata in {path}")
                        if min(int(row["train_pos"]), int(row["test_pos"])) <= 0:
                            raise ValueError(f"Category without selection/test support in {path}")
                    if source in ("geonames_continent_questions", "geonames_country_questions"):
                        expected_level = "continent" if source == "geonames_continent_questions" else "country"
                        if {row["level"] for row in source_rows} != {expected_level}:
                            raise ValueError(f"Incorrect geography levels in {path}")
                    rows.extend(source_rows)
                    sources.append(path)
                by_id = {row["category_id"]: row for row in rows}
                if len(rows) != len(by_id):
                    raise ValueError(f"Duplicate categories: {dataset}, {model}, {width}")
                if len({row["top_k"] for row in rows}) != 1:
                    raise ValueError(f"Different SAE sparsities: {dataset}, {model}, {width}")
                if dataset not in category_sets:
                    category_sets[dataset] = set(by_id)
                if set(by_id) != category_sets[dataset]:
                    raise ValueError(f"Category denominator changes: {dataset}, {model}, {width}")
                probes[dataset, model, width] = by_id
    return probes, category_sets, sources

def build_families(categories: dict, data_root: Path, gbif_root: Path, geography_prompts: str = "shared", *, min_children: int = 2) -> tuple[dict, list[dict], list[Path]]:
    relationships = {dataset: defaultdict(set) for dataset in DATASETS}
    for category in categories["wordfreq"]:
        if category.startswith("two_letter_prefix:"):
            relationships["wordfreq"][category].add("one_letter_prefix:" + category.split(":", 1)[1][0])
    geography_paths = [data_root / source / "rows.csv" for source in geography_sources(geography_prompts)]
    geography_rows = read_table(geography_paths[0])
    identity_columns = ["geoname_id", "name", "country_code", "continent", "split"]
    for path in geography_paths[1:]:
        other_rows = read_table(path)
        if len(other_rows) != len(geography_rows) or any(
            any(a[column] != b[column] for column in identity_columns)
            for a, b in zip(geography_rows, other_rows)
        ):
            raise ValueError(f"Geography prompts do not share city records and splits: {path}")
    for row in geography_rows:
        if row["country_code"]:
            relationships["geonames"]["country:" + row["country_code"]].add(
                "continent:" + row["continent"] if row["continent"] else ""
            )
    taxonomy_path = gbif_root / "category_threshold_f1_d131072/species_common_name_pairs.csv"
    with taxonomy_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            for parent, child in zip(RANKS, RANKS[1:]):
                if row[child]:
                    relationships["gbif"][f"{child}:{row[child]}"].add(
                        f"{parent}:{row[parent]}" if row[parent] else ""
                    )
    families, excluded = {}, []
    for dataset in DATASETS:
        children = defaultdict(list)
        for child, parents in sorted(relationships[dataset].items()):
            if child not in categories[dataset]:
                continue
            if len(parents) != 1 or "" in parents:
                excluded.append({"dataset": dataset, "child": child, "reason": "missing_or_ambiguous_immediate_parent", "parents": sorted(parents)})
                continue
            parent = next(iter(parents))
            if parent not in categories[dataset]:
                excluded.append({"dataset": dataset, "child": child, "reason": "parent_not_in_probe_categories", "parents": [parent]})
                continue
            children[parent].append(child)
        families[dataset] = {parent: sorted(ids) for parent, ids in sorted(children.items()) if len(ids) >= min_children}
        if not families[dataset]:
            raise ValueError(f"No eligible families: {dataset}")
    return families, excluded, geography_paths + [taxonomy_path]
