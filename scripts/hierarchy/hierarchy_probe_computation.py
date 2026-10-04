"""Hierarchy probe computation: targets, threshold search, metrics, and result files."""
from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from hierarchy_probe_config import GBIF_METADATA_DIR, HIERARCHY_DATA_DIR

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
            idx = np.asarray([i for i, row in enumerate(rows) if row[column] == value], dtype=np.int32)
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


def build_geonames_targets(rows: list[dict[str, str]], min_count: int) -> list[dict[str, Any]]:
    return targets_from_column(rows, column="continent", level="continent", min_count=min_count) + targets_from_column(
        rows, column="country_code", level="country", min_count=min_count
    )


def build_geonames_country_targets(rows: list[dict[str, str]], min_count: int) -> list[dict[str, Any]]:
    return targets_from_column(rows, column="country_code", level="country", min_count=min_count)


def build_geonames_continent_targets(rows: list[dict[str, str]], min_count: int) -> list[dict[str, Any]]:
    return targets_from_column(rows, column="continent", level="continent", min_count=min_count)


def build_geonames_state_targets(rows: list[dict[str, str]], min_count: int) -> list[dict[str, Any]]:
    column = "state_code" if rows and "state_code" in rows[0] else "admin1_code"
    return targets_from_column(rows, column=column, level="state", min_count=min_count)


def build_geonames_county_targets(rows: list[dict[str, str]], min_count: int) -> list[dict[str, Any]]:
    column = "county_code" if rows and "county_code" in rows[0] else "admin2_code"
    return targets_from_column(rows, column=column, level="county", min_count=min_count)


def build_wordfreq_targets(rows: list[dict[str, str]], min_count: int) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    texts = [row["text"] for row in rows]
    # Build separate binary tasks for every one- and two-letter prefix.
    for n, level in [(1, "one_letter_prefix"), (2, "two_letter_prefix")]:
        counts = Counter(text[:n] for text in texts if len(text) >= n)
        for prefix, count in sorted(counts.items()):
            if count < min_count:
                continue
            idx = np.asarray([i for i, text in enumerate(texts) if len(text) >= n and text[:n] == prefix], dtype=np.int32)
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


def build_gbif_targets(rows: list[dict[str, str]], min_count: int) -> list[dict[str, Any]]:
    # External GBIF tables define eligible categories and species membership.
    categories_path = GBIF_METADATA_DIR / "category_threshold_f1_d131072" / "categories_min100_species.tsv"
    pairs_path = GBIF_METADATA_DIR / "category_threshold_f1_d131072" / "species_common_name_pairs.csv"

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
    category_to_rows: dict[str, list[int]] = {category_id: [] for category_id in category_meta}
    for i, row in enumerate(rows):
        if row["nameType"] == "common":
            ids = common_row_to_categories.get(int(row["row_idx"]), set())
        else:
            ids = scientific_to_categories.get(row["text"], set())
        for category_id in ids:
            category_to_rows[category_id].append(i)

    targets = []
    for category_id, meta in sorted(category_meta.items(), key=lambda kv: (kv[1]["level"], kv[1]["category"])):
        targets.append({**meta, "positive_rows": np.asarray(sorted(set(category_to_rows[category_id])), dtype=np.int32)})
    return targets


def build_targets(dataset: str, rows: list[dict[str, str]], min_count: int) -> list[dict[str, Any]]:
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
        return build_gbif_targets(rows, min_count)
    raise ValueError(f"Unknown dataset: {dataset}")


def best_for_feature(values: np.ndarray, labels: np.ndarray, total_pos: int, total_neg: int, min_tp: int) -> dict[str, Any] | None:
    # Sort activations once to evaluate every distinct threshold cumulatively.
    if values.size == 0 or total_pos == 0:
        return None
    order = np.argsort(values, kind="mergesort")[::-1]
    sorted_values = values[order]
    sorted_labels = labels[order].astype(np.int64, copy=False)
    tp_cum = np.cumsum(sorted_labels, dtype=np.int64)
    pred_cum = np.arange(1, values.size + 1, dtype=np.int64)
    # Tied activations are evaluated together at the end of each equal-value run.
    distinct_ends = np.flatnonzero(np.r_[sorted_values[1:] != sorted_values[:-1], True])
    tp = tp_cum[distinct_ends]
    pred_pos = pred_cum[distinct_ends]
    fp = pred_pos - tp
    fn = total_pos - tp
    denom = 2 * tp + fp + fn
    valid = (tp >= min_tp) & (denom > 0)
    if not np.any(valid):
        return None
    f1 = np.zeros_like(denom, dtype=np.float64)
    f1[valid] = (2 * tp[valid]) / denom[valid]
    best = int(np.argmax(f1))
    return {
        "threshold": float(sorted_values[distinct_ends[best]]),
        "train_f1": float(f1[best]),
        "train_tp": int(tp[best]),
        "train_fp": int(fp[best]),
        "train_fn": int(fn[best]),
        "train_tn": int(total_neg - fp[best]),
    }


def metrics_at_threshold(
    f_rows: np.ndarray,
    f_values: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    # Missing sparse rows are implicit negative predictions.
    keep = f_values >= threshold
    pred_rows = f_rows[keep]
    pred_rows = pred_rows[mask[pred_rows]]
    pred_rows = np.unique(pred_rows) if pred_rows.size else pred_rows
    tp = int(np.count_nonzero(y[pred_rows]))
    fp = int(pred_rows.size - tp)
    total_pos = int(np.count_nonzero(y & mask))
    total_neg = int(np.count_nonzero((~y) & mask))
    fn = total_pos - tp
    tn = total_neg - fp
    denom = 2 * tp + fp + fn
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if total_pos == 0 else tp / total_pos
    f1 = 0.0 if denom == 0 else (2 * tp) / denom
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "pred_pos": int(pred_rows.size),
    }


def run_probe(args: argparse.Namespace) -> None:
    # One probe invocation handles one dataset, model family, and SAE width.
    rows_path = HIERARCHY_DATA_DIR / args.dataset / "rows.csv"
    sparse_path = HIERARCHY_DATA_DIR / args.dataset / f"sparse_{args.model}_m{args.width}_k{args.top_k}.npz"
    out_dir = args.out_base / args.dataset / f"{args.model}_m{args.width}_k{args.top_k}"
    summary_path = out_dir / "summary.tsv"
    if summary_path.exists() and not args.overwrite:
        print(f"Output exists, skipping: {summary_path}")
        return

    rows = read_rows(rows_path)
    train_mask, test_mask = split_masks(rows)
    targets = build_targets(args.dataset, rows, args.min_count)
    print(f"{args.dataset}: rows={len(rows):,} targets={len(targets):,}", flush=True)

    # Sparse activations use aligned row, feature, and value coordinate arrays.
    sparse = np.load(_paper_location(sparse_path))
    row_indices = sparse["row_indices"].astype(np.int64, copy=False)
    feature_indices = sparse["feature_indices"].astype(np.int64, copy=False)
    values = sparse["values"].astype(np.float32, copy=False)
    shape = tuple(int(x) for x in sparse["shape"])
    if shape[0] != len(rows):
        raise ValueError(f"Activation rows {shape[0]} do not match rows {len(rows)}")

    print("Sorting activations by feature...", flush=True)
    # Group activations by feature so each feature occupies one contiguous slice.
    order = np.argsort(feature_indices, kind="mergesort")
    sorted_features = feature_indices[order]
    sorted_rows = row_indices[order]
    sorted_values = values[order]
    bounds = np.r_[0, np.flatnonzero(sorted_features[1:] != sorted_features[:-1]) + 1, sorted_features.size]
    feature_ids = sorted_features[bounds[:-1]].astype(np.int64)
    print(f"Prepared {len(feature_ids):,} active features", flush=True)

    out_rows: list[dict[str, Any]] = []
    n = len(rows)
    for target_num, target in enumerate(targets, 1):
        # Construct this category's binary label vector.
        y = np.zeros(n, dtype=bool)
        y[target["positive_rows"]] = True
        total_pos = int(np.count_nonzero(y))
        train_pos = int(np.count_nonzero(y & train_mask))
        test_pos = int(np.count_nonzero(y & test_mask))
        train_neg = int(np.count_nonzero((~y) & train_mask))
        if train_pos == 0 or test_pos == 0:
            print(f"Skipping {target['category_id']}: train_pos={train_pos} test_pos={test_pos}", flush=True)
            continue

        # Search features and thresholds using training examples only.
        best: dict[str, Any] | None = None
        best_feature = -1
        for feature, start, end in zip(feature_ids, bounds[:-1], bounds[1:]):
            f_rows = sorted_rows[start:end]
            train_keep = train_mask[f_rows]
            if not np.any(train_keep):
                continue
            stats = best_for_feature(
                sorted_values[start:end][train_keep],
                y[f_rows[train_keep]],
                train_pos,
                train_neg,
                args.min_train_tp,
            )
            if stats is None:
                continue
            if best is None or (stats["train_f1"], stats["train_tp"], -stats["train_fp"]) > (
                best["train_f1"],
                best["train_tp"],
                -best["train_fp"],
            ):
                best = stats
                best_feature = int(feature)

        if best is None:
            continue
        # Apply the winning feature and fixed threshold to train and held-out test.
        pos = int(np.searchsorted(feature_ids, best_feature))
        start, end = int(bounds[pos]), int(bounds[pos + 1])
        f_rows = sorted_rows[start:end]
        f_values = sorted_values[start:end]
        train_metrics = metrics_at_threshold(f_rows, f_values, y, train_mask, best["threshold"])
        test_metrics = metrics_at_threshold(f_rows, f_values, y, test_mask, best["threshold"])
        out_rows.append(
            {
                "dataset": args.dataset,
                "model": args.model,
                "width": args.width,
                "top_k": args.top_k,
                "level": target["level"],
                "category_id": target["category_id"],
                "category": target["category"],
                "positive_count": total_pos,
                "train_pos": train_pos,
                "test_pos": test_pos,
                "best_feature": best_feature,
                "threshold": best["threshold"],
                "train_f1": train_metrics["f1"],
                "train_precision": train_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "train_tp": train_metrics["tp"],
                "train_fp": train_metrics["fp"],
                "train_fn": train_metrics["fn"],
                "test_f1": test_metrics["f1"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_tp": test_metrics["tp"],
                "test_fp": test_metrics["fp"],
                "test_fn": test_metrics["fn"],
            }
        )
        if target_num % 25 == 0 or target_num == len(targets):
            print(f"Finished {target_num}/{len(targets)} targets", flush=True)

    fields = [
        "dataset",
        "model",
        "width",
        "top_k",
        "level",
        "category_id",
        "category",
        "positive_count",
        "train_pos",
        "test_pos",
        "best_feature",
        "threshold",
        "train_f1",
        "train_precision",
        "train_recall",
        "train_tp",
        "train_fp",
        "train_fn",
        "test_f1",
        "test_precision",
        "test_recall",
        "test_tp",
        "test_fp",
        "test_fn",
    ]
    write_tsv(summary_path, out_rows, fields)
    meta = {
        "rows": str(rows_path),
        "sparse": str(sparse_path),
        "summary": str(summary_path),
        "dataset": args.dataset,
        "model": args.model,
        "width": args.width,
        "top_k": args.top_k,
        "min_count": args.min_count,
        "targets": len(targets),
        "written_targets": len(out_rows),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    print(json.dumps(meta, indent=2, sort_keys=True), flush=True)


