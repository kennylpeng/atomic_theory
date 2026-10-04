"""Shared download, hashing, CSV, and deterministic-split helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def download(url: str, path: Path) -> Path:
    """Download ``url`` unless ``path`` already exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"Downloading {url} -> {path}", flush=True)
        urllib.request.urlretrieve(url, path)
    return path


def stable_id(namespace: str, text: str, length: int = 24) -> str:
    return hashlib.sha1(f"{namespace}\0{text}".encode("utf-8")).hexdigest()[:length]


def stratified_half_split(keys: Iterable[str], seed: int) -> list[str]:
    """Split each key group in half using stable hash ordering.

    Odd groups put the extra example in train. Hash ordering makes the result
    independent of Python, NumPy, and input-row ordering.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    keys = list(keys)
    for index, key in enumerate(keys):
        groups[key].append(index)
    result = [""] * len(keys)
    for key, indices in groups.items():
        ordered = sorted(
            indices,
            key=lambda i: hashlib.sha256(f"{seed}\0{key}\0{i}".encode()).digest(),
        )
        n_test = len(ordered) // 2
        for index in ordered[:n_test]:
            result[index] = "test"
        for index in ordered[n_test:]:
            result[index] = "train"
    return result


def write_rows(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_meta(path: Path, data: dict[str, object]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
