"""Select and resolve artwork examples for curated color features."""

from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import csv

import heapq

from pathlib import Path

import numpy as np

DEFAULT_RESULTS = _paper_path(
    "full_experiments/results/wikiart_feature_enrichment/"
    "gemini_m131072_k128_full3072"
)

DEFAULT_INDEX = _paper_path(
    "/resources/wikiart_dir/runs/full_batch_3072/embeddings_index.csv"
)

def load_metadata(path: Path, expected_rows: int) -> list[dict[str, str]]:
    records: list[dict[str, str] | None] = [None] * expected_rows
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            vector_index = int(row["vector_index"])
            if not 0 <= vector_index < expected_rows:
                raise ValueError(f"vector_index out of range: {vector_index}")
            if records[vector_index] is not None:
                raise ValueError(f"duplicate vector_index: {vector_index}")
            records[vector_index] = row
    if any(row is None for row in records):
        raise ValueError("Embedding index does not cover every activation row")
    return [row for row in records if row is not None]

def collect_top_images(
    indices: np.ndarray,
    values: np.ndarray,
    feature_ids: list[int],
    images_per_feature: int,
) -> dict[int, list[tuple[float, int]]]:
    """Select exact top activations in one chunked pass over sparse TopK data."""
    selected = np.asarray(feature_ids, dtype=np.int32)
    heaps: dict[int, list[tuple[float, int]]] = {
        feature_id: [] for feature_id in feature_ids
    }
    for row_start in range(0, indices.shape[0], 4096):
        row_stop = min(row_start + 4096, indices.shape[0])
        chunk_indices = np.asarray(indices[row_start:row_stop])
        chunk_values = np.asarray(values[row_start:row_stop])
        mask = np.isin(chunk_indices, selected) & (chunk_values > 0)
        local_rows, positions = np.nonzero(mask)
        for local_row, position in zip(local_rows.tolist(), positions.tolist()):
            feature_id = int(chunk_indices[local_row, position])
            item = (
                float(chunk_values[local_row, position]),
                row_start + local_row,
            )
            heap = heaps[feature_id]
            if len(heap) < images_per_feature:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
    return {
        feature_id: sorted(heap, reverse=True)
        for feature_id, heap in heaps.items()
    }
