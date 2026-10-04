#!/usr/bin/env python3
"""Recompute the two a-prefix decoder matrices directly from SAE checkpoints."""
from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(_paper_path(__file__).resolve().parents[2]))
from project_paths import get_path
from scripts.hierarchy.run_a_prefix_linear_reconstruction import a_prefix_rows, read_summary, state_dict, vectors


def compute(models, probes, output):
    output.mkdir(parents=True, exist_ok=True)
    results, inputs = [], {}
    for width, k in [(32768, 64), (131072, 128)]:
        checkpoint = models / f"gemini_m{width}_k{k}"
        summary = probes / "wordfreq" / checkpoint.name / "summary.tsv"
        rows = a_prefix_rows(read_summary(summary))
        if len(rows) != 23:
            raise ValueError(f"Expected 23 a-prefix probes, got {len(rows)}")
        features = {}
        for row in rows:
            features.setdefault(str(int(row["best_feature"])), []).append(row["category"])
        ids = list(map(int, features))
        if len(ids) != 22:
            raise ValueError(f"Expected 22 unique features, got {len(ids)}")
        _, decoder = vectors(state_dict(checkpoint), ids)
        decoder = np.asarray(decoder, dtype=np.float64)
        norms = np.linalg.norm(decoder, axis=1)
        if not np.isfinite(decoder).all() or np.any(norms == 0):
            raise ValueError("Invalid decoder directions")
        normalized = decoder / norms[:, None]
        matrix = normalized @ normalized.T
        labels = [str(i) + " " + "/".join(features[str(i)]) for i in ids]
        with (output / f"gemini_{width}_cosines.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["feature", *labels])
            writer.writerows([label, *values] for label, values in zip(labels, matrix))
        values = matrix[np.triu_indices(len(ids), 1)]
        results.append(dict(width=width, top_k=k, features=features, median_cosine=float(np.median(values)),
                            unique_off_diagonal_pairs=len(values), categories=len(rows)))
        for path in [checkpoint, summary]:
            with path.open("rb") as stream:
                digest = hashlib.sha256()
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            inputs[str(path)] = digest.hexdigest()
    (output / "summary.json").write_text(json.dumps(dict(results=results, input_sha256=inputs,
        method="Signed cosine of stored decoder rows; selection-trained probes; no test-F1 filter; float64 arithmetic."), indent=2) + "\n")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--probe-results-dir", type=Path)
    parser.add_argument("--output", type=Path, default=_paper_path("full_experiments/results/a_prefix_probe_decoder_cosines"))
    args = parser.parse_args()
    print(json.dumps(compute(args.models_dir or _paper_path(get_path("models_dir")),
          args.probe_results_dir or _paper_path(get_path("hierarchy_data_dir")) / "probe_results", args.output), indent=2))


if __name__ == "__main__":
    main()
