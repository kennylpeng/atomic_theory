#!/usr/bin/env python3
"""Fixed-k prefix recovery using the established threshold protocol and extra SAEs."""
from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from numba import njit
from hierarchy_probe_computation import (best_for_feature, build_targets,
    metrics_at_threshold, read_rows, split_masks, write_tsv)
from hierarchy_probe_config import HIERARCHY_DATA_DIR
from scripts.control_models import FIXED as FIXED_SEEDS, checkpoint_path

ROOT = _paper_path(__file__).resolve().parents[2]
OUT = ROOT / 'full_experiments/results/prefix_recovery_k128'
CACHE = _paper_path('/resources/scratch_dir/prefix_recovery_k128')
WIDTHS = [512, 4096, 32768, 65536, 131072]


def sha(path):
    h = hashlib.sha256()
    with _paper_path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


@njit(cache=True)
def select_rule(rows, values, bounds, labels, total_pos):
    """Inputs sorted by feature then decreasing value; same tie rules as reference."""
    best_f1, best_tp, best_fp, best_feature, best_threshold = -1., -1, 0, -1, 0.
    for feature in range(len(bounds) - 1):
        begin, end = bounds[feature], bounds[feature + 1]
        tp = 0
        local_f1, local_tp, local_fp, local_threshold = -1., -1, 0, 0.
        for i in range(begin, end):
            tp += int(labels[rows[i]])
            if i + 1 < end and values[i] == values[i + 1]:
                continue
            if tp < 1:
                continue
            predicted = i - begin + 1
            f1 = 2. * tp / (predicted + total_pos)
            # np.argmax in the reference retains the first threshold on a tie.
            if f1 > local_f1:
                local_f1, local_tp, local_fp = f1, tp, predicted - tp
                local_threshold = values[i]
        if local_tp < 0:
            continue
        if (local_f1 > best_f1 or
            (local_f1 == best_f1 and (local_tp > best_tp or
             (local_tp == best_tp and local_fp < best_fp)))):
            best_f1, best_tp, best_fp = local_f1, local_tp, local_fp
            best_feature, best_threshold = feature, local_threshold
    return best_feature, best_threshold, best_f1, best_tp, best_fp


def prepare(rr, ff, vv, train, width):
    keep = train[rr]
    tr, tf, tv = rr[keep], ff[keep], vv[keep]
    order = np.lexsort((-tv, tf))
    bounds = np.r_[0, np.cumsum(np.bincount(tf, minlength=width))]
    return tr[order], tv[order], bounds


def validate_search():
    rng = np.random.default_rng(192)
    for trial in range(100):
        n, width = 100, 11
        dense = rng.integers(0, 8, size=(n, width)).astype(np.float32)
        dense[rng.random(dense.shape) < .7] = 0
        y = rng.random(n) < .25
        train = rng.random(n) < .8
        rr, ff = np.nonzero(dense)
        vv = dense[rr, ff]
        prepared = prepare(rr, ff, vv, train, width)
        got = select_rule(*prepared, y, int(y[train].sum()))
        expected = None
        for feature in range(width):
            active = train & (dense[:, feature] > 0)
            stats = best_for_feature(dense[active, feature], y[active], int(y[train].sum()), int((~y[train]).sum()), 1)
            if stats is None:
                continue
            key = stats['train_f1'], stats['train_tp'], -stats['train_fp']
            if expected is None or key > expected[0]:
                expected = key, feature, stats['threshold']
        assert got == (expected[1], expected[2], expected[0][0], expected[0][1], -expected[0][2]), (trial, got, expected)
    print('Reference threshold-search agreement: 100 randomized cases passed.', flush=True)


def probe(family, width, top_k, sparse_path, folder, compare=None, dataset="wordfreq"):
    rows_path = HIERARCHY_DATA_DIR / dataset / 'rows.csv'
    rows = read_rows(rows_path)
    train, test = split_masks(rows)
    targets = build_targets(dataset, rows, 100)
    with np.load(_paper_location(sparse_path)) as z:
        rr, ff, vv = z['row_indices'], z['feature_indices'], z['values'].astype(np.float32)
        assert tuple(z['shape']) == (len(rows), width)
        assert int(z['top_k'][0]) == top_k
    assert np.isfinite(vv).all() and (vv >= 0).all()
    # Include any values rounded to zero, matching the existing stored-value protocol.
    prepared = prepare(rr, ff, vv, train, width)
    print(f'{family} width={width}: {len(rows)} rows, {len(targets)} targets', flush=True)
    records = []
    for i, target in enumerate(targets):
        y = np.zeros(len(rows), dtype=np.bool_)
        y[target['positive_rows']] = True
        train_pos, test_pos = int(y[train].sum()), int(y[test].sum())
        if not train_pos or not test_pos:
            continue
        feature, threshold, train_f1, tp, fp = select_rule(*prepared, y, train_pos)
        assert feature >= 0
        keep = ff == feature
        row = dict(dataset=dataset, model=family, width=width, top_k=top_k,
                   **{k: v for k, v in target.items() if k != 'positive_rows'},
                   train_pos=train_pos, test_pos=test_pos, best_feature=int(feature), threshold=float(threshold))
        row['positive_count'] = int(y.sum())
        for split, mask in [('train', train), ('test', test)]:
            metrics = metrics_at_threshold(rr[keep], vv[keep], y, mask, threshold)
            row.update({f'{split}_{key}': metrics[key] for key in ('f1', 'precision', 'recall', 'tp', 'fp', 'fn')})
        assert row['train_f1'] == train_f1 and row['train_tp'] == tp and row['train_fp'] == fp
        records.append(row)
        if (i + 1) % 25 == 0:
            print(f'Finished {i+1}/{len(targets)}', flush=True)
    if compare:
        with _paper_path(compare).open() as f:
            baseline = list(csv.DictReader(f, delimiter='\t'))
        assert len(baseline) == len(records)
        by_id = {r['category_id']: r for r in baseline}
        for row in records:
            for key, value in row.items():
                previous = by_id[row['category_id']][key]
                assert (float(previous) == value if isinstance(value, (int, float)) else previous == value), (row['category_id'], key, previous, value)
        print(f'Exact agreement with all {len(records)} existing baseline probe rows.', flush=True)
    folder.mkdir(parents=True, exist_ok=True)
    write_tsv(folder / 'summary.tsv', records, list(records[0]))
    metadata = dict(rows=str(rows_path), rows_sha256=sha(rows_path), sparse=str(sparse_path),
        dataset=dataset, model=family, width=width, top_k=top_k, min_count=100, min_train_tp=1,
        targets=len(targets), written_targets=len(records), slurm_job=os.environ.get('SLURM_JOB_ID'),
        protocol='Original train-only feature/threshold maximization; inclusive threshold; unchanged held-out test split.',
        script_sha256=sha(__file__))
    (folder / 'meta.json').write_text(json.dumps(metadata, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', type=int, choices=range(6))
    p.add_argument('--validate', action='store_true')
    args = p.parse_args()
    validate_search()
    if args.validate:
        probe('gemini', 512, 32, HIERARCHY_DATA_DIR / 'wordfreq/sparse_gemini_m512_k32.npz',
              OUT / 'validation/gemini_m512_k32', HIERARCHY_DATA_DIR / 'probe_results/wordfreq/gemini_m512_k32/summary.tsv')
        return
    if args.task is None:
        p.error('--task or --validate is required')
    family, width = [(f, w) for f in FIXED_SEEDS for w in FIXED_SEEDS[f]][args.task]
    seed = FIXED_SEEDS[family][width]
    checkpoint = checkpoint_path(family, width, 128, seed)
    import torch
    payload = torch.load(_paper_location(checkpoint), map_location='cpu', mmap=True, weights_only=False)
    config = payload['config']
    assert config['top_k'] == 128 and config['dict_size'] == width
    assert not config.get('input_unit_norm', False)
    del payload
    sparse = CACHE / 'wordfreq' / f'sparse_{family}_m{width}_k128.npz'
    subprocess.run([sys.executable, str(ROOT / 'hierarchy_data_prep/02_compute_sparse_activations.py'),
        '--model-path', str(checkpoint), '--embeddings', str(HIERARCHY_DATA_DIR / f'wordfreq/embeddings_{family}.npy'),
        '--rows', str(HIERARCHY_DATA_DIR / 'wordfreq/rows.csv'), '--output', str(sparse),
        '--top-k', '128', '--batch-size', '2048', '--device', 'cuda', '--dtype', 'float32'], check=True)
    folder = OUT / 'wordfreq' / f'{family}_m{width}_k128'
    probe(family, width, 128, sparse, folder)
    manifest = dict(path=str(checkpoint), sha256=sha(checkpoint), config=config,
                    sparse=str(sparse), sparse_sha256=sha(sparse))
    (folder / 'checkpoint.json').write_text(json.dumps(manifest, indent=2, default=str) + '\n')
    print(f'COMPLETE: {folder}', flush=True)


if __name__ == '__main__':
    main()
