"""Exact within-family, across-training-split KMeans activation matching.

Evaluate both dictionaries on the source split, using binary nearest-cluster
activations and maximum-cardinality matching at signed Pearson >= threshold.
Only full-embedding 16K models with completed training artifacts are included.
"""
from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import maximum_bipartite_matching

from scripts.stability.dense_matching_certificate import certify_maximum_batched

ROOT = _paper_path(__file__).resolve().parents[1]
OUT = ROOT / 'full_experiments/results/split_kmeans_raw_activation_matches'
SPLITS = ('main', 'wikipedia', 'random1')
FAMILIES = ('gemini', 'nemotron')
WIDTH = 16384
REFERENCE = ROOT / 'full_experiments/results/split_sae_activation_sketch512_candidates32_top5'


def read(path):
    return json.loads(_paper_path(path).read_text())


def write(path, value):
    path = _paper_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with _paper_path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare(args):
    from project_paths import get_path
    for family, dim in zip(FAMILIES, (3072, 4096)):
        models = []
        for split in SPLITS:
            if split == 'main':
                folder = _paper_path(get_path('models_dir')) / f'{family}_kmeans_raw_sample10m_iter50/k16384'
            else:
                run = ('wiki_trained_no_wiki_kmeans_raw_allrows_iter100' if split == 'wikipedia'
                       else 'random1_trained_random2_pearson')
                completion = read(ROOT / 'full_experiments/results' / run / 'models' / family / 'completed_100_iterations.json')
                folder = _paper_path(completion['model_dir'])
                assert completion['actual_iterations'] == 100
            metadata = read(folder / 'metadata.json')
            centers = folder / 'centroids.npy'
            c = np.load(_paper_location(centers), mmap_mode='r')
            assert c.shape == (WIDTH, dim) and np.isfinite(c).all()
            assert metadata['input_space'] == 'raw' and metadata['training_dim'] == dim
            assert metadata['actual_iterations'] == (50 if split == 'main' else 100)
            models.append(dict(split=split, path=str(centers), sha256=digest(centers),
                               training=metadata, metadata_sha256=digest(folder / 'metadata.json')))
        references = {s: read(REFERENCE / family / s / 'manifest.json') for s in SPLITS}
        shards = references['main']['shards']
        all_rows = {s['relative_shard']: s['rows'] for s in shards}
        evaluations = {}
        for split, reference in references.items():
            names = [s['relative_shard'] for s in reference['shards']]
            assert len(set(names)) == len(names)
            assert all(all_rows[s['relative_shard']] == s['rows'] for s in reference['shards'])
            assert sum(all_rows[n] for n in names) == reference['rows']
            evaluations[split] = dict(shards=names, rows=reference['rows'], datasets=reference['datasets'],
                                      reference=str(REFERENCE / family / split / 'manifest.json'),
                                      reference_sha256=digest(REFERENCE / family / split / 'manifest.json'))
        manifest = dict(family=family, width=WIDTH, dimension=dim, models=models, shards=shards,
                        embeddings_dir=str(get_path(f'{family}_embeddings_dir')), evaluations=evaluations,
                        rows=sum(all_rows.values()), tasks=args.tasks, threshold=args.threshold,
                        assignment='float32 squared Euclidean distance; TF32 disabled; lowest-index ties',
                        matching='maximum one-to-one cardinality at signed Pearson >= threshold',
                        evaluation='both models on source training distribution; all cached rows',
                        candidate_limited=False, source_code_sha256=digest(__file__))
        manifest['id'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        path = args.output / family / 'manifest.json'
        if path.exists():
            assert read(path) == manifest, 'Manifest changed; use a new output directory'
        write(path, manifest)
        print(f'Prepared {family}: {len(shards)} shards, {manifest["rows"]:,} rows', flush=True)


def assign(values, centers, squared_norms):
    # Same rule as match_cross_model_baseline_activations.assign_kmeans.
    scores = values @ centers.T
    scores.mul_(2).sub_(squared_norms)
    return scores.argmax(dim=1)


def compute(args):
    import torch
    manifest = read(args.output / args.family / 'manifest.json')
    assert 0 <= args.task < manifest['tasks']
    assert digest(__file__) == manifest['source_code_sha256']
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    centers = []
    for model in manifest['models']:
        assert digest(model['path']) == model['sha256']
        centers.append(torch.from_numpy(np.load(_paper_location(model['path']))).to(device))
    norms = [c.square().sum(1) for c in centers]
    assigned = manifest['shards'][args.task::manifest['tasks']]
    started = time.monotonic()
    for index, shard in enumerate(assigned):
        folder = args.output / args.family / 'assignments' / shard['relative_shard']
        record_path = folder / 'complete.json'
        if record_path.exists():
            record = read(record_path)
            assert record['manifest_id'] == manifest['id'] and record['shard'] == shard
            assert digest(folder / 'labels.npy') == record['labels_sha256']
            continue
        embedding_path = _paper_path(manifest['embeddings_dir']) / shard['relative_shard'] / 'embeddings.npy'
        stat = embedding_path.stat()
        x = np.load(_paper_location(embedding_path), mmap_mode='r')
        assert x.shape == (shard['rows'], manifest['dimension'])
        labels = np.empty((shard['rows'], len(centers)), dtype=np.uint16)
        with torch.inference_mode():
            for start in range(0, len(x), args.batch_rows):
                stop = min(start + args.batch_rows, len(x))
                batch = torch.from_numpy(np.array(x[start:stop], dtype=np.float32)).to(device)
                assert torch.isfinite(batch).all(), embedding_path
                for column, (c, norm) in enumerate(zip(centers, norms)):
                    labels[start:stop, column] = assign(batch, c, norm).cpu().numpy()
        assert embedding_path.stat().st_mtime_ns == stat.st_mtime_ns
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / f'labels.{os.getpid()}.tmp.npy'
        np.save(_paper_location(tmp), labels)
        tmp.replace(folder / 'labels.npy')
        write(record_path, dict(manifest_id=manifest['id'], shard=shard,
                                labels_sha256=digest(folder / 'labels.npy'),
                                embedding_path=str(embedding_path), embedding_size=stat.st_size,
                                embedding_mtime_ns=stat.st_mtime_ns))
        print(f'{args.family} task {args.task}: {index+1}/{len(assigned)} shards; '
              f'{time.monotonic()-started:.1f}s', flush=True)
    write(args.output / args.family / 'tasks' / f'{args.task:03d}.json',
          dict(complete=True, manifest_id=manifest['id'], task=args.task, shards=assigned))


def threshold_graph(counts, threshold):
    """Exhaustive positive-threshold graph; absent cooccurrences have r <= 0."""
    assert 0 < threshold <= 1
    counts = counts.tocsr()
    left = np.asarray(counts.sum(axis=1)).ravel().astype(np.float64)
    right = np.asarray(counts.sum(axis=0)).ravel().astype(np.float64)
    n = float(left.sum())
    assert n > 0 and n == right.sum()
    coo = counts.tocoo()
    lv = left - left * left / n
    rv = right - right * right / n
    denom = np.sqrt(lv[coo.row] * rv[coo.col])
    correlation = np.full(coo.nnz, np.nan)
    np.divide(coo.data - left[coo.row] * right[coo.col] / n, denom,
              out=correlation, where=denom > 0)
    correlation = np.clip(correlation, -1, 1)
    keep = np.isfinite(correlation) & (correlation >= threshold)
    graph = sparse.csr_matrix((correlation[keep], (coo.row[keep], coo.col[keep])), shape=counts.shape)
    return graph, left.astype(np.int64), right.astype(np.int64)


def reduce(args):
    rows = []
    for family in FAMILIES:
        manifest = read(args.output / family / 'manifest.json')
        for task in range(manifest['tasks']):
            record = read(args.output / family / 'tasks' / f'{task:03d}.json')
            assert record['complete'] and record['manifest_id'] == manifest['id']
            assert record['shards'] == manifest['shards'][task::manifest['tasks']]
        # Three labels per example, not a dense N-by-K activation matrix.
        labels = np.empty((manifest['rows'], len(SPLITS)), dtype=np.uint16)
        membership = np.zeros((manifest['rows'], len(SPLITS)), dtype=bool)
        name_sets = [set(manifest['evaluations'][s]['shards']) for s in SPLITS]
        offset = 0
        for shard in manifest['shards']:
            folder = args.output / family / 'assignments' / shard['relative_shard']
            record = read(folder / 'complete.json')
            assert record['manifest_id'] == manifest['id'] and record['shard'] == shard
            assert digest(folder / 'labels.npy') == record['labels_sha256']
            embedding_stat = _paper_path(record['embedding_path']).stat()
            assert embedding_stat.st_size == record['embedding_size']
            assert embedding_stat.st_mtime_ns == record['embedding_mtime_ns']
            a = np.load(_paper_location(folder / 'labels.npy'))
            assert a.shape == (shard['rows'], len(SPLITS)) and a.max() < manifest['width']
            end = offset + len(a)
            labels[offset:end] = a
            membership[offset:end] = [shard['relative_shard'] in names for names in name_sets]
            offset = end
        assert offset == len(labels)
        for i, source in enumerate(SPLITS):
            selected = labels[membership[:, i]]
            n = len(selected)
            assert n == manifest['evaluations'][source]['rows']
            for j, target in enumerate(SPLITS):
                if i == j:
                    continue
                # Integer contingency counts retain every example exactly once.
                counts = sparse.coo_matrix((np.ones(n, dtype=np.int64),
                                             (selected[:, i].astype(np.int32), selected[:, j].astype(np.int32))),
                                            shape=(manifest['width'], manifest['width'])).tocsr()
                graph, left, right = threshold_graph(counts, manifest['threshold'])
                assert int(counts.sum()) == int(left.sum()) == int(right.sum()) == n
                assignment = maximum_bipartite_matching(graph, perm_type='column')
                src = np.flatnonzero(assignment >= 0)
                dst = assignment[src]
                certify_maximum_batched(graph, src, dst)
                scores = np.asarray(graph[src, dst]).ravel() if len(src) else np.empty(0)
                assert np.all(scores >= manifest['threshold'])
                folder = args.output / family / source
                folder.mkdir(parents=True, exist_ok=True)
                sparse.save_npz(folder / f'contingency_to_{target}.npz', counts)
                sparse.save_npz(folder / f'graph_to_{target}.npz', graph)
                np.savez(_paper_location(folder / f'matching_to_{target}.npz'), source=src, destination=dst, pearson=scores,
                         source_counts=left, target_counts=right, rows=n)
                row = dict(family=family, source_split=source, comparison_split=target,
                           evaluation_rows=n, source_features=manifest['width'], target_features=manifest['width'],
                           threshold=manifest['threshold'], matches=len(src), proportion=len(src)/manifest['width'],
                           threshold_edges=graph.nnz, source_zero_variance=int(((left == 0) | (left == n)).sum()),
                           target_zero_variance=int(((right == 0) | (right == n)).sum()),
                           maximum_matching_certified=True, candidate_limited=False)
                rows.append(row)
                print(row, flush=True)
                del counts
            del selected
        del labels, membership
    with (args.output / 'counts.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write(args.output / 'summary.json', dict(complete=True, comparisons=rows, splits=SPLITS,
          unavailable_training_splits=['no_wikipedia', 'random2'],
          validation='All shard assignments and model hashes, complete row coverage, integer contingency marginals, '
                     'threshold edges, one-to-one pairs, and full-graph minimum vertex-cover certificates checked.'))
    plot(args.output, rows)


def plot(output, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), layout='constrained')
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('#eeeeee')
    for ax, family in zip(axes, FAMILIES):
        matrix = np.full((3, 3), np.nan)
        for row in rows:
            if row['family'] == family:
                matrix[SPLITS.index(row['source_split']), SPLITS.index(row['comparison_split'])] = row['proportion']
        im = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(3), ['Full', 'Wikipedia', 'Random1'])
        ax.set_yticks(range(3), ['Full', 'Wikipedia', 'Random1'])
        ax.set_title(f'{family.capitalize()} — 16K KMeans')
        for row in rows:
            if row['family'] == family:
                ax.text(SPLITS.index(row['comparison_split']), SPLITS.index(row['source_split']),
                        f'{row["matches"]:,}\n({row["proportion"]:.1%})', ha='center', va='center',
                        color='white' if row['proportion'] < .5 else 'black')
    axes[0].set_ylabel('Source training and evaluation split')
    fig.supxlabel('Comparison model training split')
    fig.colorbar(im, ax=axes, fraction=.03, pad=.02, format=PercentFormatter(1),
                 label=f'Matched features (Pearson ≥ {rows[0]["threshold"]:g})')
    for ext in ['png', 'pdf']:
        fig.savefig(output / f'activation_split_heatmaps_t{rows[0]["threshold"]:g}.{ext}', dpi=220)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'compute', 'reduce'])
    p.add_argument('--output', type=Path, default=OUT)
    p.add_argument('--family', choices=FAMILIES)
    p.add_argument('--task', type=int, default=0)
    p.add_argument('--tasks', type=int, default=8)
    p.add_argument('--threshold', type=float, default=.7)
    p.add_argument('--batch-rows', type=int, default=2048)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    assert args.tasks > 0 and 0 < args.threshold <= 1
    globals()[args.stage](args)


if __name__ == '__main__':
    main()
