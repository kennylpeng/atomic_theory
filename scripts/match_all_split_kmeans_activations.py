"""Exact activation matching for five fresh full-dimensional 16K KMeans dictionaries."""
from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import maximum_bipartite_matching

from scripts import match_split_kmeans_activations as base
from scripts import train_kmeans_full_corpus as common

TRAINING = common.ROOT / 'full_experiments/results/kmeans_raw_all_splits_iter20'
OUT = TRAINING / 'activation_similarity_k16384'
SPLITS = ('main', 'wikipedia', 'no_wikipedia', 'random1', 'random2')
LABELS = ('Full', 'Wikipedia', 'No Wikipedia', 'Random1', 'Random2')
FAMILIES = ('gemini', 'nemotron')
WIDTH = 16384


def prepare(args):
    from project_paths import get_path
    catalog = common.read(TRAINING / 'summary.json')
    assert catalog['complete'] and catalog['iterations'] == 20
    for family in FAMILIES:
        main_path = common.OUT / family / 'plan.json'
        main = common.read(main_path)
        models, evaluations = [], {}
        for split in SPLITS:
            output = common.OUT if split == 'main' else TRAINING / split
            plan_path = output / family / 'plan.json'
            plan = common.read(plan_path)
            model = output / 'models' / family / f'k{WIDTH}'
            done, meta = common.read(model / 'completed.json'), common.read(model / 'metadata.json')
            assert done['complete'] and done['plan_id'] == meta['plan_id'] == plan['id']
            assert meta['actual_iterations'] == len(meta['objective']) == 20
            assert meta['from_scratch'] and meta['initial_centroids'] is None
            assert meta['sample_rows'] == plan['rows'] and meta['all_corpus_rows']
            assert not meta['faiss_subsampling'] and meta['training_split'] == split
            assert meta['input_space'] == 'raw' and meta['training_dim'] == main['dimension']
            assert common.digest(model / 'metadata.json') == done['metadata_sha256']
            centers = model / 'centroids.npy'
            assert common.digest(centers) == done['centroids_sha256']
            matrix = np.load(_paper_location(centers), mmap_mode='r')
            assert matrix.shape == (WIDTH, main['dimension']) and np.isfinite(matrix).all()
            matrix._mmap.close()
            models.append(dict(split=split, path=str(centers), sha256=done['centroids_sha256'],
                               training=meta, metadata_sha256=done['metadata_sha256']))
            evaluations[split] = dict(shards=[s['relative_shard'] for s in plan['shards']],
                                      rows=plan['rows'], reference=str(plan_path),
                                      reference_sha256=common.digest(plan_path))
        full_names = {s['relative_shard']: s['rows'] for s in main['shards']}
        for evaluation in evaluations.values():
            names = evaluation['shards']
            assert len(names) == len(set(names)) and set(names) <= set(full_names)
            assert sum(full_names[name] for name in names) == evaluation['rows']
        for shard in main['shards']:
            common.stat_matches(shard['path'], shard['size'], shard['mtime_ns'])
        manifest = dict(family=family, width=WIDTH, dimension=main['dimension'], models=models,
                        splits=list(SPLITS), evaluations=evaluations, shards=main['shards'],
                        rows=main['rows'], tasks=args.tasks, threshold=args.threshold,
                        embeddings_dir=str(get_path(f'{family}_embeddings_dir')),
                        assignment='Full float32 squared Euclidean search; TF32 disabled; lowest-index ties',
                        activation='Binary indicator of nearest centroid',
                        matching='Exhaustive maximum one-to-one cardinality at signed Pearson >= threshold',
                        evaluation='Both models on the row/source training distribution; every row',
                        candidate_limited=False, source_code_sha256=common.digest(base.__file__),
                        analysis_code_sha256=common.digest(__file__))
        manifest['id'] = common.object_digest(manifest)
        path = args.output / family / 'manifest.json'
        if path.exists():
            assert common.read(path) == manifest
        else:
            common.write(path, manifest)
        print(f'Prepared {family}: five dictionaries, {main["rows"]:,} rows, {args.tasks} tasks', flush=True)


def load_manifest(args):
    manifest = common.read(args.output / args.family / 'manifest.json')
    assert manifest['analysis_code_sha256'] == common.digest(__file__)
    assert manifest['source_code_sha256'] == common.digest(base.__file__)
    assert manifest['splits'] == list(SPLITS)
    return manifest


def compute(args):
    manifest = load_manifest(args)
    assert 0 <= args.task < manifest['tasks']
    for shard in manifest['shards'][args.task::manifest['tasks']]:
        common.stat_matches(shard['path'], shard['size'], shard['mtime_ns'])
    # Reuse the previously validated exhaustive GPU assignment implementation.
    base.compute(args)


def load_labels(args, manifest):
    for task in range(manifest['tasks']):
        record = common.read(args.output / args.family / 'tasks' / f'{task:03d}.json')
        assert record['complete'] and record['manifest_id'] == manifest['id']
        assert record['shards'] == manifest['shards'][task::manifest['tasks']]
    names = set(manifest['evaluations'][args.source]['shards'])
    shards = [s for s in manifest['shards'] if s['relative_shard'] in names]
    n = manifest['evaluations'][args.source]['rows']
    labels = np.empty((n, len(SPLITS)), dtype=np.uint16)
    offset = 0
    for shard in shards:
        folder = args.output / args.family / 'assignments' / shard['relative_shard']
        record = common.read(folder / 'complete.json')
        assert record['manifest_id'] == manifest['id'] and record['shard'] == shard
        assert record['embedding_path'] == shard['path']
        assert record['embedding_size'] == shard['size']
        assert record['embedding_mtime_ns'] == shard['mtime_ns']
        common.stat_matches(shard['path'], shard['size'], shard['mtime_ns'])
        assert common.digest(folder / 'labels.npy') == record['labels_sha256']
        part = np.load(_paper_location(folder / 'labels.npy'), allow_pickle=False)
        assert part.shape == (shard['rows'], len(SPLITS)) and part.dtype == np.uint16
        assert part.max() < WIDTH
        labels[offset:offset+len(part)] = part
        offset += len(part)
    assert offset == n and len(shards) == len(names)
    return labels


def assignment_audit(args, manifest):
    """Independently check representative cached assignments against float64 distances."""
    chosen = set(np.linspace(0, len(manifest['shards'])-1, 16, dtype=int).tolist())
    seen = set()
    for index, shard in enumerate(manifest['shards']):
        dataset = shard['relative_shard'].split('/')[0]
        if dataset not in seen:
            chosen.add(index)
            seen.add(dataset)
    vectors, observed = [], []
    for index in sorted(chosen):
        shard = manifest['shards'][index]
        sample = np.load(_paper_location(shard['path']), mmap_mode='r')
        rows = np.unique([shard['rows']//3, (2*shard['rows'])//3])
        vectors.append(np.array(sample[rows], dtype=np.float64))
        sample._mmap.close()
        path = args.output / args.family / 'assignments' / shard['relative_shard'] / 'labels.npy'
        observed.append(np.load(_paper_location(path))[rows])
    x, ids = np.concatenate(vectors), np.concatenate(observed)
    checks = []
    for column, model in enumerate(manifest['models']):
        centers = np.load(_paper_location(model['path'])).astype(np.float64)
        scores = 2*x @ centers.T - np.einsum('ij,ij->i', centers, centers)
        reference = scores.argmax(axis=1)
        gaps = scores[np.arange(len(x)), reference] - scores[np.arange(len(x)), ids[:, column]]
        assert gaps.max() <= 1e-5, (model['split'], gaps.max())
        checks.append(dict(split=model['split'], checked=len(x), exact_labels=int((reference == ids[:, column]).sum()),
                           max_float64_score_gap=float(gaps.max()), tolerance=1e-5))
    common.write(args.output / args.family / 'assignment_validation.json',
                 dict(passed=True, manifest_id=manifest['id'], datasets=sorted(seen), checks=checks))


def match_labels(labels, source_column, target_column, width, threshold):
    n = len(labels)
    counts = sparse.coo_matrix((np.ones(n, dtype=np.int64),
              (labels[:, source_column].astype(np.int32), labels[:, target_column].astype(np.int32))),
              shape=(width, width)).tocsr()
    graph, left, right = base.threshold_graph(counts, threshold)
    assert int(counts.sum()) == int(left.sum()) == int(right.sum()) == n
    assignment = maximum_bipartite_matching(graph, perm_type='column')
    source = np.flatnonzero(assignment >= 0)
    destination = assignment[source]
    base.certify_maximum_batched(graph, source, destination)
    scores = np.asarray(graph[source, destination]).ravel() if len(source) else np.empty(0)
    assert np.all(scores >= threshold)
    return counts, graph, source, destination, scores, left, right


def reduce(args):
    manifest = load_manifest(args)
    folder = args.output / args.family / args.source
    summary_path = folder / 'summary.json'
    if summary_path.exists():
        summary = common.read(summary_path)
        assert summary['complete'] and summary['manifest_id'] == manifest['id']
        return
    for model in manifest['models']:
        assert common.digest(model['path']) == model['sha256']
    labels = load_labels(args, manifest)
    if args.source == 'main':
        assignment_audit(args, manifest)
    folder.mkdir(parents=True, exist_ok=True)
    source_column = SPLITS.index(args.source)
    started, results = time.monotonic(), []
    for target_column, target in enumerate(SPLITS):
        if target == args.source:
            continue
        counts, graph, source, destination, scores, left, right = match_labels(
            labels, source_column, target_column, WIDTH, manifest['threshold'])
        sparse.save_npz(folder / f'contingency_to_{target}.npz', counts)
        sparse.save_npz(folder / f'graph_to_{target}.npz', graph)
        np.savez(_paper_location(folder / f'matching_to_{target}.npz'), source=source, destination=destination,
                 pearson=scores, source_counts=left, target_counts=right, rows=len(labels))
        record = dict(family=args.family, source_split=args.source, comparison_split=target,
                      evaluation_split=args.source, evaluation_rows=len(labels),
                      source_features=WIDTH, target_features=WIDTH, threshold=manifest['threshold'],
                      matches=len(source), proportion=len(source)/WIDTH, threshold_edges=graph.nnz,
                      source_zero_variance=int(((left == 0) | (left == len(labels))).sum()),
                      target_zero_variance=int(((right == 0) | (right == len(labels))).sum()),
                      maximum_matching_certified=True, candidate_limited=False)
        results.append(record)
        print(record, 'elapsed', round(time.monotonic()-started, 1), flush=True)
        del counts, graph
    common.write(summary_path, dict(complete=True, manifest_id=manifest['id'], comparisons=results))


def plot_heatmap(output, records, threshold):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'font.size': 11, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.7), layout='constrained')
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('#eeeeee')
    vmax = min(1., max(.1, np.ceil(max(row['proportion'] for row in records)*10)/10))
    for ax, family in zip(axes, FAMILIES):
        matrix = np.full((len(SPLITS), len(SPLITS)), np.nan)
        counts = np.full_like(matrix, np.nan)
        for row in records:
            if row['family'] == family:
                i, j = SPLITS.index(row['source_split']), SPLITS.index(row['comparison_split'])
                assert np.isnan(matrix[i, j])
                matrix[i, j], counts[i, j] = row['proportion'], row['matches']
        assert np.isfinite(matrix).sum() == 20 and np.isnan(np.diag(matrix)).all()
        im = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=0, vmax=vmax)
        ax.set_xticks(range(5), LABELS, rotation=35, ha='right')
        ax.set_yticks(range(5), LABELS)
        ax.set_title(f'{family.capitalize()} · 16,384-cluster KMeans', pad=12, fontsize=14)
        for i in range(5):
            for j in range(5):
                if i == j:
                    ax.text(j, i, '—', ha='center', va='center', color='#999999')
                    continue
                r, g, b, _ = cmap(im.norm(matrix[i, j]))
                color = 'black' if .2126*r + .7152*g + .0722*b > .55 else 'white'
                ax.text(j, i, f'{int(counts[i,j]):,}\n({matrix[i,j]:.1%})',
                        ha='center', va='center', color=color, fontsize=10.5)
        for name, values in [('counts', counts), ('proportions', matrix)]:
            with (output / f'{family}_{name}_matrix.csv').open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['source/evaluation_split', *SPLITS])
                for split, row in zip(SPLITS, values):
                    writer.writerow([split, *['' if np.isnan(value) else value for value in row]])
    axes[0].set_ylabel('Source model training distribution\nand evaluation distribution')
    fig.supxlabel('Comparison model training distribution', fontsize=12)
    bar = fig.colorbar(im, ax=axes, fraction=.032, pad=.025, format=PercentFormatter(1))
    bar.set_label(f'Matched clusters (Pearson ≥ {threshold:g})')
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(output / f'activation_split_heatmaps_t{threshold:g}.{extension}',
                    dpi=240, bbox_inches='tight', pad_inches=.08)
    plt.close(fig)
    return dict(color_min=0, color_max=vmax, colormap='viridis', diagonal='omitted self-comparisons')


def finish(args):
    records, manifest_ids = [], {}
    for family in FAMILIES:
        args.family = family
        manifest = load_manifest(args)
        manifest_ids[family] = manifest['id']
        audit = common.read(args.output / family / 'assignment_validation.json')
        assert audit['passed'] and audit['manifest_id'] == manifest['id']
        for source in SPLITS:
            summary = common.read(args.output / family / source / 'summary.json')
            assert summary['complete'] and summary['manifest_id'] == manifest['id']
            assert len(summary['comparisons']) == 4
            for row in summary['comparisons']:
                assert row['family'] == family and row['source_split'] == source
                assert row['evaluation_rows'] == manifest['evaluations'][source]['rows']
                assert row['maximum_matching_certified'] and not row['candidate_limited']
                assert row['threshold'] == manifest['threshold']
                records.append(row)
    assert len(records) == len({(r['family'], r['source_split'], r['comparison_split']) for r in records}) == 40
    threshold = records[0]['threshold']
    assert all(r['threshold'] == threshold for r in records)
    with (args.output / 'counts.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    plot_metadata = plot_heatmap(args.output, records, threshold)
    common.write(args.output / 'summary.json', dict(complete=True, splits=list(SPLITS), comparisons=records,
                 threshold=threshold, manifest_ids=manifest_ids, plot=plot_metadata,
                 completed_utc=datetime.now(timezone.utc).isoformat(),
                 validation='All rows, hashes, integer marginals, full threshold graphs, matching certificates, '
                            'and independent float64 assignment spot checks verified.'))
    jobs_path = args.output / 'jobs.json'
    if jobs_path.exists():
        jobs = common.read(jobs_path)
        jobs.update(complete=True, completed_utc=datetime.now(timezone.utc).isoformat())
        common.write(jobs_path, jobs)
    print('All 40 comparisons verified; PNG, PDF, SVG, and count matrices saved.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'compute', 'reduce', 'finish'))
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--family', choices=FAMILIES)
    parser.add_argument('--source', choices=SPLITS)
    parser.add_argument('--task', type=int, default=0)
    parser.add_argument('--tasks', type=int, default=16)
    parser.add_argument('--threshold', type=float, default=.7)
    parser.add_argument('--batch-rows', type=int, default=2048)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    assert args.tasks > 0 and 0 < args.threshold <= 1
    if args.stage in ('compute', 'reduce'):
        assert args.family
    if args.stage == 'reduce':
        assert args.source
    globals()[args.stage](args)


if __name__ == '__main__':
    main()
