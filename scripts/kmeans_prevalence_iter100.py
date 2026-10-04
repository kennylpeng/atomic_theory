"""Full-corpus Figure 10 analogue for the three 100-iteration KMeans widths."""
from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import argparse
import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import time

import numpy as np

from scripts import match_split_kmeans_activations as core
from scripts import train_kmeans_full_corpus as common

ROOT = common.ROOT
TRAINING = ROOT / 'full_experiments/results/kmeans_raw_full_corpus_iter100'
SPLIT_TRAINING = ROOT / 'full_experiments/results/kmeans_raw_all_splits_iter100'
CACHE = SPLIT_TRAINING / 'activation_similarity_k16384/iter100'
OUT = TRAINING / 'prevalence'
WIDTHS = (4096, 16384, 131072)
FAMILIES = ('gemini', 'nemotron')
BANDWIDTH = .12
GRID = np.linspace(-8.5, 0, 600)


def immutable(path, value):
    if path.exists():
        assert common.read(path) == value, path
    else:
        common.write(path, value)


def atomic_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.stem}.{os.getpid()}.tmp.npz')
    np.savez_compressed(_paper_location(temporary), **arrays)
    temporary.replace(path)


def valid_counts(counts, width, rows):
    assert counts.dtype == np.int64 and counts.shape == (width,)
    assert np.all((counts >= 0) & (counts <= rows)) and int(counts.sum()) == rows


def prepare(args):
    assert common.read(TRAINING / 'summary.json')['complete']
    assert common.read(CACHE / 'summary.json')['complete']
    for family in FAMILIES:
        plan_path = common.OUT / family / 'plan.json'
        plan = common.read(plan_path)
        cache_path = CACHE / family / 'manifest.json'
        cache = common.read(cache_path)
        assert cache['iterations'] == 100 and cache['rows'] == plan['rows']
        assert cache['shards'] == plan['shards']
        assert common.read(CACHE / family / 'main/summary.json')['manifest_id'] == cache['id']
        audit_path = CACHE / family / 'assignment_validation.json'
        audit = common.read(audit_path)
        assert audit['passed'] and audit['manifest_id'] == cache['id']
        models = []
        for width in WIDTHS:
            folder = ((SPLIT_TRAINING / 'main') if width == 16384 else TRAINING) / 'models' / family / f'k{width}'
            meta = common.read(folder / 'metadata.json')
            done = common.read(folder / 'completed.json')
            assert done['complete'] and done['actual_iterations'] == meta['actual_iterations'] == len(meta['objective']) == 100
            assert meta['sample_rows'] == plan['rows'] and meta['sample_path'] is None
            assert meta['training_split'] == 'main' and meta['all_corpus_rows'] and not meta['faiss_subsampling']
            assert meta['input_space'] == 'raw' and meta['k'] == width and meta['training_dim'] == plan['dimension']
            path = folder / 'centroids.npy'
            assert common.digest(path) == done['centroids_sha256']
            assert common.digest(folder / 'metadata.json') == done['metadata_sha256']
            centers = np.load(_paper_location(path), mmap_mode='r')
            assert centers.shape == (width, plan['dimension']) and centers.dtype == np.float32
            assert np.isfinite(centers).all()
            centers._mmap.close()
            models.append(dict(width=width, path=str(path), sha256=done['centroids_sha256'],
                               metadata_path=str(folder / 'metadata.json'), metadata_sha256=done['metadata_sha256']))
            if width == 16384:
                assert cache['models'][0]['split'] == 'main' and cache['models'][0]['sha256'] == done['centroids_sha256']
        selected = set(np.linspace(0, len(plan['shards'])-1, 16, dtype=int).tolist())
        datasets = set()
        for i, shard in enumerate(plan['shards']):
            common.stat_matches(shard['path'], shard['size'], shard['mtime_ns'])
            dataset = shard['relative_shard'].split('/')[0]
            if dataset not in datasets:
                selected.add(i)
                datasets.add(dataset)
        paths = [CACHE / family / 'main' / f'matching_to_{split}.npz'
                 for split in ('wikipedia', 'no_wikipedia', 'random1', 'random2')]
        counts = None
        for path in paths:
            with np.load(_paper_location(path), allow_pickle=False) as z:
                assert int(z['rows']) == plan['rows']
                current = z['source_counts']
                valid_counts(current, 16384, plan['rows'])
                if counts is not None:
                    np.testing.assert_array_equal(current, counts)
                counts = current.copy()
        manifest = dict(family=family, dimension=plan['dimension'], rows=plan['rows'],
                        shards=plan['shards'], models=models, widths=list(WIDTHS), compute_widths=[4096, 131072],
                        iterations=100, training_split='main', evaluation_split='main',
                        tasks=args.tasks, batch_rows=args.batch_rows, audit_shards=sorted(selected),
                        audit_datasets=sorted(datasets), plan_path=str(plan_path), plan_sha256=common.digest(plan_path),
                        cache_manifest_path=str(cache_path), cache_manifest_sha256=common.digest(cache_path),
                        cache_audit_path=str(audit_path), cache_audit_sha256=common.digest(audit_path),
                        cache_counts=[dict(path=str(p), sha256=common.digest(p)) for p in paths],
                        assignment='Exhaustive float32 nearest Euclidean centroid; TF32 disabled; lowest-index ties',
                        storage='Original embedding shards read directly; only counts and audit labels saved',
                        source_code_sha256=common.digest(__file__), core_code_sha256=common.digest(core.__file__))
        manifest['id'] = common.object_digest(manifest)
        immutable(args.output / family / 'manifest.json', manifest)
        atomic_npz(args.output / family / 'k16384_counts.npz', counts=counts, rows=plan['rows'],
                   manifest_id=manifest['id'])
        print(f'{family}: prepared {plan["rows"]:,} rows; reused identical 16K marginals from four comparisons', flush=True)


def load_manifest(output, family):
    manifest = common.read(output / family / 'manifest.json')
    assert manifest['id'] == common.object_digest({k: v for k, v in manifest.items() if k != 'id'})
    assert manifest['source_code_sha256'] == common.digest(__file__)
    assert manifest['core_code_sha256'] == common.digest(core.__file__)
    return manifest


def count_shard(shard, centers, batch_rows, device, audit):
    import torch
    common.stat_matches(shard['path'], shard['size'], shard['mtime_ns'])
    x = np.load(_paper_location(shard['path']), mmap_mode='r', allow_pickle=False)
    assert len(x) == shard['rows'] and x.shape[1] == next(iter(centers.values()))[0].shape[1]
    counts = {width: np.zeros(width, dtype=np.int64) for width in centers}
    audit_rows = np.unique([len(x)//3, 2*len(x)//3]) if audit else np.empty(0, dtype=np.int64)
    observed = {width: np.empty(len(audit_rows), dtype=np.int64) for width in centers}
    with torch.inference_mode():
        for start in range(0, len(x), batch_rows):
            stop = min(start+batch_rows, len(x))
            batch = torch.from_numpy(np.array(x[start:stop], dtype=np.float32)).to(device)
            assert torch.isfinite(batch).all()
            keep = np.flatnonzero((audit_rows >= start) & (audit_rows < stop))
            for width, (c, norm) in centers.items():
                labels = core.assign(batch, c, norm).cpu().numpy()
                assert labels.min() >= 0 and labels.max() < width
                counts[width] += np.bincount(labels, minlength=width)
                observed[width][keep] = labels[audit_rows[keep]-start]
    vectors = np.array(x[audit_rows], dtype=np.float64)
    x._mmap.close()
    common.stat_matches(shard['path'], shard['size'], shard['mtime_ns'])
    for width, count in counts.items():
        valid_counts(count, width, shard['rows'])
    return counts, audit_rows, vectors, observed


def audit_labels(vectors, observed, cpu_centers):
    checks = []
    if not len(vectors):
        return checks
    for width, centers in cpu_centers.items():
        # Independent float64 CPU search, blocked to avoid another multi-GB copy.
        best = np.full(len(vectors), -np.inf)
        reference = np.zeros(len(vectors), dtype=np.int64)
        chosen = centers[observed[width]].astype(np.float64)
        observed_scores = 2*np.einsum('ij,ij->i', vectors, chosen) - np.einsum('ij,ij->i', chosen, chosen)
        for start in range(0, width, 2048):
            c = centers[start:start+2048].astype(np.float64)
            scores = 2*vectors @ c.T - np.einsum('ij,ij->i', c, c)
            local = scores.argmax(axis=1)
            values = scores[np.arange(len(vectors)), local]
            better = values > best
            reference[better] = start+local[better]
            best[better] = values[better]
        gap = float(np.maximum(0, best-observed_scores).max())
        assert gap <= 1e-5, (width, gap)
        checks.append(dict(width=width, checked=len(vectors), exact_labels=int((reference == observed[width]).sum()),
                           max_float64_score_gap=gap, tolerance=1e-5))
    return checks


def compute(args):
    import torch
    manifest = load_manifest(args.output, args.family)
    assert 0 <= args.task < manifest['tasks']
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    centers, cpu_centers = {}, {}
    for model in manifest['models']:
        if model['width'] not in manifest['compute_widths']:
            continue
        assert common.digest(model['path']) == model['sha256']
        width = model['width']
        cpu_centers[width] = np.load(_paper_location(model['path']), allow_pickle=False)
        c = torch.from_numpy(cpu_centers[width]).to(args.device)
        assert torch.isfinite(c).all()
        centers[width] = (c, c.square().sum(1))
    started = time.monotonic()
    assigned = list(range(args.task, len(manifest['shards']), manifest['tasks']))
    for position, index in enumerate(assigned):
        shard = manifest['shards'][index]
        folder = args.output / args.family / 'shards' / shard['relative_shard']
        common.stat_matches(shard['path'], shard['size'], shard['mtime_ns'])
        if (folder / 'complete.json').exists():
            done = common.read(folder / 'complete.json')
            assert done['manifest_id'] == manifest['id'] and done['shard'] == shard
            assert common.digest(folder / 'counts.npz') == done['counts_sha256']
            continue
        counts, audit_rows, vectors, labels = count_shard(
            shard, centers, manifest['batch_rows'], args.device, index in manifest['audit_shards'])
        checks = audit_labels(vectors, labels, cpu_centers)
        atomic_npz(folder / 'counts.npz', **{f'k{w}': c for w, c in counts.items()},
                   audit_rows=audit_rows, **{f'audit_labels_k{w}': a for w, a in labels.items()})
        common.write(folder / 'complete.json', dict(manifest_id=manifest['id'], shard=shard,
                     counts_sha256=common.digest(folder / 'counts.npz'), audit=checks))
        print(f'{args.family} task {args.task}: {position+1}/{len(assigned)} shards; '
              f'{time.monotonic()-started:.1f}s', flush=True)
    common.write(args.output / args.family / 'tasks' / f'{args.task:03d}.json',
                 dict(complete=True, manifest_id=manifest['id'], indices=assigned,
                      elapsed_seconds=time.monotonic()-started, gpu=torch.cuda.get_device_name() if args.device != 'cpu' else 'cpu'))


def collect(output):
    populations, provenance = {}, []
    for family in FAMILIES:
        manifest = load_manifest(output, family)
        for key in ('plan', 'cache_manifest', 'cache_audit'):
            assert common.digest(manifest[f'{key}_path']) == manifest[f'{key}_sha256']
        for model in manifest['models']:
            assert common.digest(model['path']) == model['sha256']
            assert common.digest(model['metadata_path']) == model['metadata_sha256']
        with np.load(_paper_location(output / family / 'k16384_counts.npz'), allow_pickle=False) as z:
            assert str(z['manifest_id']) == manifest['id'] and int(z['rows']) == manifest['rows']
            cached = z['counts'].copy()
        for record in manifest['cache_counts']:
            assert common.digest(record['path']) == record['sha256']
            with np.load(_paper_location(record['path']), allow_pickle=False) as z:
                np.testing.assert_array_equal(cached, z['source_counts'])
                assert int(z['rows']) == manifest['rows']
        populations[family, 16384] = (cached, manifest['rows'])
        counts = {width: np.zeros(width, dtype=np.int64) for width in manifest['compute_widths']}
        for task in range(manifest['tasks']):
            record = common.read(output / family / 'tasks' / f'{task:03d}.json')
            assert record['complete'] and record['manifest_id'] == manifest['id']
            assert record['indices'] == list(range(task, len(manifest['shards']), manifest['tasks']))
        audits, covered = [], 0
        for index, shard in enumerate(manifest['shards']):
            common.stat_matches(shard['path'], shard['size'], shard['mtime_ns'])
            folder = output / family / 'shards' / shard['relative_shard']
            done = common.read(folder / 'complete.json')
            assert done['manifest_id'] == manifest['id'] and done['shard'] == shard
            assert common.digest(folder / 'counts.npz') == done['counts_sha256']
            with np.load(_paper_location(folder / 'counts.npz'), allow_pickle=False) as z:
                for width in counts:
                    valid_counts(z[f'k{width}'], width, shard['rows'])
                    counts[width] += z[f'k{width}']
                expected = len(np.unique([shard['rows']//3, 2*shard['rows']//3])) if index in manifest['audit_shards'] else 0
                assert len(z['audit_rows']) == expected
            assert len(done['audit']) == (len(counts) if expected else 0)
            assert {a['width'] for a in done['audit']} == (set(counts) if expected else set())
            for audit in done['audit']:
                assert audit['checked'] == expected and audit['max_float64_score_gap'] <= audit['tolerance'] == 1e-5
                audits.append(dict(shard=shard['relative_shard'], **audit))
            covered += shard['rows']
        assert covered == manifest['rows']
        for width, values in counts.items():
            valid_counts(values, width, covered)
            populations[family, width] = (values, covered)
        provenance.append(dict(family=family, manifest_id=manifest['id'], rows=covered,
                               manifest_path=str(output / family / 'manifest.json'), audit=audits))
    return populations, provenance


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(output, populations):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter
    from scipy.stats import gaussian_kde
    summaries, curves, curve_rows = [], {}, []
    for family in FAMILIES:
        for width in WIDTHS:
            counts, n = populations[family, width]
            valid_counts(counts, width, n)
            rates = counts.astype(np.float64)/n
            np.testing.assert_allclose(rates.sum(), 1, rtol=0, atol=1e-14)
            positive = rates > 0
            log_rates = np.log10(rates[positive])
            kde = gaussian_kde(log_rates, bw_method=BANDWIDTH/log_rates.std(ddof=1))
            np.testing.assert_allclose(np.sqrt(kde.covariance[0, 0]), BANDWIDTH)
            density = kde(GRID)*positive.sum()
            area = float(np.trapz(density, GRID))
            np.testing.assert_allclose(area, positive.sum(), rtol=1e-4)
            curves[family, width] = density
            summaries.append(dict(family=family, width=width, iterations=100, evaluation_rows=n,
                positive_rate_clusters=int(positive.sum()), zero_rate_clusters=int((~positive).sum()),
                assignment_count_sum=int(counts.sum()), prevalence_sum=float(rates.sum()),
                mean_prevalence=float(rates.mean()), median_prevalence=float(np.median(rates)),
                p05_prevalence=float(np.quantile(rates, .05)), p95_prevalence=float(np.quantile(rates, .95)),
                minimum_positive_prevalence=float(rates[positive].min()), maximum_prevalence=float(rates.max()),
                kde_bandwidth_log10=BANDWIDTH, kde_grid_integral=area))
            curve_rows.extend(dict(family=family, width=width, log10_prevalence=float(x), cluster_count_density=float(y))
                              for x, y in zip(GRID, density))
            atomic_npz(output / f'{family}_k{width}_prevalence.npz', cluster_id=np.arange(width),
                       counts=counts, prevalence=rates, evaluation_rows=n, log10_grid=GRID, count_density=density)
    plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    colors = plt.get_cmap('viridis')(np.linspace(.05, .95, 9))[[3, 5, 8]]
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 8.5), sharex=True, sharey=True)
    upper = max(y.max() for y in curves.values())*1.12
    root_ticks = MaxNLocator(nbins=5).tick_values(0, np.sqrt(upper))
    ticks = root_ticks[(root_ticks >= 0) & (root_ticks <= np.sqrt(upper))]**2
    for ax, family in zip(axes, FAMILIES):
        for color, width in zip(colors, WIDTHS):
            ax.plot(GRID, curves[family, width], color=color, linewidth=2.5, label=f'{width:,}')
        ax.set_yscale('function', functions=(lambda y: np.sqrt(np.maximum(y, 0)), np.square))
        ax.set_ylim(0, upper)
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        ax.set_xlim(GRID[0], GRID[-1])
        ax.set_xticks(np.arange(-8, 1))
        ax.set_ylabel('Clusters per\nlog₁₀ prevalence unit')
        ax.set_title(f'{family.capitalize()} · {populations[family, WIDTHS[0]][1]:,} examples', loc='left', fontsize=14)
        ax.grid(alpha=.18)
    axes[-1].set_xlabel('log₁₀ prevalence (fraction of examples assigned to the cluster)')
    fig.suptitle('Full-corpus KMeans · 100 iterations', fontsize=18, y=.98)
    fig.legend(axes[0].lines, [f'{width:,}' for width in WIDTHS], title='Number of clusters',
               loc='upper center', bbox_to_anchor=(.55, .943), ncol=3, frameon=False)
    zero_lines = []
    for family in FAMILIES:
        selected = [r for r in summaries if r['family'] == family]
        zero_lines.append(f'{family.capitalize()}: ' + ', '.join(f"{r['width']:,}: {r['zero_rate_clusters']:,}" for r in selected))
    fig.text(.14, .025, 'Count-scaled KDEs; bandwidth 0.12; square-root y-axis. All clusters included.\n'
             'Zero-rate clusters omitted from the log axis (width: count):\n' + '\n'.join(zero_lines),
             fontsize=8.5, color='#444444', va='bottom')
    fig.subplots_adjust(left=.14, right=.98, top=.835, bottom=.17, hspace=.31)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(output / f'kmeans_prevalence_figure10.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)
    write_csv(output / 'summary.csv', summaries)
    write_csv(output / 'density_curves.csv', curve_rows)
    return summaries


def finish(args):
    populations, provenance = collect(args.output)
    summaries = plot(args.output, populations)
    common.write(args.output / 'summary.json', dict(complete=True, iterations=100, models=summaries,
        completed_utc=datetime.now(timezone.utc).isoformat(), inputs=provenance,
        definition='P(nearest-centroid binary activation > 0) = assignment count / full-corpus rows',
        population='All clusters; no matching filter. Training and evaluation both use the entire main corpus.',
        median_threshold='Not plotted: all positive binary activations equal one, so activation > positive median is always false.',
        figure_reference='Figure 10: Feature activation rates',
        plotting=dict(bandwidth_log10=BANDWIDTH, scale='count-scaled KDE, square-root y-axis'),
        validation='All rows and source hashes checked; integer counts sum to corpus size; 16K marginals agree '
                   'across four comparisons; new assignments audited with independent float64 CPU distances; KDE areas checked.'))
    print('Completed all six prevalence distributions:', args.output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'compute', 'finish'])
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--family', choices=FAMILIES)
    parser.add_argument('--task', type=int, default=0)
    parser.add_argument('--tasks', type=int, default=32)
    parser.add_argument('--batch-rows', type=int, default=1024)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    globals()[args.stage](args)


if __name__ == '__main__':
    main()
