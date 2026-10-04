#!/usr/bin/env python3
"""Stream complete main-SAE caches; save exact counts, moments, and rate plots."""
from project_paths import resource_path as _paper_path, resource_location as _paper_location
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from numba import njit

WIDTHS = [512 * 2**i for i in range(9)]
TOPKS = [32, 32, 32, 32, 64, 64, 64, 128, 128]
BASE = _paper_path('/resources/activation_cache_dir')
DEFAULT_OUT = _paper_path('full_experiments/results/main_sae_activation_statistics')
EDGES = np.array([0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
THRESHOLDS = np.array([0.01, 0.05], dtype=np.float32)


@njit(cache=True)
def accumulate(indptr, indices, data, counts, sums, squares, maxima, seen):
    """Float64 moments, int64 counts; positive support must be unique per row."""
    row_stats = np.zeros((3, 4), np.float64)  # sum, sumsq, min, max per L0 threshold
    row_stats[:, 2] = np.inf
    seen[:] = -1
    for row in range(len(indptr) - 1):
        active = np.zeros(3, np.int64)
        for j in range(indptr[row], indptr[row + 1]):
            f = indices[j]
            v = np.float64(data[j])
            if f < 0 or f >= len(sums) or not np.isfinite(v) or v < 0:
                raise ValueError('Invalid index or activation')
            if v == 0:
                continue
            if seen[f] == row:
                raise ValueError('Duplicate positive feature within row')
            seen[f] = row
            sums[f] += v
            squares[f] += v*v
            maxima[f] = max(maxima[f], v)
            counts[0, f] += 1
            active[0] += 1
            for t in range(2):
                if data[j] > THRESHOLDS[t]:
                    counts[t+1, f] += 1
                    active[t+1] += 1
        for t in range(3):
            row_stats[t, 0] += active[t]
            row_stats[t, 1] += active[t]**2
            row_stats[t, 2] = min(row_stats[t, 2], active[t])
            row_stats[t, 3] = max(row_stats[t, 3], active[t])
    return row_stats


def specification(task, base):
    family = ['gemini', 'nemotron'][task // 9]
    width, k = WIDTHS[task % 9], TOPKS[task % 9]
    if family == 'nemotron':
        cache = 'nemotron_all_experiment_saes_post_topk'
    elif width in (4096, 131072):
        cache = 'gemini_all_corpus_post_topk'
    elif width == 16384:
        cache = 'gemini_m16384_all_corpus_post_topk'
    else:
        cache = 'gemini_remaining_experiment_saes_post_topk'
    return family, width, k, base / cache


def compute(args):
    family, width, k, root = specification(args.task, args.cache_base)
    name = f'{family}_m{width}_k{k}'
    source = root / 'manifest.json'
    if not source.exists():
        source = root / 'plan.json'
    payload = source.read_bytes()
    meta = json.loads(payload)
    model = next(m for m in meta['models'] if m['name'] == name)
    assert model['width'] == width and model['top_k'] == k
    shards = meta.get('shards', meta.get('source_shards'))
    assert len(shards) == meta['expected_shards']
    assert len({s['relative_shard'] for s in shards}) == len(shards)
    counts = np.zeros((3, width), np.int64)
    sums, squares, maxima = np.zeros((3, width), np.float64)
    seen = np.full(width, -1, np.int64)
    row_stats = np.zeros((3, 4), np.float64)
    row_stats[:, 2] = np.inf
    rows = 0
    for i, shard in enumerate(shards):
        assert shard['global_row_start'] == rows
        assert shard['global_row_stop'] == rows + shard['rows']
        directory = root / 'shards' / shard['relative_shard']
        complete = json.loads((directory / 'complete.json').read_text())
        assert complete['complete'] and complete['spec_id'] == meta['spec_id']
        assert complete['source']['relative_shard'] == shard['relative_shard']
        output = complete['outputs'][name]
        assert output['shape'] == [shard['rows'], width] and output['top_k'] == k
        arrays = {}
        for component, dtype in [('indptr', 'int32'), ('indices', 'int32'), ('data', 'float32')]:
            entry = output['components'][component]
            path = directory / entry['file']
            assert path.stat().st_size == entry['size_bytes']
            array = np.load(_paper_location(path), mmap_mode='r')
            assert str(array.dtype) == dtype and list(array.shape) == entry['shape']
            arrays[component] = array
        indptr, indices, data = (arrays[x] for x in ('indptr', 'indices', 'data'))
        assert len(indptr) == shard['rows'] + 1 and indptr[0] == 0
        assert indptr[-1] == len(data) == len(indices)
        assert np.all(np.diff(indptr) == k)
        old_positive = counts[0].sum()
        stats = accumulate(indptr, indices, data, counts, sums, squares, maxima, seen)
        assert counts[0].sum() - old_positive == output['positive_nnz']
        row_stats[:, :2] += stats[:, :2]
        row_stats[:, 2] = np.minimum(row_stats[:, 2], stats[:, 2])
        row_stats[:, 3] = np.maximum(row_stats[:, 3], stats[:, 3])
        rows += shard['rows']
        if (i+1) % 50 == 0 or i+1 == len(shards):
            print(f'{name}: {i+1}/{len(shards)} shards, {rows:,} rows', flush=True)
    assert rows == meta['expected_rows']
    assert np.all(counts[2] <= counts[1]) and np.all(counts[1] <= counts[0])
    assert np.all(counts[0] <= rows)
    out = args.output / 'per_feature'
    out.mkdir(parents=True, exist_ok=True)
    provenance = dict(family=family, width=width, top_k=k, rows=rows, shards=len(shards),
                      model=model, cache_root=str(root), metadata_path=str(source),
                      metadata_sha256=hashlib.sha256(payload).hexdigest(), spec_id=meta['spec_id'])
    np.savez_compressed(_paper_location(out / f'{name}.npz'), feature_id=np.arange(width), rows=rows,
                        thresholds=THRESHOLDS, counts=counts, rates=counts/rows,
                        activation_sum=sums, activation_sum_squares=squares,
                        activation_mean=sums/rows,
                        activation_std=np.sqrt(np.maximum(0, squares/rows-(sums/rows)**2)),
                        activation_max=maxima, row_l0_statistics=row_stats,
                        provenance=json.dumps(provenance))
    (out / f'{name}.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(f'Saved {name}', flush=True)


def write_csv(path, records):
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def describe(x):
    return dict(mean=float(x.mean()), std=float(x.std()), min=float(x.min()),
                **{f'p{q:g}': float(np.percentile(x, q)) for q in [1, 5, 25, 50, 75, 95, 99]},
                max=float(x.max()))


def finish(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    plots = args.output / 'plots'
    plots.mkdir(parents=True, exist_ok=True)
    bins, rate_summary, activation_summary, top, all_data = [], [], [], [], {}
    for task in range(18):
        family, width, k, _ = specification(task, args.cache_base)
        name = f'{family}_m{width}_k{k}'
        with np.load(_paper_location(args.output / 'per_feature' / f'{name}.npz')) as archive:
            d = {key: archive[key] for key in archive.files}
        rows = int(d['rows'])
        rates, counts = d['rates'], d['counts']
        assert rates.shape == (3, width)
        assert np.allclose(rates, counts/rows, rtol=0, atol=0)
        all_data[family, width] = rates
        positive_mean = np.divide(d['activation_sum'], counts[0], out=np.zeros(width), where=counts[0]>0)
        positive_std = np.sqrt(np.maximum(0, np.divide(d['activation_sum_squares'], counts[0], out=np.zeros(width), where=counts[0]>0)-positive_mean**2))
        records = [dict(feature_id=f, count_gt_0=int(counts[0,f]), count_gt_0p01=int(counts[1,f]),
                        count_gt_0p05=int(counts[2,f]), rate_gt_0=float(rates[0,f]),
                        rate_gt_0p01=float(rates[1,f]), rate_gt_0p05=float(rates[2,f]),
                        activation_mean=float(d['activation_mean'][f]), activation_std=float(d['activation_std'][f]),
                        activation_max=float(d['activation_max'][f]),
                        positive_activation_mean=float(positive_mean[f]) if counts[0,f] else None,
                        positive_activation_std=float(positive_std[f]) if counts[0,f] else None) for f in range(width)]
        write_csv(args.output/'per_feature'/f'{name}.csv', records)
        total = rows*width
        total_sum, total_squares = d['activation_sum'].sum(), d['activation_sum_squares'].sum()
        positive_count = int(counts[0].sum())
        a = dict(family=family, width=width, top_k=k, rows=rows, feature_example_pairs=total,
                 positive_activations=positive_count, zero_fraction=1-positive_count/total,
                 activation_mean=float(total_sum/total),
                 activation_std=float(np.sqrt(max(0,total_squares/total-(total_sum/total)**2))),
                 activation_min=0., activation_max=float(d['activation_max'].max()),
                 positive_activation_mean=float(total_sum/positive_count),
                 positive_activation_std=float(np.sqrt(max(0,total_squares/positive_count-(total_sum/positive_count)**2))),
                 dead_features=int(np.sum(counts[0]==0)))
        for t, label in enumerate(['gt_0','gt_0p01','gt_0p05']):
            s = d['row_l0_statistics'][t]
            a.update({f'row_l0_{label}_mean':float(s[0]/rows),
                      f'row_l0_{label}_std':float(np.sqrt(max(0,s[1]/rows-(s[0]/rows)**2))),
                      f'row_l0_{label}_min':int(s[2]), f'row_l0_{label}_max':int(s[3])})
        activation_summary.append(a)
        for t, threshold in enumerate([0.01,0.05],1):
            r = rates[t]
            h, _ = np.histogram(r, bins=EDGES)
            assert h.sum() == width
            for j, count in enumerate(h):
                bins.append(dict(family=family, width=width, threshold=threshold, rows=rows,
                                 bin_lower=EDGES[j], bin_upper=EDGES[j+1],
                                 interval=f'[{EDGES[j]:g}, {EDGES[j+1]:g}'+(']' if j==6 else ')'),
                                 feature_count=int(count)))
            rate_summary.append(dict(family=family,width=width,threshold=threshold,rows=rows,
                                     zero_rate_features=int((r==0).sum()),**describe(r)))
            order = np.argsort(-r, kind='stable')[:512]
            top.extend(dict(family=family,width=width,threshold=threshold,rank=rank+1,
                            feature_id=int(f),activation_rate=float(r[f])) for rank,f in enumerate(order))
    for name, records in [('bin_counts',bins),('activation_rate_summary',rate_summary),
                          ('activation_summary',activation_summary),('top_512',top)]:
        write_csv(args.output/f'{name}.csv',records)
        (args.output/f'{name}.json').write_text(json.dumps(records,indent=2)+'\n')
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    colors = plt.get_cmap('viridis')(np.linspace(.05,.95,9))
    grid = np.linspace(-8.5,0,600)
    def save(fig, name):
        for ext in ['png','pdf','svg']:
            fig.savefig(plots/f'{name}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    for family in ['gemini','nemotron']:
        fig, axes = plt.subplots(1,2,figsize=(15,5),sharex=True,sharey=True)
        facet, fax = plt.subplots(3,3,figsize=(15,11),sharex=True,sharey=True)
        for i,width in enumerate(WIDTHS):
            for t,threshold in enumerate([0.01,0.05],1):
                r = all_data[family,width][t]
                log = np.log10(r[r>0])
                # Fixed bandwidth 0.12 log10 units makes widths comparable.
                if len(log)>1 and log.std(ddof=1)>0:
                    density = gaussian_kde(log,bw_method=.12/log.std(ddof=1))(grid)
                else:
                    density = np.zeros_like(grid)
                    if len(log):
                        density = np.exp(-.5*((grid-log[0])/.12)**2)/(.12*np.sqrt(2*np.pi))
                axes[t-1].plot(grid,density * len(log),color=colors[i],label=f'{width:,} ({(r==0).sum():,} zeros)')
                fax.flat[i].plot(grid,density,label=f'> {threshold:g}; {(r==0).sum():,} zeros')
            fax.flat[i].set_title(f'{width:,} features')
            fax.flat[i].legend(fontsize=8)
        for ax,threshold in zip(axes,[0.01,0.05]):
            ax.set_title(f'Activation > {threshold:g}')
            ax.set_xlabel('log₁₀ activation rate')
            ax.grid(alpha=.2)
            ax.legend(fontsize=8,title='SAE size (zero-rate count)')
        axes[0].set_ylabel('Features per log₁₀ activation-rate unit')
        fig.suptitle(f'{family.capitalize()} · log activation rates · unnormalized feature density')
        facet.supxlabel('log₁₀ activation rate')
        facet.supylabel('Density among positive-rate features', x=.005)
        facet.suptitle(f'{family.capitalize()} · log activation rates by SAE size')
        facet.tight_layout(rect=(.025,.015,1,.98))
        save(fig,f'{family}_log_rate_density')
        save(facet,f'{family}_log_rate_density_by_size')
        for scale in ['linear','log']:
            fig, axes = plt.subplots(1,2,figsize=(15,5),sharex=True,sharey=True)
            for i,width in enumerate(WIDTHS):
                for t,threshold in enumerate([0.01,0.05],1):
                    r = np.sort(all_data[family,width][t])[::-1][:512]
                    axes[t-1].plot(np.arange(1,513),np.where(r>0,r,np.nan) if scale=='log' else r,
                                   color=colors[i],label=f'{width:,}')
            for ax,threshold in zip(axes,[0.01,0.05]):
                ax.set_title(f'Activation > {threshold:g}')
                ax.set_xlabel('Feature rank (descending activation rate)')
                ax.set_xlim(1,512)
                ax.set_yscale(scale)
                ax.grid(alpha=.2)
                ax.legend(fontsize=8,title='SAE size')
            axes[0].set_ylabel('Activation rate (fraction of examples)')
            fig.suptitle(f'{family.capitalize()} · 512 highest activation rates per SAE')
            save(fig,f'{family}_top_512_{scale}')
    (args.output/'README.md').write_text('''# Main SAE activation statistics

All nine main SAE sizes, 512 through 131,072, for Gemini and Nemotron. Split-trained variants are excluded.

Rates are the fraction of full-corpus examples with cached post-top-k activation strictly greater than 0.01 or 0.05 (float32 comparisons). Gemini uses all 89,827,558 examples, Nemotron all 89,227,558; every width within a family uses the same corpus. These are full-corpus summaries, not estimates or an aligned-corpus subset. The corpora differ by 600,000 examples, so cross-family comparisons also reflect this coverage difference.

`bin_counts.csv/json` contains the requested seven bins: [0, 1e-6), [1e-6, 1e-5), [1e-5, 1e-4), [1e-4, 1e-3), [1e-3, 1e-2), [1e-2, 0.1), [0.1, 1]. Zero rates are included in the first bin and counted separately in `activation_rate_summary.csv/json`. Bin counts sum to the SAE width.

`per_feature/` contains one CSV and NPZ per SAE, with zero-based feature IDs, exact int64 activation counts at >0, >0.01, >0.05, float64 rates, and continuous activation moments. NPZ counts/rates have shape (3, width), ordered >0, >0.01, >0.05. Missing sparse entries and padding values are zero. Means and population standard deviations include those zeros; positive activation moments in CSV condition on activation >0 (blank if no positives). NPZ also stores sums, sums of squares, maxima, and per-row L0 aggregates (columns: sum, sumsq, minimum, maximum; rows: the three thresholds). JSON sidecars record model fingerprints, cache paths, metadata hashes, row counts, and shard counts.

`activation_summary.csv/json` gives pooled activation means, population standard deviations, extrema, zero fractions, dead feature counts, and per-example L0 statistics. `activation_rate_summary.csv/json` gives feature-rate mean, population standard deviation, minimum, maximum and percentiles (NumPy linear interpolation). `top_512.csv/json` preserves feature IDs and ranks for both thresholds; ties are ordered by feature ID.

`plots/` has PNG, PDF, and SVG figures for each family. Density overlays and 3×3 per-size panels show Gaussian KDEs of log10 strictly positive rates, with a common bandwidth of 0.12 log10 units. Overlaid curves are unnormalized: each KDE is multiplied by the number of positive-rate features, so its integral over the real line equals that feature count. The overlay y-axis is features per log10 activation-rate unit. The 3×3 per-size panels retain unit-area densities. Zero-rate features are omitted from log densities and counted in legends. KDE smoothing can extend beyond the empirical support. Top-512 plots sort independently within each SAE and threshold; both linear and logarithmic y-axis versions are included (zero rates omitted only on log axes).

Computation streams every source shard and validates completion/spec IDs, contiguous corpus coverage, array shape/dtype/size, row top-k structure, finite nonnegative values, in-range indices, unique positive feature support, and recorded positive counts. Some caches have plan.json and shard completion files but no final root manifest; these are validated directly rather than treated as root-audited caches. No cache inputs are modified. Counts use int64 and moments use float64. Validation does not rehash every input payload or rerun model inference.

Reproduce from repository root:

```bash
for task in $(seq 0 17); do
  python scripts/summarize_main_sae_activations.py compute --task "$task"
done
python scripts/summarize_main_sae_activations.py finish
```
''')
    with (args.output/'README.md').open('a') as f:
        f.write('\n## Feature counts by activation-rate bin\n\n')
        for family in ['gemini', 'nemotron']:
            for threshold in [0.01, 0.05]:
                f.write(f'### {family.capitalize()}, activation > {threshold:g}\n\n')
                f.write('| SAE size | [0, 1e-6) | [1e-6, 1e-5) | [1e-5, 1e-4) | [1e-4, 1e-3) | [1e-3, .01) | [.01, .1) | [.1, 1] |\n')
                f.write('|---:|---:|---:|---:|---:|---:|---:|---:|\n')
                for width in WIDTHS:
                    values = [r['feature_count'] for r in bins if r['family']==family and r['width']==width and r['threshold']==threshold]
                    assert len(values)==7 and sum(values)==width
                    f.write('| ' + ' | '.join(f'{v:,}' for v in [width,*values]) + ' |\n')
                f.write('\n')
    (args.output/'COMPLETE.json').write_text(json.dumps(dict(
        complete=True, sae_count=18, threshold_summaries=len(rate_summary),
        binned_feature_counts=len(bins), ranked_features=len(top),
        total_features=sum(WIDTHS)*2, plots=sorted(p.name for p in plots.iterdir())),indent=2)+'\n')
    print('Validated and wrote all summaries and plots to',args.output,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['compute','finish'])
    parser.add_argument('--task',type=int,choices=range(18))
    parser.add_argument('--cache-base',type=Path,default=BASE)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.command=='compute':
        if args.task is None:
            parser.error('compute requires --task')
        compute(args)
    else:
        finish(args)


if __name__=='__main__':
    main()
