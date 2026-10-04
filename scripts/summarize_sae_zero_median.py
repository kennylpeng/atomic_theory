#!/usr/bin/env python3
"""Exact zero/pooled-positive-median SAE prevalence from completed sparse caches."""
from project_paths import resource_path as _paper_path, resource_location as _paper_location
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from numba import njit

sys.path.insert(0, str(_paper_path(__file__).resolve().parents[1]))
from scripts.summarize_main_sae_activations import (
    BASE, DEFAULT_OUT as ORIGINAL_OUT, WIDTHS, TOPKS, EDGES,
    specification, write_csv, describe,
)

DEFAULT_OUT = _paper_path('full_experiments/results/main_sae_activation_statistics_zero_median')


@njit(cache=True)
def accumulate_bounds(data, indices, k, upper, positive, above, row_above):
    for row in range(len(row_above)):
        n = 0
        for j in range(row*k, (row+1)*k):
            v, f = data[j], indices[j]
            if not np.isfinite(v) or v < 0 or f < 0 or f >= len(positive):
                raise ValueError('Invalid sparse activation or feature index')
            if v > 0:
                positive[f] += 1
            if v > upper:
                above[f] += 1
                n += 1
        row_above[row] = n


def exact_median(candidates, positive_count, below_count):
    """Return the exact float64 midpoint of the central float32 order statistics."""
    ranks = ((positive_count-1)//2-below_count, positive_count//2-below_count)
    if not (0 <= ranks[0] <= ranks[1] < len(candidates)):
        raise ValueError('Global median lies outside the sampled bracket; rerun with wider --quantiles')
    ordered = np.partition(candidates, ranks)
    lo, hi = float(ordered[ranks[0]]), float(ordered[ranks[1]])
    return (lo+hi)/2, lo, hi


def compute(args):
    family, width, k, root = specification(args.task, args.cache_base)
    name = f'{family}_m{width}_k{k}'
    with np.load(_paper_location(args.original/'per_feature'/f'{name}.npz')) as z:
        original = {key:z[key] for key in z.files}
    provenance = json.loads(str(original['provenance']))
    source = root/'manifest.json'
    if not source.exists(): source = root/'plan.json'
    payload = source.read_bytes(); meta = json.loads(payload)
    assert hashlib.sha256(payload).hexdigest() == provenance['metadata_sha256']
    shards = meta.get('shards', meta.get('source_shards'))
    rows = int(original['rows'])
    assert rows == meta['expected_rows'] and len(shards) == meta['expected_shards']
    rng = np.random.default_rng(args.seed+args.task)
    selected = np.sort(rng.choice(rows, size=min(args.sample_rows, rows), replace=False))
    samples = []
    for i,s in enumerate(shards):
        start,stop = s['global_row_start'],s['global_row_stop']
        a,b = np.searchsorted(selected,[start,stop])
        if b>a:
            data = np.load(_paper_location(root/'shards'/s['relative_shard']/name/'data.npy'),mmap_mode='r')
            assert data.dtype == np.float32 and data.size == s['rows']*k
            v = np.asarray(data.reshape(s['rows'],k)[selected[a:b]-start]).reshape(-1)
            samples.append(v[v>0])
        if (i+1)%200==0: print(f'{name}: calibration {i+1}/{len(shards)} shards',flush=True)
    sample = np.concatenate(samples); del samples
    assert len(sample)>0 and np.all(np.isfinite(sample))
    lower,upper = np.quantile(sample,args.quantiles,method='nearest').astype(np.float32)
    sample_count = len(sample); sample_median = float(np.median(sample.astype(np.float64)))
    del sample
    print(f'{name}: sampled median {sample_median:.9g}; exact bracket [{lower:.9g}, {upper:.9g}]',flush=True)
    assert 0 < lower <= upper
    positive = np.zeros(width,np.int64); above = np.zeros(width,np.int64)
    row_above = np.zeros(rows,np.uint16)
    values,features,event_rows = [],[],[]
    bracket_count = 0; rows_done = 0
    for i,s in enumerate(shards):
        assert s['global_row_start']==rows_done and s['global_row_stop']==rows_done+s['rows']
        directory = root/'shards'/s['relative_shard']
        complete = json.loads((directory/'complete.json').read_text())
        assert complete['complete'] and complete['spec_id']==meta['spec_id']
        assert complete['source']['relative_shard']==s['relative_shard']
        o = complete['outputs'][name]
        assert o['shape']==[s['rows'],width] and o['top_k']==k
        arrays={}
        for component,dtype in [('data','float32'),('indices','int32'),('indptr','int32')]:
            entry=o['components'][component];p=directory/entry['file']
            assert p.stat().st_size==entry['size_bytes']
            arrays[component]=np.load(_paper_location(p),mmap_mode='r')
            assert str(arrays[component].dtype)==dtype and list(arrays[component].shape)==entry['shape']
        data,ids,ptr=(arrays[key] for key in ['data','indices','indptr'])
        assert len(ptr)==s['rows']+1 and ptr[0]==0 and ptr[-1]==len(data)==len(ids)
        assert np.all(np.diff(ptr)==k)
        before=positive.sum()
        accumulate_bounds(data,ids,k,upper,positive,above,row_above[rows_done:rows_done+s['rows']])
        assert positive.sum()-before==o['positive_nnz']
        positions=np.flatnonzero((data>=lower)&(data<=upper))
        bracket_count+=len(positions)
        if lower!=upper:
            values.append(np.asarray(data[positions]).copy())
            features.append(np.asarray(ids[positions]).copy())
            event_rows.append((positions//k+rows_done).astype(np.uint32))
        rows_done+=s['rows']
        if (i+1)%50==0 or i+1==len(shards):
            print(f'{name}: exact pass {i+1}/{len(shards)} shards; {bracket_count:,} bracket events',flush=True)
    assert rows_done==rows
    np.testing.assert_array_equal(positive,original['counts'][0])
    n=int(positive.sum());below=n-int(above.sum())-bracket_count
    assert below<=(n-1)//2 and n//2<below+bracket_count, 'Global median outside bracket; increase --quantiles range'
    if lower==upper:
        median=central_lo=central_hi=float(lower)
    else:
        combined=np.concatenate(values)
        median,central_lo,central_hi=exact_median(combined,n,below)
        del combined
        for v,f,r in zip(values,features,event_rows):
            # Cast explicitly: an even-sample median may lie BETWEEN float32 values.
            mask=v.astype(np.float64)>median
            above+=np.bincount(f[mask],minlength=width)
            np.add.at(row_above,r[mask],1)
    assert np.all(above<=positive) and int(above.sum())<=n//2
    np.testing.assert_equal(above.sum(),row_above.sum(dtype=np.int64))
    l0=row_above.astype(np.float64)
    stats=np.stack([original['row_l0_statistics'][0],
                    [l0.sum(),np.square(l0).sum(),l0.min(),l0.max()]])
    counts=np.stack([positive,above])
    calibration=dict(method='Exact pooled median of all strictly positive cached post-top-k activations',
        sample_rows=len(selected),sample_positive_values=sample_count,seed=args.seed+args.task,
        sample_quantiles=args.quantiles,sample_median=sample_median,
        bracket_lower=float(lower),bracket_upper=float(upper),bracket_events=bracket_count,
        below_bracket_events=below,positive_events=n,central_lower=central_lo,central_upper=central_hi,
        median_positive=median,median_rank_bracket_verified=True)
    provenance.update(original_statistics=str((args.original/'per_feature'/f'{name}.npz').resolve()),
                      calibration=calibration)
    args.output.joinpath('per_feature').mkdir(parents=True,exist_ok=True)
    np.savez_compressed(_paper_location(args.output/'per_feature'/f'{name}.npz'),feature_id=np.arange(width),rows=rows,
        thresholds=np.array([0,median],np.float64),threshold_modes=np.array(['zero','median_positive']),
        counts=counts,rates=counts/rows,row_l0_statistics=stats,
        activation_sum=original['activation_sum'],activation_sum_squares=original['activation_sum_squares'],
        activation_mean=original['activation_mean'],activation_std=original['activation_std'],
        activation_max=original['activation_max'],provenance=json.dumps(provenance))
    (args.output/'per_feature'/f'{name}.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(f'Saved {name}: exact positive median {median:.10g}; {int(above.sum()):,}/{n:,} positive events exceed it',flush=True)


def finish(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    bins=[];summaries=[];thresholds=[];tops=[];activation=[];all_data={}
    for task in range(18):
        family,width,k,_=specification(task,args.cache_base);name=f'{family}_m{width}_k{k}'
        with np.load(_paper_location(args.output/'per_feature'/f'{name}.npz')) as z: d={key:z[key] for key in z.files}
        rows=int(d['rows']);c=d['counts'];r=d['rates'];ts=d['thresholds']
        np.testing.assert_array_equal(r,c/rows)
        np.testing.assert_array_equal(c.sum(1),d['row_l0_statistics'][:,0])
        assert np.all(c[1]<=c[0])
        all_data[family,width]=d
        prov=json.loads(str(d['provenance']));cal=prov['calibration']
        thresholds.append(dict(family=family,width=width,top_k=k,rows=rows,threshold_zero=0,
            threshold_median_positive=float(ts[1]),positive_events=int(c[0].sum()),
            events_above_median=int(c[1].sum()),fraction_positive_events_above_median=float(c[1].sum()/c[0].sum()),
            sample_median=cal['sample_median'],central_lower=cal['central_lower'],central_upper=cal['central_upper']))
        positive_mean=np.divide(d['activation_sum'],c[0],out=np.zeros(width),where=c[0]>0)
        positive_std=np.sqrt(np.maximum(0,np.divide(d['activation_sum_squares'],c[0],out=np.zeros(width),where=c[0]>0)-positive_mean**2))
        records=[dict(feature_id=f,count_gt_0=int(c[0,f]),count_gt_median=int(c[1,f]),
            rate_gt_0=float(r[0,f]),rate_gt_median=float(r[1,f]),activation_mean=float(d['activation_mean'][f]),
            activation_std=float(d['activation_std'][f]),activation_max=float(d['activation_max'][f]),
            positive_activation_mean=float(positive_mean[f]) if c[0,f] else None,
            positive_activation_std=float(positive_std[f]) if c[0,f] else None) for f in range(width)]
        write_csv(args.output/'per_feature'/f'{name}.csv',records)
        total=rows*width;sm=float(d['activation_sum'].sum());sq=float(d['activation_sum_squares'].sum());npos=int(c[0].sum())
        a=dict(family=family,width=width,top_k=k,rows=rows,feature_example_pairs=total,positive_activations=npos,activation_mean=sm/total,
            activation_std=float(np.sqrt(max(0,sq/total-(sm/total)**2))),activation_min=0,
            activation_max=float(d['activation_max'].max()),positive_activation_mean=sm/npos,
            positive_activation_std=float(np.sqrt(max(0,sq/npos-(sm/npos)**2))),
            median_positive=float(ts[1]),zero_fraction=1-npos/total,dead_features=int((c[0]==0).sum()))
        for t,mode in enumerate(['zero','median_positive']):
            base=dict(family=family,width=width,threshold_mode=mode,activation_threshold=float(ts[t]),rows=rows)
            h,_=np.histogram(r[t],bins=EDGES);assert h.sum()==width
            bins.extend(dict(**base,bin_lower=float(EDGES[j]),bin_upper=float(EDGES[j+1]),
                interval=f'[{EDGES[j]:g}, {EDGES[j+1]:g}'+(']' if j==6 else ')'),feature_count=int(count)) for j,count in enumerate(h))
            summaries.append(dict(**base,zero_rate_features=int((c[t]==0).sum()),**describe(r[t])))
            ids=np.argsort(-r[t],kind='stable')[:512]
            tops.extend(dict(**base,rank=rank+1,feature_id=int(f),activation_rate=float(r[t,f])) for rank,f in enumerate(ids))
            s=d['row_l0_statistics'][t]
            a.update({f'row_l0_{mode}_mean':float(s[0]/rows),f'row_l0_{mode}_std':float(np.sqrt(max(0,s[1]/rows-(s[0]/rows)**2))),
                      f'row_l0_{mode}_min':int(s[2]),f'row_l0_{mode}_max':int(s[3])})
        activation.append(a)
    for name,records in [('thresholds',thresholds),('bin_counts',bins),('activation_rate_summary',summaries),('activation_summary',activation),('top_512',tops)]:
        write_csv(args.output/f'{name}.csv',records)
        (args.output/f'{name}.json').write_text(json.dumps(records,indent=2)+'\n')
    plots=args.output/'plots';plots.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    colors=plt.get_cmap('viridis')(np.linspace(.05,.95,9));grid=np.linspace(-8.5,0,600)
    titles=['Activation > 0','Activation > SAE positive median']
    def save(fig,name,close=True):
        for ext in ['png','pdf','svg']:fig.savefig(plots/f'{name}.{ext}',dpi=180,bbox_inches='tight')
        if close:plt.close(fig)
    for family in ['gemini','nemotron']:
        fig,axes=plt.subplots(1,2,figsize=(15,5),sharex=True,sharey=True)
        facet,fax=plt.subplots(3,3,figsize=(15,11),sharex=True,sharey=True)
        for i,width in enumerate(WIDTHS):
            d=all_data[family,width]
            for t in range(2):
                r=d['rates'][t];log=np.log10(r[r>0]);density=np.zeros_like(grid)
                if len(log)>1 and log.std(ddof=1)>0:
                    density=gaussian_kde(log,bw_method=.12/log.std(ddof=1))(grid)
                elif len(log):density=np.exp(-.5*((grid-log[0])/.12)**2)/(.12*np.sqrt(2*np.pi))
                zeros=int((r==0).sum())
                axes[t].plot(grid,density*len(log),color=colors[i],label=f'{width:,} ({zeros:,} zeros)')
                label=f'> {d["thresholds"][t]:.5g}' + (' (median)' if t else '') + f'; {zeros:,} zeros'
                fax.flat[i].plot(grid,density,label=label)
            fax.flat[i].set_title(f'{width:,} features');fax.flat[i].legend(fontsize=8)
        for ax,title in zip(axes,titles):
            ax.set_title(title);ax.set_xlabel('log₁₀ activation rate');ax.grid(alpha=.2)
            ax.legend(fontsize=8,title='SAE size (zero-rate count)')
        axes[0].set_ylabel('Features per log₁₀ activation-rate unit')
        fig.suptitle(f'{family.capitalize()} · unnormalized feature density · zero and median thresholds')
        facet.supxlabel('log₁₀ activation rate');facet.supylabel('Density among positive-rate features',x=.005)
        facet.suptitle(f'{family.capitalize()} · zero and median thresholds by SAE size')
        facet.tight_layout(rect=(.025,.015,1,.98))
        save(fig,f'{family}_log_rate_density',close=False)
        # Keep negligible Gaussian tails from setting hundreds of log-axis decades.
        peak=max(float(np.max(line.get_ydata())) for ax in axes for line in ax.lines)
        for ax in axes:
            ax.set_yscale('log',nonpositive='mask')
            ax.set_ylim(.1,max(1,peak*1.5))
            ax.legend(fontsize=8,title='SAE size (zero-rate count)',loc='upper center',bbox_to_anchor=(.5,-.17),ncol=3)
        fig.set_size_inches(15,7)
        fig.subplots_adjust(bottom=.30)
        fig.suptitle(f'{family.capitalize()} · unnormalized feature density · logarithmic y-axis')
        save(fig,f'{family}_log_rate_density_log_y',close=False)
        from matplotlib.ticker import MaxNLocator, ScalarFormatter
        upper=max(1,peak*1.1)
        root_ticks=MaxNLocator(nbins=6).tick_values(0,np.sqrt(upper))
        ticks=root_ticks[(root_ticks>=0)&(root_ticks<=np.sqrt(upper))]**2
        for ax in axes:
            ax.set_yscale('function',functions=(lambda y:np.sqrt(np.maximum(y,0)),lambda y:np.square(y)))
            ax.set_ylim(0,upper)
            ax.set_yticks(ticks)
            ax.yaxis.set_major_formatter(ScalarFormatter())
        for ax in axes:
            ax.get_legend().remove()
        # Figure 10 is reduced to manuscript width; enlarge all plot text.
        for ax in axes:
            ax.title.set_fontsize(22)
            ax.xaxis.label.set_fontsize(20)
            ax.tick_params(axis='both',labelsize=18)
            ax.yaxis.get_offset_text().set_fontsize(18)
        axes[0].set_ylabel('Features per\nlog₁₀ activation-rate unit',fontsize=20)
        fig.subplots_adjust(bottom=.27,top=.85)
        fig.legend(axes[0].lines,[f'{width:,}' for width in WIDTHS],
                   title='SAE size',loc='lower center',bbox_to_anchor=(.5,.01),
                   ncol=5,fontsize=16,title_fontsize=18,frameon=False)
        fig.suptitle(f'{family.capitalize()} · unnormalized feature density',fontsize=26)
        save(fig,f'{family}_log_rate_density_sqrt_y')
        save(facet,f'{family}_log_rate_density_by_size')
        for scale in ['linear','log']:
            fig,axes=plt.subplots(1,2,figsize=(15,5),sharex=True,sharey=True)
            for i,width in enumerate(WIDTHS):
                for t in range(2):
                    r=np.sort(all_data[family,width]['rates'][t])[::-1][:512]
                    axes[t].plot(np.arange(1,513),np.where(r>0,r,np.nan) if scale=='log' else r,color=colors[i],label=f'{width:,}')
            for ax,title in zip(axes,titles):
                ax.set_title(title);ax.set_xlabel('Feature rank (descending activation rate)');ax.set_xlim(1,512)
                ax.set_yscale(scale);ax.grid(alpha=.2);ax.legend(fontsize=8,title='SAE size')
            axes[0].set_ylabel('Activation rate (fraction of examples)')
            fig.suptitle(f'{family.capitalize()} · 512 highest activation rates per SAE')
            save(fig,f'{family}_top_512_{scale}')
    text='''# Zero and median-positive SAE activation experiments

All nine main SAE sizes (512–131,072), separately for Gemini and Nemotron. The two conditions are activation strictly greater than zero, and strictly greater than that SAE's exact pooled median of strictly positive post-top-k activations. Each SAE has one global median threshold; this is not a separate threshold per feature, and zeros are excluded when defining the median. The median is calculated on the full evaluation corpus, so these are descriptive full-corpus statistics, not held-out calibration results.

[Threshold values](thresholds.csv) · [Activation summary](activation_summary.csv) · [Rate summary](activation_rate_summary.csv) · [Bin counts](bin_counts.csv) · [Per-feature rates](per_feature/) · [Plots](plots/)

Gemini uses 89,827,558 examples and Nemotron 89,227,558. All widths within a family have identical coverage. Counts are exact int64; rates and moments are float64. Structural zero padding is excluded from positive counts. Threshold-zero counts and unthresholded activation moments use the full-corpus summaries; the median computation independently verifies every zero-threshold feature count.

The median is exact: a reproducible uniform sample of corpus rows supplies a narrow candidate interval (sample positive quantiles 0.47 and 0.53). The full pass counts all positive events below and above this interval and retains every event inside it. Both global central order statistics must lie inside the interval, which is explicitly checked. Partitioning the retained float32 values gives the exact central values; their float64 midpoint defines the median. Final comparisons use float64 so an even-sample midpoint is not rounded back to float32. Sampling affects computation efficiency, not the reported threshold or rate accuracy. Ties at the median are excluded by the strict > comparison. Thus at most half of all positive activation events exceed the median; individual feature rates remain unconstrained.

Per-feature NPZ counts/rates have shape (2, width), ordered zero then median_positive. Feature IDs are zero-based. JSON sidecars record the original cache/model provenance, sample seed and row count, bracket counts, central order statistics, and exact threshold. Population standard deviations are used. NPZ row-L0 statistics have columns sum, sum of squares, minimum, maximum. Continuous activation means/stds include implicit zeros; positive activation means/stds exclude zeros.

The seven rate bins are [0, 1e-6), [1e-6, 1e-5), [1e-5, 1e-4), [1e-4, 1e-3), [1e-3, .01), [.01, .1), [.1, 1]. Zero rates are in the first bin and counted separately. All bin totals equal the SAE width. Rate percentiles use linear interpolation.

Overlaid log10-rate KDEs are unnormalized: each curve integrates over the real line to the number of positive-rate features. Zero-rate features are omitted from logarithms and counted in legends. The bandwidth is 0.12 log10 units. Additional `*_log_rate_density_log_y` overlays use a logarithmic y-axis, with a displayed lower bound of 0.1 features per log10-rate unit to omit negligible KDE tails; densities retain the same count scaling. Additional `*_log_rate_density_sqrt_y` overlays use a square-root y-axis starting at zero, with tick labels in original density units and the same unnormalized KDE values. These square-root overlays have one shared legend listing only SAE sizes. Per-size 3×3 density panels retain unit-area KDEs. Top-512 lines are independently sorted within each SAE and condition; both linear and logarithmic rate axes are supplied. Ties are ordered by feature ID. PNG/PDF/SVG versions are included.

Reproduce from the repository root (requires the original full-corpus statistics):

```bash
for task in $(seq 0 17); do
  python scripts/summarize_sae_zero_median.py compute --task "$task"
done
python scripts/summarize_sae_zero_median.py finish
```

## Exact thresholds

| Family | SAE size | Median positive activation | Mean active features at >0 | Mean active features at >median |
|---|---:|---:|---:|---:|
'''
    for a in activation:
        text+=f'| {a["family"]} | {a["width"]:,} | {a["median_positive"]:.9g} | {a["row_l0_zero_mean"]:.5f} | {a["row_l0_median_positive_mean"]:.5f} |\n'
    for family in ['gemini','nemotron']:
        for mode in ['zero','median_positive']:
            text+=f'\n## {family.capitalize()}: {mode}\n\n| SAE size | [0, 1e-6) | [1e-6, 1e-5) | [1e-5, 1e-4) | [1e-4, 1e-3) | [1e-3, .01) | [.01, .1) | [.1, 1] |\n|---:|---:|---:|---:|---:|---:|---:|---:|\n'
            for width in WIDTHS:
                counts=[r['feature_count'] for r in bins if (r['family'],r['width'],r['threshold_mode'])==(family,width,mode)]
                assert len(counts)==7 and sum(counts)==width
                text+='| '+' | '.join(f'{v:,}' for v in [width,*counts])+' |\n'
    (args.output/'README.md').write_text(text)
    (args.output/'COMPLETE.json').write_text(json.dumps(dict(complete=True,sae_count=18,total_features=sum(WIDTHS)*2,
        threshold_conditions=['zero','median_positive'],bin_records=len(bins),ranked_records=len(tops),plot_files=36),indent=2)+'\n')
    print('Completed zero/median summaries and plots:',args.output,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['compute','finish'])
    p.add_argument('--task',type=int,choices=range(18))
    p.add_argument('--cache-base',type=Path,default=BASE)
    p.add_argument('--original',type=Path,default=ORIGINAL_OUT)
    p.add_argument('--output',type=Path,default=DEFAULT_OUT)
    p.add_argument('--sample-rows',type=int,default=10000)
    p.add_argument('--seed',type=int,default=20260910)
    p.add_argument('--quantiles',type=float,nargs=2,default=[.47,.53])
    a=p.parse_args()
    if a.command=='compute':
        if a.task is None:p.error('compute requires --task')
        if not (0<a.quantiles[0]<.5<a.quantiles[1]<1):p.error('quantiles must bracket 0.5')
        compute(a)
    else:finish(a)


if __name__=='__main__':main()
