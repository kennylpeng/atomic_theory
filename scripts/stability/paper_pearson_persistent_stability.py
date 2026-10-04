"""Gemini persistent stability on retained, full-corpus Pearson neighbor edges."""
from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.sparse import coo_matrix, save_npz
from scripts.stability.persistent_matching import persistent_matching, validate_witness

ROOT = _paper_path(__file__).resolve().parents[2]
SOURCE = ROOT/'full_experiments/results/gemini_cross_width_sketch512_candidates32_top5'
OUTPUT = ROOT/'full_experiments/results/persistent_stability_pearson_0.7'


def threshold_graph(forward_ids, forward_scores, reverse_ids, reverse_scores, threshold=.7):
    """Union retained edges in both directions, with source/target orientation fixed."""
    n, m = len(forward_ids), len(reverse_ids)
    rr, cc = [], []
    diagnostics = {}
    for name, ids, scores, target_width, reverse in (
        ('forward', forward_ids, forward_scores, m, False),
        ('reverse', reverse_ids, reverse_scores, n, True),
    ):
        assert ids.shape == scores.shape and ids.ndim == 2
        assert np.all((ids >= -1) & (ids < target_width))
        valid = (ids >= 0) & np.isfinite(scores) & (scores >= np.float32(threshold))
        rows, slots = np.nonzero(valid)
        targets = ids[rows, slots]
        rr.append(targets if reverse else rows)
        cc.append(rows if reverse else targets)
        diagnostics[name+'_qualifying_saved_edges'] = int(valid.sum())
        diagnostics[name+'_rows_with_all_saved_slots_above_threshold'] = int(valid.all(axis=1).sum())
    graph = coo_matrix((np.ones(sum(map(len, rr)), dtype=bool),
                       (np.concatenate(rr), np.concatenate(cc))), shape=(n, m)).tocsr()
    graph.sum_duplicates(); graph.sort_indices()
    diagnostics['union_edges'] = int(graph.nnz)
    return graph, diagnostics


def plot(rows, output, family="gemini"):
    display_name=family.capitalize()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, PercentFormatter
    from scripts.stability.stability_plotting import apply_stability_grid, apply_stability_style
    apply_stability_style()
    widths = [r['width'] for r in rows]
    labels = [str(w) if w < 1024 else f'{w//1024}K' for w in widths]
    baseline = json.loads((ROOT/f'full_experiments/results/persistent_stability_absolute_0.7/{family}/summary.json').read_text())
    assert [r['width'] for r in baseline] == widths
    for comparison in (False, True):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), layout='constrained')
        for ax, key in zip(axes, ['persistent_count','proportion']):
            ax.plot(widths, [r[key] for r in rows], color='#0072B2', marker='o', linewidth=2.5,
                    label=f'{display_name}: Pearson ≥ 0.7')
            if comparison:
                ax.plot(widths, [r[key] for r in baseline], color='#777777', marker='s', linestyle='--',
                        linewidth=2, label=f'{display_name}: |decoder cosine| ≥ 0.7')
            ax.set_xscale('log', base=2); ax.set_xticks(widths, labels)
            ax.set_xlabel('SAE width'); apply_stability_grid(ax, proportion=key=='proportion')
            ax.spines[['top','right']].set_visible(False); ax.legend(frameon=False)
            if key == 'proportion':
                ax.set_ylim(0,1); ax.set_ylabel('persistent feature proportion'); ax.yaxis.set_major_formatter(PercentFormatter(1))
            else:
                ax.set_ylim(bottom=0); ax.set_ylabel('Persistent feature count'); ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:,.0f}'))
        name = 'persistent_stability_comparison' if comparison else 'persistent_stability'
        for ext in ('png','pdf'):fig.savefig(output/f'{name}.{ext}',dpi=220)
        plt.close(fig)


def compute(source=SOURCE, output=OUTPUT, family="gemini"):
    output.mkdir(parents=True,exist_ok=True)
    source_manifest=json.loads((source/'manifest.json').read_text())
    source_summary=json.loads((source/'summary.json').read_text())
    assert source_summary['complete'] and source_summary['manifest_id']==source_manifest['id']
    widths=[m['width'] for m in source_manifest['models']]
    manifest=dict(model=family, threshold=.7, comparison='signed Pearson >= float32(0.7)',
                  selection='intersection of independently selected pairwise maximum matchings into every larger width',
                  edge_source='union of qualifying saved top-five neighbors in both directions',
                  source_manifest_id=source_manifest['id'], source_directory=str(source),
                  source_files_sha256={}, rows=source_manifest['rows'], widths=widths,
                  exhaustive=False, interpretation='candidate-restricted independent intersection; no guaranteed bound on exhaustive intersection',
                  implementation_sha256=hashlib.sha256(_paper_path(__file__).read_bytes()).hexdigest())
    results=[]
    for i,width in enumerate(widths[:-1]):
        started=time.monotonic();graphs=[];diagnostics=[]
        targets=widths[i+1:]
        for target in targets:
            paths=[source/f'{family}_{width}_to_{target}_top5.npz',source/f'{family}_{target}_to_{width}_top5.npz']
            for path in paths:manifest['source_files_sha256'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(_paper_location(paths[0])) as f, np.load(_paper_location(paths[1])) as r:
                for data,sw,tw in [(f,width,target),(r,target,width)]:
                    assert str(data['manifest_id'])==source_manifest['id']
                    assert int(data['rows'])==source_manifest['rows']
                    assert int(data['source_width'])==sw and int(data['target_width'])==tw
                    assert np.array_equal(data['source_feature_ids'], np.arange(sw))
                graph,diag=threshold_graph(f['target_feature_ids'],f['pearson'],r['target_feature_ids'],r['pearson'])
            graphs.append(graph);diagnostics.append(dict(target_width=target,**diag))
            save_npz(_paper_location(output/f'graph_{width}_to_{target}.npz'),graph)
        selected,destinations,stats=persistent_matching(graphs)
        validate_witness(graphs,selected,destinations)
        np.savez_compressed(_paper_location(output/f'witness_{width}.npz'),source=selected,
                            **{f'target_{w}':d for w,d in zip(targets,destinations)})
        row=dict(model=family,width=width,persistent_count=len(selected),proportion=len(selected)/width,
                 target_widths=targets,threshold=.7,elapsed_seconds=time.monotonic()-started,**stats)
        (output/f'result_{width}.json').write_text(json.dumps(dict(**row,graphs=diagnostics),indent=2)+'\n')
        results.append(row)
        print(json.dumps(row),flush=True)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    with (output/'counts.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['model','width','persistent_count','proportion','candidates','certificate'])
        writer.writeheader();writer.writerows({k:r[k] for k in writer.fieldnames} for r in results)
    plot(results,output,family)
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family',choices=['gemini','nemotron'],default='gemini')
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args();compute(args.source,args.output,args.family)
