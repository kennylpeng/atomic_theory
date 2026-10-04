"""Directional split-SAE matching using distribution-specific activation sketches.

Each evaluation distribution gets its own centered 512-dimensional CountSketch,
32 candidates per feature/direction, and exact full-distribution Pearson rescoring.
The primary graph unions the saved top five in both feature-search directions,
just like the existing activation-persistence analysis. Both searches use the
SAME evaluation distribution; transposed heatmap cells use different data.
"""
from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from scipy.sparse import save_npz
from scipy.sparse.csgraph import maximum_bipartite_matching
from scripts import match_gemini_width_sketch as core
from scripts.stability.paper_pearson_persistent_stability import threshold_graph
from scripts.stability.validate_split_cardinality_outputs import certify_maximum

ROOT = _paper_path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'full_experiments/results/split_sae_activation_sketch512_candidates32_top5'
SPLITS = ('main', 'wikipedia', 'no_wikipedia', 'random1', 'random2')
LABELS = ('full', 'wiki', 'no_wiki', 'rand1', 'rand2')
FAMILIES = ('gemini', 'nemotron')


def digest(path):
    return hashlib.sha256(_paper_path(path).read_bytes()).hexdigest()


def directions(m):
    e = m['evaluation_index']
    return [(i, j) for i in range(len(m['models'])) for j in range(len(m['models']))
            if i != j and e in (i, j)]


def prepare(args):
    family = args.family
    cache_roots = [core.CACHE / (('gemini_m16384_all_corpus_post_topk' if split == 'main'
                    else 'gemini_remaining_experiment_saes_post_topk') if family == 'gemini'
                    else 'nemotron_all_experiment_saes_post_topk') for split in SPLITS]
    plans = [core.read_json(root / 'plan.json') for root in cache_roots]
    reference = plans[0]
    by_name = lambda plan: {s['relative_shard']: s for s in plan['source_shards']}
    ref_shards = by_name(reference)
    models = []
    for split, root, plan in zip(SPLITS, cache_roots, plans):
        assert plan['complete'] and by_name(plan) == ref_shards, 'Unaligned activation caches'
        name = f'{family}_m16384_k64' if split == 'main' else f'{family}_{split}_m16384_k64'
        model = next(x for x in plan['models'] if x['name'] == name)
        stat = _paper_path(model['path']).stat()
        assert stat.st_size == model['size_bytes'] and stat.st_mtime_ns == model['mtime_ns'], name
        models.append(dict(name=name, split=split, width=16384, top_k=64, cache=str(root),
                           sha256=model['sha256'], cache_spec_id=plan['spec_id']))
    meta_path = ROOT / f'model_configs/baselines/{family}/{args.evaluation}/pca_metadata.json'
    meta = core.read_json(meta_path)
    datasets = sorted(meta['datasets'])
    # Cross-check membership against the actual SAE training configuration.
    if args.evaluation != 'main':
        import yaml
        token = dict(wikipedia='wikipedia', no_wikipedia='no-wikipedia', random1='random-1', random2='random-2')[args.evaluation]
        configs = list((ROOT/'configs_test').glob(f'{family}_{token}_d16384_k64_*.yaml'))
        assert len(configs) == 1
        assert sorted(yaml.safe_load(configs[0].read_text())['data']['datasets']) == datasets
    shards = []
    for s in reference['source_shards']:
        if s['relative_shard'].split('/')[0] in datasets:
            shards.append(dict(relative_shard=s['relative_shard'], rows=int(s['rows']),
                               offset=int(s['global_row_start'])))
    rows = sum(s['rows'] for s in shards)
    assert rows == meta['row_count'], (rows, meta['row_count'])
    assert len(set(s['relative_shard'] for s in shards)) == len(shards)
    m = dict(family=family, evaluation_split=args.evaluation, evaluation_index=SPLITS.index(args.evaluation),
             models=models, shards=shards, rows=rows, datasets=datasets, tasks=args.tasks,
             sketch_dim=512, candidate_k=32, save_k=5, threshold=0.7,
             hash='SplitMix64, original family full-corpus global row offsets (gaps retained)',
             weighting='uniform over all cached rows belonging to the evaluation distribution',
             metric='signed continuous post-TopK activation Pearson correlation', candidate_limited=True,
             sketch_dtype='float32 partials; float64 reduction', moments_and_products_dtype='float64',
             dataset_metadata=str(meta_path), dataset_metadata_sha256=digest(meta_path),
             source_code_sha256=digest(__file__), core_code_sha256=digest(core.__file__))
    m['id'] = hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()
    path = args.output/'manifest.json'
    if path.exists():
        assert core.read_json(path) == m, 'Manifest changed; use a fresh output directory'
    core.atomic_json(path, m)
    print(f'{family}/{args.evaluation}: {rows:,} rows, {len(shards)} shards, {len(directions(m))} feature-search directions', flush=True)


def candidates(args, m):
    combined = {}
    rows = 0
    for t in range(m['tasks']):
        with core.checked(core.task_path(args, 'sketch', t), m) as d:
            expected = sum(s['rows'] for s in m['shards'][t::m['tasks']])
            assert int(d['rows']) == expected
            rows += expected
            for key in ('sketch', 'sums', 'squares', 'support', 'ones'):
                value = d[key].astype(np.int64 if key == 'support' else np.float64)
                if key in combined:
                    combined[key] += value
                else:
                    combined[key] = value
    assert rows == m['rows']
    normalized, live = core.normalized_sketch(
        *(torch.as_tensor(combined[k], device=args.device) for k in ('sketch', 'sums', 'squares', 'ones')), rows)
    core.save(args.output/'sketch.npz', **combined, normalized=normalized.cpu().numpy(),
              live=live.cpu().numpy(), rows=rows, manifest_id=m['id'])
    core.save(args.output/'moments.npz', **{k:v for k,v in combined.items() if k != 'sketch'},
              rows=rows, manifest_id=m['id'])
    offsets = np.cumsum([0] + [x['width'] for x in m['models']])
    result = {}
    for i, j in directions(m):
        a, b = slice(offsets[i], offsets[i+1]), slice(offsets[j], offsets[j+1])
        result[f'{i}_{j}'] = core.candidates_for(normalized[:,a], normalized[:,b], live[a], live[b],
                                                m['candidate_k']).cpu().numpy().astype(np.int32)
    core.save(args.output/'candidates.npz', **result, manifest_id=m['id'])
    print(f'Generated {len(result)} candidate arrays on {m["evaluation_split"]}', flush=True)


def rescore(args, m):
    pairs = directions(m)
    with core.checked(args.output/'candidates.npz', m) as c:
        candidate = {f'{i}_{j}':torch.as_tensor(c[f'{i}_{j}'].astype(np.int64), device=args.device) for i,j in pairs}
    dots = {key:torch.zeros(value.shape, dtype=torch.float64, device=args.device) for key,value in candidate.items()}
    sources = {j:[i for i,jj in pairs if j == jj] for j in range(len(m['models']))}
    assigned = m['shards'][args.task::m['tasks']]
    rows = 0
    start_time = time.monotonic()
    for si, shard in enumerate(assigned):
        caches = [core.Slots(model, shard) for model in m['models']]
        for start in range(0, shard['rows'], args.batch_rows):
            end = min(start+args.batch_rows, shard['rows'])
            batches = [cache.batch(start, end, args.device) for cache in caches]
            for j, model in enumerate(m['models']):
                dense = torch.zeros((end-start, model['width']), device=args.device)
                dense.scatter_add_(1, *batches[j])
                for i in sources[j]:
                    key = f'{i}_{j}'
                    core.accumulate_edges(dots[key], candidate[key], *batches[i], dense)
                del dense
        rows += shard['rows']
        if si % 10 == 0 or si == len(assigned)-1:
            print(f'Rescore {args.task}: {si+1}/{len(assigned)} shards, {rows:,} rows, {time.monotonic()-start_time:.1f}s', flush=True)
    core.save(core.task_path(args, 'rescore', args.task), **{k:v.cpu().numpy() for k,v in dots.items()},
              rows=rows, manifest_id=m['id'])


def match_graph(forward, reverse, threshold):
    graph, diag = threshold_graph(*forward, *reverse, threshold)
    assignment = maximum_bipartite_matching(graph, perm_type='column')
    source = np.flatnonzero(assignment >= 0)
    target = assignment[source]
    certify_maximum(graph, source, target)
    return graph, source, target, diag


def reduce(args, m):
    offsets = np.cumsum([0] + [x['width'] for x in m['models']])
    top, all_candidates = {}, {}
    with core.checked(args.output/'moments.npz', m) as moments, core.checked(args.output/'candidates.npz', m) as c:
        partials = [core.checked(core.task_path(args, 'rescore', t), m) for t in range(m['tasks'])]
        for t, p in enumerate(partials):
            assert int(p['rows']) == sum(s['rows'] for s in m['shards'][t::m['tasks']])
        assert sum(int(p['rows']) for p in partials) == m['rows'] == int(moments['rows'])
        for i,j in directions(m):
            key = f'{i}_{j}'
            dots = sum((p[key] for p in partials), np.zeros(c[key].shape, dtype=np.float64))
            a, b = slice(offsets[i], offsets[i+1]), slice(offsets[j], offsets[j+1])
            score = core.pearson_edges(dots, c[key], moments['sums'][a], moments['squares'][a],
                                        moments['sums'][b], moments['squares'][b], m['rows'])
            ids, values = core.best_five(c[key], score, m['save_k'])
            top[key] = ids, values
            all_candidates[key] = c[key], score.astype(np.float32)
            core.save(args.output/f'{m["models"][i]["split"]}_to_{m["models"][j]["split"]}_top5.npz',
                      target_feature_ids=ids, pearson=values, source_feature_ids=np.arange(len(ids), dtype=np.int32),
                      evaluation_split=m['evaluation_split'], rows=m['rows'], manifest_id=m['id'])
            core.save(args.output/f'{key}_candidates_rescored.npz', target_feature_ids=c[key], pearson=score,
                      rows=m['rows'], manifest_id=m['id'])
        for p in partials:
            p.close()
    e = m['evaluation_index']
    results = []
    for j in range(len(m['models'])):
        if j == e:
            continue
        graph, source, target, diag = match_graph(top[f'{e}_{j}'], top[f'{j}_{e}'], m['threshold'])
        full_graph, full_source, _, _ = match_graph(all_candidates[f'{e}_{j}'], all_candidates[f'{j}_{e}'], m['threshold'])
        assert len(full_source) >= len(source)
        split = m['models'][j]['split']
        save_npz(_paper_location(args.output/f'graph_to_{split}.npz'), graph)
        core.save(args.output/f'matching_to_{split}.npz', source=source, destination=target,
                  manifest_id=m['id'], evaluation_split=m['evaluation_split'])
        width = m['models'][e]['width']
        row = dict(family=m['family'], source_split=m['evaluation_split'], comparison_split=split,
                   evaluation_split=m['evaluation_split'], rows=m['rows'], threshold=m['threshold'],
                   matches=len(source), source_features=width, proportion=len(source)/width,
                   all32_matches=len(full_source), all32_edges=full_graph.nnz,
                   certified_on_retained_graph=True, candidate_limited=True, **diag)
        results.append(row)
        print(json.dumps(row), flush=True)
    core.atomic_json(args.output/'summary.json', dict(complete=True, manifest_id=m['id'], comparisons=results))


def plot(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    from scripts.plot_style import apply_plot_style
    apply_plot_style()
    rows = []
    manifest_ids = {}
    for family in FAMILIES:
        for evaluation in SPLITS:
            folder = root/family/evaluation
            m = core.read_json(folder/'manifest.json')
            summary = core.read_json(folder/'summary.json')
            assert summary['complete'] and summary['manifest_id'] == m['id']
            assert len(summary['comparisons']) == 4
            manifest_ids[f'{family}/{evaluation}'] = m['id']
            rows.extend(summary['comparisons'])
    assert len(rows) == 40
    with (root/'counts.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    matrices = {}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.7), layout='constrained')
    cmap = plt.get_cmap('viridis').copy(); cmap.set_bad('#f2f2f2')
    vmax = max(r['proportion'] for r in rows)
    vmax = min(1., np.ceil(vmax*10)/10)
    for family, ax in zip(FAMILIES, axes):
        matrix = np.full((5,5), np.nan)
        for r in rows:
            if r['family'] == family:
                matrix[SPLITS.index(r['source_split']), SPLITS.index(r['comparison_split'])] = r['proportion']
        matrices[family] = matrix
        im = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=0, vmax=vmax)
        ax.set_xticks(range(5), LABELS, rotation=35, ha='right')
        ax.set_yticks(range(5), LABELS)
        ax.set_xlabel('Comparison SAE training distribution')
        ax.set_title(f'{family.capitalize()} SAE activations')
        for i in range(5):
            for j in range(5):
                if i == j:
                    continue
                value = matrix[i,j]
                red, green, blue, _ = cmap(im.norm(value))
                color = 'black' if .2126*red+.7152*green+.0722*blue > .55 else 'white'
                ax.text(j, i, f'{100*value:.1f}%', ha='center', va='center', fontsize=11, color=color)
        with (root/f'{family}_proportions.csv').open('w') as f:
            writer=csv.writer(f);writer.writerow(['source_and_evaluation_distribution',*SPLITS])
            writer.writerows([split, *matrix[i]] for i,split in enumerate(SPLITS))
    axes[0].set_ylabel('Source SAE and evaluation distribution')
    cbar = fig.colorbar(im, ax=axes, fraction=.035, pad=.025, format=PercentFormatter(1))
    cbar.set_label('Matched feature proportion (Pearson ≥ 0.7)')
    for ext in ('png', 'pdf'):
        fig.savefig(root/f'activation_split_heatmaps_t0.7.{ext}', dpi=240)
    plt.close(fig)
    core.atomic_json(root/'summary.json', dict(complete=True, comparisons=rows, manifest_ids=manifest_ids,
        interpretation='Maximum-cardinality matchings on bidirectional top-five candidate graphs; lower bounds on exhaustive Pearson matching',
        evaluation='Row distribution for both SAEs; each transposed cell is independently estimated',
        candidate_truncation_maximum_count_difference=max(r['all32_matches']-r['matches'] for r in rows)))
    print(f'Created asymmetric activation heatmaps and counts for {len(rows)} comparisons in {root}', flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'sketch', 'candidates', 'rescore', 'reduce', 'plot'])
    parser.add_argument('--root', type=Path, default=OUTPUT)
    parser.add_argument('--family', choices=FAMILIES, default='gemini')
    parser.add_argument('--evaluation', choices=SPLITS, default='main')
    parser.add_argument('--tasks', type=int, default=8)
    parser.add_argument('--task', type=int, default=0)
    parser.add_argument('--batch-rows', type=int, default=2048)
    parser.add_argument('--device', default='cuda')
    args=parser.parse_args()
    torch.set_num_threads(4); torch.set_float32_matmul_precision('highest')
    if args.stage == 'plot':
        plot(args.root); return
    args.output=args.root/args.family/args.evaluation
    if args.stage == 'prepare':
        prepare(args); return
    m=core.read_json(args.output/'manifest.json')
    assert m['source_code_sha256'] == digest(__file__) and m['core_code_sha256'] == digest(core.__file__)
    assert 0 <= args.task < m['tasks']
    if args.stage in ('sketch', 'rescore') and core.task_path(args,args.stage,args.task).exists():
        with core.checked(core.task_path(args,args.stage,args.task),m) as d:
            assert int(d['rows']) == sum(s['rows'] for s in m['shards'][args.task::m['tasks']])
        print('Already complete:', args.stage, args.task); return
    if args.stage == 'sketch':
        core.sketch(args,m)
    else:
        globals()[args.stage](args,m)

if __name__ == '__main__':
    main()
