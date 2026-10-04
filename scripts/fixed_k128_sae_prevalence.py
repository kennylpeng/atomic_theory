"""Exact full-corpus positive-median prevalence for the six extra k=128 SAEs.

Read original embedding shards once per family, evaluating all three encoders.
Keep only counts and activation events in a sampled median bracket. The full
corpus rank check makes the final median exact, independent of the sample.
"""
from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np

from scripts import train_kmeans_full_corpus as common

ROOT = common.ROOT
OUT = ROOT / 'full_experiments/results/fixed_k128_sae_prevalence'
ORIGINAL = ROOT / 'full_experiments/results/main_sae_activation_statistics_zero_median'
WIDTHS = (512, 4096, 32768)
ALL_WIDTHS = (*WIDTHS, 65536, 131072)
FAMILIES = ('gemini', 'nemotron')
TOP_K = 128


def immutable(path, value):
    if path.exists():
        assert common.read(path) == value, path
    else:
        common.write(path, value)


def save_npz(path, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.stem}.{os.getpid()}.tmp.npz')
    np.savez_compressed(_paper_location(temporary), **values)
    temporary.replace(path)


def code_hashes():
    return dict(runner_sha256=common.digest(__file__), common_sha256=common.digest(common.__file__))


def prepare(args):
    for family in FAMILIES:
        plan_path = common.OUT / family / 'plan.json'
        plan = common.read(plan_path)
        models = []
        for width in WIDTHS:
            reference = ROOT / 'full_experiments/results/prefix_recovery_k128/wordfreq' / f'{family}_m{width}_k128/checkpoint.json'
            record = common.read(reference)
            config, path = record['config'], _paper_path(record['path'])
            assert config['dict_size'] == width and config['top_k'] == TOP_K
            assert config['act_size'] == plan['dimension'] and not config['input_unit_norm']
            assert common.digest(path) == record['sha256']
            st = path.stat()
            models.append(dict(width=width, top_k=TOP_K, path=str(path), sha256=record['sha256'],
                size=st.st_size, mtime_ns=st.st_mtime_ns, reference=str(reference),
                reference_sha256=common.digest(reference), config=config))
        reused = []
        for width in ALL_WIDTHS[-2:]:
            path = ORIGINAL / 'per_feature' / f'{family}_m{width}_k128.npz'
            with np.load(_paper_location(path), allow_pickle=False) as z:
                prov = json.loads(str(z['provenance']))
                assert int(z['rows']) == plan['rows'] and prov['top_k'] == TOP_K
                assert prov['family'] == family and prov['width'] == width
                assert prov['calibration']['median_rank_bracket_verified']
                assert z['threshold_modes'].tolist() == ['zero', 'median_positive']
            reused.append(dict(width=width, path=str(path), sha256=common.digest(path)))
        selected = set(np.linspace(0, len(plan['shards'])-1, 16, dtype=int).tolist())
        datasets = set()
        for i, shard in enumerate(plan['shards']):
            common.stat_matches(shard['path'], shard['size'], shard['mtime_ns'])
            dataset = shard['relative_shard'].split('/')[0]
            if dataset not in datasets:
                selected.add(i)
                datasets.add(dataset)
        manifest = dict(family=family, dimension=plan['dimension'], rows=plan['rows'], shards=plan['shards'],
            models=models, reused=reused, widths=list(ALL_WIDTHS), compute_widths=list(WIDTHS), top_k=TOP_K,
            source_plan=str(plan_path), source_plan_sha256=common.digest(plan_path),
            tasks=args.tasks, batch_rows=args.batch_rows, sample_rows=args.sample_rows,
            seed=20261001+FAMILIES.index(family), sample_quantiles=[.47, .53],
            audit_shards=sorted(selected), audit_datasets=sorted(datasets),
            activation='TopK_128(relu((embedding - b_dec) @ W_enc)); float32; TF32 disabled',
            preprocessing='Original float16 embeddings converted to float32; no normalization',
            storage='No embedding copies or full activation cache; only counts and median-bracket events',
            threshold='Exact pooled median of strictly positive activations; strict > comparison in float64',
            **code_hashes())
        manifest['id'] = common.object_digest(manifest)
        immutable(args.output / family / 'manifest.json', manifest)
        print(f'Prepared {family}: {plan["rows"]:,} rows, three new SAEs, two existing summaries.', flush=True)


def manifest_at(output, family):
    m = common.read(output / family / 'manifest.json')
    assert m['id'] == common.object_digest({k: v for k, v in m.items() if k != 'id'})
    assert all(m[k] == v for k, v in code_hashes().items())
    return m


def load_models(manifest, device):
    import torch
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    models, cpu_models = {}, {}
    for record in manifest['models']:
        common.stat_matches(record['path'], record['size'], record['mtime_ns'])
        assert common.digest(record['path']) == record['sha256']
        checkpoint = torch.load(_paper_location(record['path']), map_location='cpu', weights_only=False, mmap=True)
        assert json.loads(json.dumps(checkpoint['config'], default=str)) == record['config']
        state = checkpoint['model_state_dict']
        w, b = state['W_enc'].detach().float(), state['b_dec'].detach().float()
        width = record['width']
        assert tuple(w.shape) == (manifest['dimension'], width) and tuple(b.shape) == (manifest['dimension'],)
        assert torch.isfinite(w).all() and torch.isfinite(b).all()
        models[width] = (w.to(device), b.to(device))
        cpu_models[width] = (w.numpy(), b.numpy())
    return models, cpu_models


def encode(x, model, top_k=TOP_K):
    import torch
    weights, bias = model
    activation = torch.relu((x-bias) @ weights)
    values, indices = torch.topk(activation, k=top_k, dim=1, largest=True, sorted=True)
    return values.cpu().numpy(), indices.cpu().numpy()


def calibrate(args):
    import torch
    manifest = manifest_at(args.output, args.family)
    target = args.output / args.family / 'calibration.json'
    if target.exists():
        assert common.read(target)['manifest_id'] == manifest['id']
        return
    models, _ = load_models(manifest, args.device)
    selected = np.sort(np.random.default_rng(manifest['seed']).choice(
        manifest['rows'], min(manifest['sample_rows'], manifest['rows']), replace=False))
    samples = {width: [] for width in models}
    started = time.monotonic()
    with torch.inference_mode():
        for i, shard in enumerate(manifest['shards']):
            common.stat_matches(shard['path'], shard['size'], shard['mtime_ns'])
            a, b = np.searchsorted(selected, [shard['offset'], shard['offset']+shard['rows']])
            if a == b:
                continue
            x = np.load(_paper_location(shard['path']), mmap_mode='r')
            batch = torch.from_numpy(np.array(x[selected[a:b]-shard['offset']], dtype=np.float32)).to(args.device)
            x._mmap.close()
            assert torch.isfinite(batch).all()
            for width, model in models.items():
                values, _ = encode(batch, model)
                samples[width].append(values[values > 0])
            if (i+1) % 100 == 0:
                print(f'{args.family}: calibration {i+1}/{len(manifest["shards"])} shards; '
                      f'{time.monotonic()-started:.1f}s', flush=True)
    records = {}
    for width, arrays in samples.items():
        values = np.concatenate(arrays)
        lo, hi = np.quantile(values, manifest['sample_quantiles'], method='nearest').astype(np.float32)
        assert 0 < lo <= hi and np.isfinite(hi)
        records[str(width)] = dict(lower=float(lo), upper=float(hi), positive_events=len(values),
                                   sample_median=float(np.median(values.astype(np.float64))))
    common.write(target, dict(complete=True, manifest_id=manifest['id'], sample_rows=len(selected),
        sampled_row_ids_sha256=common.object_digest(selected.tolist()), models=records,
        seed=manifest['seed'], sample_quantiles=manifest['sample_quantiles']))
    print(f'{args.family} calibration complete: {records}', flush=True)


def summarize_events(values, ids, width, lower, upper):
    assert values.dtype == np.float32 and values.shape == ids.shape
    assert np.isfinite(values).all() and values.min() >= 0
    assert ids.min() >= 0 and ids.max() < width
    positive = values > 0
    bracket = (values >= lower) & (values <= upper)
    if lower == upper:
        candidate_values, candidate_ids = np.empty(0, np.float32), np.empty(0, np.uint16)
    else:
        candidate_values, candidate_ids = values[bracket].copy(), ids[bracket].astype(np.uint16)
    return dict(positive=np.bincount(ids[positive], minlength=width).astype(np.int64),
                above=np.bincount(ids[values > upper], minlength=width).astype(np.int64),
                below=int((positive & (values < lower)).sum()), bracket_count=int(bracket.sum()),
                values=candidate_values, ids=candidate_ids)


def check_encoding(vectors, values, ids, cpu_model, top_k=TOP_K):
    w, b = cpu_model
    ref = np.maximum(0, (vectors.astype(np.float64)-b.astype(np.float64)) @ w.astype(np.float64))
    reference_selected = np.take_along_axis(ref, ids, axis=1)
    error = float(np.abs(reference_selected-values).max())
    np.testing.assert_allclose(values, reference_selected, rtol=5e-5, atol=2e-6)
    boundary = np.partition(ref, -top_k, axis=1)[:, -top_k]
    gap = float(np.maximum(0, boundary-reference_selected.min(1)).max())
    assert gap <= 1e-5
    for row in ids:
        assert len(np.unique(row)) == top_k
    return dict(rows=len(vectors), activation_values=int(values.size), max_float64_absolute_error=error,
                max_topk_boundary_gap=gap, topk_gap_tolerance=1e-5)


def compute(args):
    import torch
    manifest = manifest_at(args.output, args.family)
    cal_path = args.output / args.family / 'calibration.json'
    calibration = common.read(cal_path)
    assert calibration['complete'] and calibration['manifest_id'] == manifest['id']
    cal_hash = common.digest(cal_path)
    assert 0 <= args.task < manifest['tasks']
    models, cpu_models = load_models(manifest, args.device)
    assigned = list(range(args.task, len(manifest['shards']), manifest['tasks']))
    started = time.monotonic()
    for position, index in enumerate(assigned):
        shard = manifest['shards'][index]
        common.stat_matches(shard['path'], shard['size'], shard['mtime_ns'])
        folder = args.output / args.family / 'shards' / shard['relative_shard']
        if (folder / 'complete.json').exists():
            done = common.read(folder / 'complete.json')
            assert done['manifest_id'] == manifest['id'] and done['shard'] == shard and done['calibration_sha256'] == cal_hash
            assert all(common.digest(folder / f'k{w}.npz') == d['sha256'] for w, d in done['models'].items())
            continue
        x = np.load(_paper_location(shard['path']), mmap_mode='r')
        assert x.shape == (shard['rows'], manifest['dimension']) and x.dtype == np.float16
        accum = {w: dict(positive=np.zeros(w, np.int64), above=np.zeros(w, np.int64), below=0,
                        bracket_count=0, values=[], ids=[]) for w in models}
        audit_rows = np.unique([len(x)//3, 2*len(x)//3]) if index in manifest['audit_shards'] else np.empty(0, np.int64)
        audits = []
        with torch.inference_mode():
            for start in range(0, len(x), manifest['batch_rows']):
                stop = min(start+manifest['batch_rows'], len(x))
                array = np.array(x[start:stop], dtype=np.float32)
                batch = torch.from_numpy(array).to(args.device)
                assert torch.isfinite(batch).all()
                checked = audit_rows[(audit_rows >= start) & (audit_rows < stop)]-start
                for width, model in models.items():
                    values, ids = encode(batch, model)
                    bracket = calibration['models'][str(width)]
                    part = summarize_events(values, ids, width, bracket['lower'], bracket['upper'])
                    state = accum[width]
                    for key in ('positive', 'above', 'below', 'bracket_count'):
                        state[key] += part[key]
                    for key in ('values', 'ids'):
                        state[key].append(part[key])
                    if len(checked):
                        audits.append(dict(width=width, shard_rows=(checked+start).tolist(),
                            **check_encoding(array[checked], values[checked], ids[checked], cpu_models[width])))
        x._mmap.close()
        common.stat_matches(shard['path'], shard['size'], shard['mtime_ns'])
        outputs = {}
        for width, state in accum.items():
            for key in ('values', 'ids'):
                state[key] = np.concatenate(state[key])
            assert int(state['positive'].sum()) == int(state['above'].sum())+state['below']+state['bracket_count']
            assert np.all(state['above'] <= state['positive']) and state['positive'].max() <= shard['rows']
            path = folder / f'k{width}.npz'
            save_npz(_paper_location(path), **state)
            outputs[str(width)] = dict(sha256=common.digest(path), positive_events=int(state['positive'].sum()),
                                      bracket_count=state['bracket_count'])
        common.write(folder / 'complete.json', dict(complete=True, manifest_id=manifest['id'], shard=shard,
            calibration_sha256=cal_hash, models=outputs, audit=audits))
        print(f'{args.family} task {args.task}: {position+1}/{len(assigned)} shards; '
              f'{time.monotonic()-started:.1f}s', flush=True)
    common.write(args.output / args.family / 'tasks' / f'{args.task:03d}.json',
        dict(complete=True, manifest_id=manifest['id'], calibration_sha256=cal_hash, indices=assigned,
             elapsed_seconds=time.monotonic()-started))


def median_from_candidates(values, positive_events, below, bracket_count, lower, upper):
    ranks = ((positive_events-1)//2-below, positive_events//2-below)
    assert 0 <= ranks[0] <= ranks[1] < bracket_count, 'Exact median outside sampled bracket; rerun with a wider bracket'
    if lower == upper:
        assert len(values) == 0
        lo = hi = float(lower)
    else:
        assert values.dtype == np.float32 and len(values) == bracket_count
        values.partition(ranks)
        lo, hi = float(values[ranks[0]]), float(values[ranks[1]])
    return (lo+hi)/2, lo, hi


def reduce(args):
    manifest = manifest_at(args.output, args.family)
    width = args.width
    assert width in manifest['compute_widths']
    cal_path = args.output / args.family / 'calibration.json'
    cal_hash = common.digest(cal_path)
    calibration = common.read(cal_path)
    assert calibration['manifest_id'] == manifest['id']
    bracket = calibration['models'][str(width)]
    lower, upper = bracket['lower'], bracket['upper']
    for task in range(manifest['tasks']):
        done = common.read(args.output / args.family / 'tasks' / f'{task:03d}.json')
        assert done['complete'] and done['manifest_id'] == manifest['id'] and done['calibration_sha256'] == cal_hash
        assert done['indices'] == list(range(task, len(manifest['shards']), manifest['tasks']))
    model = next(m for m in manifest['models'] if m['width'] == width)
    assert common.digest(model['path']) == model['sha256']
    assert common.digest(manifest['source_plan']) == manifest['source_plan_sha256']
    positives, above = np.zeros(width, np.int64), np.zeros(width, np.int64)
    below = bracket_count = rows = 0
    arrays, audits = [], []
    for index, shard in enumerate(manifest['shards']):
        folder = args.output / args.family / 'shards' / shard['relative_shard']
        done = common.read(folder / 'complete.json')
        assert done['complete'] and done['manifest_id'] == manifest['id'] and done['shard'] == shard
        assert done['calibration_sha256'] == cal_hash
        path = folder / f'k{width}.npz'
        assert common.digest(path) == done['models'][str(width)]['sha256']
        with np.load(_paper_location(path), allow_pickle=False) as z:
            positives += z['positive']
            above += z['above']
            below += int(z['below'])
            bracket_count += int(z['bracket_count'])
            assert int(z['positive'].sum()) == done['models'][str(width)]['positive_events']
            assert int(z['bracket_count']) == done['models'][str(width)]['bracket_count']
            if lower != upper:
                arrays.append(z['values'])
                assert len(arrays[-1]) == int(z['bracket_count'])
        selected = [a for a in done['audit'] if a['width'] == width]
        expected_rows = np.unique([shard['rows']//3, 2*shard['rows']//3]).tolist() if index in manifest['audit_shards'] else []
        assert sorted(r for a in selected for r in a['shard_rows']) == expected_rows
        assert all(a['max_topk_boundary_gap'] <= a['topk_gap_tolerance'] == 1e-5 for a in selected)
        audits.extend(selected)
        rows += shard['rows']
        if (index+1) % 100 == 0:
            print(f'{args.family} {width}: gathered {index+1}/{len(manifest["shards"])} shards', flush=True)
    assert rows == manifest['rows']
    npositive = int(positives.sum())
    assert npositive == int(above.sum())+below+bracket_count
    candidates = np.concatenate(arrays) if arrays else np.empty(0, np.float32)
    del arrays
    median, lo, hi = median_from_candidates(candidates, npositive, below, bracket_count, lower, upper)
    del candidates
    if lower != upper:
        for shard in manifest['shards']:
            path = args.output / args.family / 'shards' / shard['relative_shard'] / f'k{width}.npz'
            with np.load(_paper_location(path), allow_pickle=False) as z:
                values, ids = z['values'], z['ids']
                assert ids.dtype == np.uint16 and ids.shape == values.shape
                above += np.bincount(ids[values.astype(np.float64) > median], minlength=width)
    assert np.all(above <= positives) and positives.max() <= rows
    assert int(above.sum()) <= npositive//2
    counts = np.stack([positives, above])
    provenance = dict(family=args.family, width=width, top_k=TOP_K, rows=rows,
        manifest_id=manifest['id'], model=model, activation=manifest['activation'], calibration=dict(
            lower=lower, upper=upper, sample_positive_events=bracket['positive_events'],
            sample_median=bracket['sample_median'], seed=calibration['seed'], sample_rows=calibration['sample_rows'],
            method='Exact pooled positive median from full-corpus bracket events',
            positive_events=npositive, below_bracket_events=below, bracket_events=bracket_count,
            median_positive=median, central_lower=lo, central_upper=hi, median_rank_bracket_verified=True),
        strict_threshold_comparison='Candidate float32 values promoted to float64 before > median',
        independent_float64_encoding_audits=audits)
    name = f'{args.family}_m{width}_k128'
    path = args.output / 'per_feature' / f'{name}.npz'
    save_npz(_paper_location(path), feature_id=np.arange(width), rows=rows, counts=counts, rates=counts/rows,
             thresholds=np.array([0, median], np.float64), threshold_modes=np.array(['zero', 'median_positive']),
             provenance=json.dumps(provenance))
    common.write(path.with_suffix('.json'), provenance)
    common.write(args.output / 'completed' / f'{name}.json', dict(complete=True, manifest_id=manifest['id'],
        width=width, family=args.family, rows=rows, path=str(path), sha256=common.digest(path), threshold=median,
        zero_rate_features=int((above == 0).sum()), positive_events=npositive, above_median_events=int(above.sum())))
    print(f'Complete {name}: exact median {median:.10g}; {int(above.sum()):,} events exceed it.', flush=True)


def finish(args):
    from scripts.plot_fixed_k128_sae_kmeans_prevalence import render
    for family in FAMILIES:
        m = manifest_at(args.output, family)
        for width in WIDTHS:
            done = common.read(args.output / 'completed' / f'{family}_m{width}_k128.json')
            assert done['complete'] and done['manifest_id'] == m['id']
            assert common.digest(done['path']) == done['sha256']
        for reused in m['reused']:
            assert common.digest(reused['path']) == reused['sha256']
    records = render(args.output)
    common.write(args.output / 'summary.json', dict(complete=True, top_k=TOP_K, widths=list(ALL_WIDTHS),
        families=list(FAMILIES), models=records, completed_utc=datetime.now(timezone.utc).isoformat(),
        definition='Full-corpus prevalence above each SAE exact pooled positive median',
        validation='All corpus rows; frozen models; float64 encoder checks; exact central-rank verification; '
                   'strict threshold midpoint handling; shared 65K/131K source summaries; KDE count integrals.'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'calibrate', 'compute', 'reduce', 'finish'])
    p.add_argument('--output', type=Path, default=OUT)
    p.add_argument('--family', choices=FAMILIES)
    p.add_argument('--width', type=int, choices=WIDTHS)
    p.add_argument('--task', type=int, default=0)
    p.add_argument('--tasks', type=int, default=32)
    p.add_argument('--sample-rows', type=int, default=10000)
    p.add_argument('--batch-rows', type=int, default=2048)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    globals()[args.stage](args)


if __name__ == '__main__':
    main()
