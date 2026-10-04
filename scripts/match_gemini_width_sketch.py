"""Reusable CountSketch candidates and full-corpus Pearson top-five across Gemini widths.

Results are candidate-limited, not certified global nearest neighbors. No thresholding.
"""
from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from scripts.matching_io import atomic_json, read_json
from scripts.aligned_activations import row_hash

WIDTHS = [512 * 2**i for i in range(9)]
TOPKS = [32, 32, 32, 32, 64, 64, 64, 128, 128]
CACHE = _paper_path('/resources/activation_cache_dir')
DEFAULT_OUT = _paper_path('full_experiments/results/gemini_cross_width_sketch512_candidates32_top5')


def save(path, **arrays):
    path = _paper_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with tmp.open('wb') as f:
        np.savez(_paper_location(f), **arrays)
    os.replace(tmp, path)


def root_for(width):
    name = ('gemini_all_corpus_post_topk' if width in (4096, 131072) else
            'gemini_m16384_all_corpus_post_topk' if width == 16384 else
            'gemini_remaining_experiment_saes_post_topk')
    return CACHE / name


def prepare(args):
    plans = [read_json(root_for(w) / 'plan.json') for w in WIDTHS]
    shards = []
    offset = 0
    for s in plans[0]['source_shards']:
        shards.append(dict(relative_shard=s['relative_shard'], rows=int(s['rows']), offset=offset))
        offset += int(s['rows'])
    reference = {s['relative_shard']: s['rows'] for s in shards}
    models = []
    for w, k, plan in zip(WIDTHS, TOPKS, plans):
        assert plan['complete']
        assert {s['relative_shard']: int(s['rows']) for s in plan['source_shards']} == reference
        name = f'gemini_m{w}_k{k}'
        model = next(m for m in plan['models'] if m['name'] == name)
        models.append(dict(width=w, top_k=k, name=name, cache=str(root_for(w)), sha256=model['sha256']))
    manifest = dict(models=models, shards=shards, rows=offset, sketch_dim=512,
                    candidate_k=32, save_k=5, tasks=args.tasks, seed='SplitMix64 shared global row hash',
                    metric='signed activation Pearson', candidate_limited=True,
                    sketch_dtype='float32', moments_and_candidate_products_dtype='float64',
                    source_code_sha256=hashlib.sha256(_paper_path(__file__).read_bytes()).hexdigest())
    manifest['id'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    path = args.output / 'manifest.json'
    if path.exists():
        assert read_json(path) == manifest, 'Existing manifest differs; use a new output directory'
    atomic_json(path, manifest)
    print(f'Prepared {len(models)} widths, {len(shards)} shards, {offset:,} rows', flush=True)


class Slots:
    def __init__(self, model, shard):
        folder = _paper_path(model['cache']) / 'shards' / shard['relative_shard']
        info = read_json(folder / 'complete.json')['outputs'][model['name']]
        assert tuple(info['shape']) == (shard['rows'], model['width'])
        assert info['top_k'] == model['top_k']
        folder /= model['name']
        self.ids = np.load(_paper_location(folder / 'indices.npy'), mmap_mode='r').reshape(shard['rows'], model['top_k'])
        self.values = np.load(_paper_location(folder / 'data.npy'), mmap_mode='r').reshape(self.ids.shape)
        pointers = np.load(_paper_location(folder / 'indptr.npy'), mmap_mode='r')
        assert np.array_equal(pointers, np.arange(shard['rows'] + 1, dtype=np.int64) * model['top_k'])

    def batch(self, start, end, device):
        return (torch.as_tensor(np.array(self.ids[start:end], dtype=np.int64), device=device),
                torch.as_tensor(np.array(self.values[start:end], dtype=np.float32), device=device))


def accumulate_sketch(sketch, sums, squares, support, ids, values, buckets, signs):
    locations = buckets[:, None] * sketch.shape[1] + ids
    sketch.view(-1).index_add_(0, locations.flatten(), (values * signs[:, None]).flatten())
    flat_ids, vd = ids.flatten(), values.flatten().double()
    sums.index_add_(0, flat_ids, vd)
    squares.index_add_(0, flat_ids, vd.square())
    support.index_add_(0, flat_ids, (vd != 0).long())


def normalized_sketch(sketch, sums, squares, ones, rows):
    var = squares - sums.square() / rows
    # Guard numerical cancellation for constant columns.
    live = var > 32 * torch.finfo(torch.float64).eps * squares.abs().clamp_min(1e-300)
    centered = sketch.double() - ones.double()[:, None] * (sums / rows)[None, :]
    centered /= torch.sqrt(torch.where(live, var, torch.ones_like(var)))[None, :]
    centered[:, ~live] = 0
    return centered.float(), live


def candidates_for(source, target, source_live, target_live, k, batch=256):
    ids = torch.full((source.shape[1], k), -1, dtype=torch.int64, device=source.device)
    for start in range(0, source.shape[1], batch):
        stop = min(start + batch, source.shape[1])
        score = source[:, start:stop].T @ target
        score[:, ~target_live] = -torch.inf
        count = min(k, target.shape[1])
        val, idx = torch.topk(score, count, dim=1)
        idx[~torch.isfinite(val)] = -1
        idx[~source_live[start:stop]] = -1
        ids[start:stop, :count] = idx
    return ids


def accumulate_edges(dots, candidates, ids, values, target_dense, chunk=16):
    rank = torch.arange(candidates.shape[1], device=ids.device)[None, :]
    rows = torch.arange(ids.shape[0], device=ids.device)[:, None]
    for start in range(0, ids.shape[1], chunk):
        src = ids[:, start:start + chunk]
        val = values[:, start:start + chunk]
        # Remove structural zero padding, including any repeated padding IDs.
        nz = val != 0
        src, val, row = src[nz], val[nz], rows.expand_as(src)[nz]
        dst = candidates[src]
        product = val.double()[:, None] * target_dense[row[:, None], dst.clamp_min(0)].double()
        product.masked_fill_(dst < 0, 0)
        dots.view(-1).index_add_(0, (src[:, None] * candidates.shape[1] + rank).flatten(), product.flatten())


def pearson_edges(dots, candidates, sx, qx, sy, qy, rows):
    vx, vy = qx - sx*sx/rows, qy - sy*sy/rows
    live_x = vx > 32*np.finfo(np.float64).eps*np.maximum(abs(qx), 1e-300)
    live_y = vy > 32*np.finfo(np.float64).eps*np.maximum(abs(qy), 1e-300)
    target = np.maximum(candidates, 0)
    valid = (candidates >= 0) & live_x[:, None] & live_y[target]
    numerator = dots - sx[:, None]*sy[target]/rows
    denominator = np.sqrt(np.maximum(vx[:, None], 0)*np.maximum(vy[target], 0))
    result = np.full(dots.shape, np.nan)
    np.divide(numerator, denominator, out=result, where=valid)
    return np.clip(result, -1, 1)


def best_five(candidates, scores, k=5):
    # Deterministic final ties: lower target feature ID first.
    order = np.lexsort((candidates, -np.where(np.isfinite(scores), scores, -np.inf)), axis=1)[:, :k]
    indices = np.take_along_axis(candidates, order, axis=1).copy()
    values = np.take_along_axis(scores, order, axis=1)
    indices[~np.isfinite(values)] = -1
    return indices.astype(np.int32), values.astype(np.float32)


def task_path(args, stage, task):
    return args.output / 'work' / f'{stage}_{task:03d}.npz'


def sketch(args, m):
    device = args.device
    width_sum = sum(x['width'] for x in m['models'])
    offsets = np.cumsum([0] + [x['width'] for x in m['models']])
    sketches = torch.zeros((m['sketch_dim'], width_sum), device=device)
    sums = torch.zeros(width_sum, device=device, dtype=torch.float64)
    squares = torch.zeros_like(sums)
    support = torch.zeros(width_sum, device=device, dtype=torch.int64)
    ones = torch.zeros(m['sketch_dim'], device=device, dtype=torch.float64)
    assigned = m['shards'][args.task::m['tasks']]
    started = time.monotonic()
    rows_done = 0
    # Individual contiguous tensors are required for flatten/index_add updates.
    parts = [torch.zeros((m['sketch_dim'], model['width']), device=device) for model in m['models']]
    for si, shard in enumerate(assigned):
        caches = [Slots(model, shard) for model in m['models']]
        for start in range(0, shard['rows'], args.batch_rows):
            end = min(start + args.batch_rows, shard['rows'])
            b, s = row_hash(np.arange(shard['offset']+start, shard['offset']+end, dtype=np.uint64), m['sketch_dim'])
            buckets = torch.as_tensor(b, device=device)
            signs = torch.as_tensor(s, device=device, dtype=torch.float32)
            ones.index_add_(0, buckets, signs.double())
            for j, cache in enumerate(caches):
                ids, values = cache.batch(start, end, device)
                sl = slice(offsets[j], offsets[j+1])
                accumulate_sketch(parts[j], sums[sl], squares[sl], support[sl], ids, values, buckets, signs)
        rows_done += shard['rows']
        if si % 5 == 0 or si == len(assigned)-1:
            print(f'Sketch {args.task}: {si+1}/{len(assigned)} shards; {rows_done:,} rows; {time.monotonic()-started:.1f}s', flush=True)
    sketches = torch.cat(parts, dim=1)
    save(task_path(args, 'sketch', args.task), sketch=sketches.cpu().numpy(), sums=sums.cpu().numpy(),
         squares=squares.cpu().numpy(), support=support.cpu().numpy(), ones=ones.cpu().numpy(), rows=rows_done, manifest_id=m['id'])


def checked(path, m):
    data = np.load(_paper_location(path))
    assert str(data['manifest_id']) == m['id'], str(path)
    return data


def candidates(args, m):
    combined = {}
    rows = 0
    for t in range(m['tasks']):
        with checked(task_path(args, 'sketch', t), m) as d:
            rows += int(d['rows'])
            for key in ('sketch', 'sums', 'squares', 'support', 'ones'):
                value = d[key].astype(np.float64 if key != 'support' else np.int64)
                if key not in combined: combined[key] = value
                else: combined[key] += value
    assert rows == m['rows']
    save(args.output/'moments.npz', **{k:v for k,v in combined.items() if k != 'sketch'}, rows=rows, manifest_id=m['id'])
    normalized, live = normalized_sketch(*(torch.as_tensor(combined[k], device=args.device) for k in ('sketch','sums','squares','ones')), rows)
    offsets = np.cumsum([0] + [x['width'] for x in m['models']])
    output = {}
    for i, j in itertools.permutations(range(len(m['models'])), 2):
        a, b = slice(offsets[i], offsets[i+1]), slice(offsets[j], offsets[j+1])
        output[f'{i}_{j}'] = candidates_for(normalized[:,a], normalized[:,b], live[a], live[b], m['candidate_k']).cpu().numpy().astype(np.int32)
        print(f'Candidates {WIDTHS[i]} -> {WIDTHS[j]}', flush=True)
    save(args.output/'candidates.npz', **output, manifest_id=m['id'])


def rescore(args, m):
    with checked(args.output/'candidates.npz', m) as data:
        candidate = {key:torch.as_tensor(data[key].astype(np.int64), device=args.device) for key in data.files if key != 'manifest_id'}
    dots = {key:torch.zeros(value.shape, device=args.device, dtype=torch.float64) for key,value in candidate.items()}
    assigned = m['shards'][args.task::m['tasks']]
    rows_done = 0
    started = time.monotonic()
    for si, shard in enumerate(assigned):
        caches = [Slots(model, shard) for model in m['models']]
        for start in range(0, shard['rows'], args.batch_rows):
            end = min(start+args.batch_rows, shard['rows'])
            batches = [cache.batch(start,end,args.device) for cache in caches]
            for j, model in enumerate(m['models']):
                dense = torch.zeros((end-start, model['width']), device=args.device)
                # scatter_add preserves nonzero entries in the presence of zero padding duplicates.
                dense.scatter_add_(1, batches[j][0], batches[j][1])
                for i in range(len(m['models'])):
                    if i != j:
                        key = f'{i}_{j}'
                        accumulate_edges(dots[key], candidate[key], *batches[i], dense)
                del dense
        rows_done += shard['rows']
        if si % 5 == 0 or si == len(assigned)-1:
            print(f'Rescore {args.task}: {si+1}/{len(assigned)} shards; {rows_done:,} rows; {time.monotonic()-started:.1f}s', flush=True)
    save(task_path(args, 'rescore', args.task), **{key:value.cpu().numpy() for key,value in dots.items()}, rows=rows_done, manifest_id=m['id'])


def reduce(args, m):
    offsets = np.cumsum([0] + [x['width'] for x in m['models']])
    summary = []
    with checked(args.output/'moments.npz', m) as moments, checked(args.output/'candidates.npz', m) as c:
        partials = [checked(task_path(args,'rescore', t),m) for t in range(m['tasks'])]
        assert sum(int(p['rows']) for p in partials) == m['rows']
        for i,j in itertools.permutations(range(len(m['models'])),2):
            key=f'{i}_{j}'
            dots = sum((p[key] for p in partials), np.zeros(c[key].shape, dtype=np.float64))
            a,b = slice(offsets[i],offsets[i+1]),slice(offsets[j],offsets[j+1])
            score = pearson_edges(dots,c[key],moments['sums'][a],moments['squares'][a],moments['sums'][b],moments['squares'][b],m['rows'])
            ids, values = best_five(c[key],score,m['save_k'])
            path=args.output/f'gemini_{WIDTHS[i]}_to_{WIDTHS[j]}_top5.npz'
            save(path, target_feature_ids=ids, pearson=values, source_width=WIDTHS[i],target_width=WIDTHS[j],
                 source_feature_ids=np.arange(WIDTHS[i],dtype=np.int32),rows=m['rows'],candidate_limited=True,manifest_id=m['id'])
            valid = np.isfinite(values[:,0])
            summary.append(dict(source_width=WIDTHS[i], target_width=WIDTHS[j], valid_sources=int(valid.sum()),
                                total_sources=WIDTHS[i], mean_best_valid=float(values[valid,0].mean()) if valid.any() else None,
                                mean_best_all_zero_filled=float(np.nan_to_num(values[:,0]).mean()), output=str(path)))
            print(f'Reduced {WIDTHS[i]} -> {WIDTHS[j]}', flush=True)
        for p in partials:p.close()
    atomic_json(args.output/'summary.json',dict(complete=True,manifest_id=m['id'],candidate_limited=True,comparisons=summary))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','sketch','candidates','rescore','reduce'])
    parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--tasks',type=int,default=16)
    parser.add_argument('--task',type=int,default=0)
    parser.add_argument('--batch-rows',type=int,default=2048)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    if args.stage=='prepare':prepare(args)
    else:
        m=read_json(args.output/'manifest.json')
        assert m['source_code_sha256']==hashlib.sha256(_paper_path(__file__).read_bytes()).hexdigest(), 'Code changed after prepare'
        assert 0 <= args.task < m['tasks']
        globals()[args.stage](args,m)
if __name__=='__main__':main()
