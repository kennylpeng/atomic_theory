"""All Gemini x Nemotron width pairs: sketch candidates, full-row Pearson top five."""
from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from scripts import match_gemini_width_sketch as core

SOURCE=_paper_path('full_experiments/results/nemotron_cross_width_sketch512_candidates32_top5')
OUTPUT=_paper_path('full_experiments/results/gemini_nemotron_all_widths_sketch512_candidates32_top5')


def directed_pairs(models):
    return [(i,j) for i,a in enumerate(models) for j,b in enumerate(models) if a['family']!=b['family']]


def prepare(args):
    n=core.read_json(SOURCE/'manifest.json')
    compatibility=core.read_json(SOURCE/'sketch_compatibility.json')
    assert compatibility['complete'] and compatibility['manifest_id']==n['id']
    g=core.read_json(_paper_path(n['original_gemini_directory'])/'manifest.json')
    assert g['id']==n['original_gemini_manifest_id']
    models=[dict(**x,family=family) for family,manifest in [('gemini',g),('nemotron',n)] for x in manifest['models']]
    exports={}
    for family in ('gemini','nemotron'):
        path=SOURCE/f'{family}_common_sketch.npz';stat=path.stat()
        exports[family]=dict(path=str(path),size=stat.st_size,mtime_ns=stat.st_mtime_ns)
        with np.load(_paper_location(path)) as d:
            assert str(d['family'])==family and str(d['sample_space_id'])==n['sample_space_id']
            assert int(d['rows'])==n['rows'] and str(d['manifest_id'])==n['id']
    m=dict(models=models,shards=n['shards'],rows=n['rows'],tasks=args.tasks,sketch_dim=512,candidate_k=32,save_k=5,
           sample_space_id=n['sample_space_id'],source_manifest_id=n['id'],exports=exports,
           metric='signed continuous post-TopK activation Pearson',candidate_limited=True,
           directed_pairs=directed_pairs(models),width_pair_count=81,
           source_code_sha256=hashlib.sha256(_paper_path(__file__).read_bytes()).hexdigest(),
           core_code_sha256=hashlib.sha256(_paper_path(core.__file__).read_bytes()).hexdigest())
    m['id']=hashlib.sha256(json.dumps(m,sort_keys=True).encode()).hexdigest()
    if (args.output/'manifest.json').exists():assert core.read_json(args.output/'manifest.json')==json.loads(json.dumps(m))
    core.atomic_json(args.output/'manifest.json',m)
    print(f'Prepared {len(m["directed_pairs"])} directions / 81 width pairs over {m["rows"]:,} shared rows',flush=True)


def export_data(spec,m,family):
    path=_paper_path(spec['path']);stat=path.stat()
    assert (stat.st_size,stat.st_mtime_ns)==(spec['size'],spec['mtime_ns'])
    d=np.load(_paper_location(path))
    assert str(d['manifest_id'])==m['source_manifest_id'] and str(d['sample_space_id'])==m['sample_space_id']
    assert int(d['rows'])==m['rows'] and str(d['family'])==family
    return d


def candidates(args,m):
    sketches=[];lives=[];sums=[];squares=[];ones=None
    for family in ('gemini','nemotron'):
        with export_data(m['exports'][family],m,family) as d:
            if ones is None:ones=d['ones']
            else:np.testing.assert_array_equal(ones,d['ones'])
            assert list(d['widths'])==[x['width'] for x in m['models'] if x['family']==family]
            sketches.append(torch.as_tensor(d['normalized'],device=args.device))
            lives.append(torch.as_tensor(d['live'],device=args.device))
            sums.append(d['sums']);squares.append(d['squares'])
    normalized=torch.cat(sketches,dim=1);live=torch.cat(lives)
    offsets=np.cumsum([0]+[x['width'] for x in m['models']])
    assert normalized.shape==(m['sketch_dim'],offsets[-1])
    core.save(args.output/'moments.npz',sums=np.concatenate(sums),squares=np.concatenate(squares),rows=m['rows'],manifest_id=m['id'])
    result={}
    for i,j in m['directed_pairs']:
        a,b=slice(offsets[i],offsets[i+1]),slice(offsets[j],offsets[j+1])
        result[f'{i}_{j}']=core.candidates_for(normalized[:,a],normalized[:,b],live[a],live[b],m['candidate_k']).cpu().numpy().astype(np.int32)
        left,right=m['models'][i],m['models'][j]
        print(f'Candidates {left["name"]} -> {right["name"]}',flush=True)
    core.save(args.output/'candidates.npz',**result,manifest_id=m['id'])


def rescore(args,m):
    with core.checked(args.output/'candidates.npz',m) as data:
        candidate={f'{i}_{j}':torch.as_tensor(data[f'{i}_{j}'].astype(np.int64),device=args.device) for i,j in m['directed_pairs']}
    dots={key:torch.zeros(value.shape,dtype=torch.float64,device=args.device) for key,value in candidate.items()}
    sources={j:[i for i,jj in m['directed_pairs'] if jj==j] for j in range(len(m['models']))}
    assigned=m['shards'][args.task::m['tasks']];rows_done=0;started=time.monotonic()
    for si,shard in enumerate(assigned):
        caches=[core.Slots(model,shard) for model in m['models']]
        for start in range(0,shard['rows'],args.batch_rows):
            end=min(start+args.batch_rows,shard['rows'])
            batches=[cache.batch(start,end,args.device) for cache in caches]
            for j,model in enumerate(m['models']):
                dense=torch.zeros((end-start,model['width']),device=args.device)
                dense.scatter_add_(1,*batches[j])
                for i in sources[j]:
                    key=f'{i}_{j}'
                    core.accumulate_edges(dots[key],candidate[key],*batches[i],dense)
                del dense
        rows_done+=shard['rows']
        if si%5==0 or si==len(assigned)-1:
            print(f'Cross rescore {args.task}: {si+1}/{len(assigned)} shards; {rows_done:,} rows; {time.monotonic()-started:.1f}s',flush=True)
    core.save(core.task_path(args,'rescore',args.task),**{key:value.cpu().numpy() for key,value in dots.items()},rows=rows_done,manifest_id=m['id'])


def reduce(args,m):
    offsets=np.cumsum([0]+[x['width'] for x in m['models']]);summary=[]
    with core.checked(args.output/'moments.npz',m) as moments,core.checked(args.output/'candidates.npz',m) as c:
        partials=[core.checked(core.task_path(args,'rescore',t),m) for t in range(m['tasks'])]
        for t,p in enumerate(partials):assert int(p['rows'])==sum(s['rows'] for s in m['shards'][t::m['tasks']])
        assert sum(int(p['rows']) for p in partials)==m['rows']
        for i,j in m['directed_pairs']:
            key=f'{i}_{j}';dots=sum((p[key] for p in partials),np.zeros(c[key].shape,dtype=np.float64))
            a,b=slice(offsets[i],offsets[i+1]),slice(offsets[j],offsets[j+1])
            scores=core.pearson_edges(dots,c[key],moments['sums'][a],moments['squares'][a],moments['sums'][b],moments['squares'][b],m['rows'])
            ids,values=core.best_five(c[key],scores,m['save_k']);left,right=m['models'][i],m['models'][j]
            path=args.output/f'{left["family"]}_{left["width"]}_to_{right["family"]}_{right["width"]}_top5.npz'
            core.save(path,target_feature_ids=ids,pearson=values,source_feature_ids=np.arange(left['width'],dtype=np.int32),
                      source_family=left['family'],target_family=right['family'],source_width=left['width'],target_width=right['width'],
                      rows=m['rows'],candidate_limited=True,manifest_id=m['id'])
            valid=np.isfinite(values[:,0]);filled=np.nan_to_num(values[:,0])
            summary.append(dict(source_family=left['family'],target_family=right['family'],source_width=left['width'],target_width=right['width'],
                valid_sources=int(valid.sum()),total_sources=left['width'],mean_best_valid=float(values[valid,0].mean()) if valid.any() else None,
                mean_best_all_zero_filled=float(filled.mean()),median_best_all_zero_filled=float(np.median(filled)),
                fraction_best_ge_07=float((filled>=.7).mean()),fraction_best_ge_09=float((filled>=.9).mean()),output=str(path)))
            print('Reduced',path.name,flush=True)
        for p in partials:p.close()
    core.atomic_json(args.output/'summary.json',dict(complete=True,manifest_id=m['id'],candidate_limited=True,comparisons=summary))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','candidates','rescore','reduce'])
    parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--tasks',type=int,default=16)
    parser.add_argument('--task',type=int,default=0);parser.add_argument('--batch-rows',type=int,default=2048);parser.add_argument('--device',default='cuda')
    args=parser.parse_args();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    if args.stage=='prepare':prepare(args);return
    m=core.read_json(args.output/'manifest.json')
    assert m['source_code_sha256']==hashlib.sha256(_paper_path(__file__).read_bytes()).hexdigest()
    assert m['core_code_sha256']==hashlib.sha256(_paper_path(core.__file__).read_bytes()).hexdigest()
    assert 0<=args.task<m['tasks']
    if args.stage=='rescore' and core.task_path(args,'rescore',args.task).exists():
        with core.checked(core.task_path(args,'rescore',args.task),m) as d:
            assert int(d['rows'])==sum(s['rows'] for s in m['shards'][args.task::m['tasks']])
        print('Task already complete',args.task);return
    globals()[args.stage](args,m)

if __name__=='__main__':main()
