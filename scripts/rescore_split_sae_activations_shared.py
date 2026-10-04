"""Rescore split-specific candidate lists while reading each cached row only once.

This is an I/O optimization of match_split_sae_activation_sketch.rescore. Its
partitions follow the full-corpus shard assignment; each partial records the
exact contributing shards and row count for every evaluation distribution.
"""
from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from scipy.sparse import save_npz
from scripts import match_split_sae_activation_sketch as base
from scripts import match_gemini_width_sketch as core


def load(root, family):
    manifests=[core.read_json(root/family/split/'manifest.json') for split in base.SPLITS]
    assert all(m['models']==manifests[0]['models'] and m['tasks']==manifests[0]['tasks'] for m in manifests)
    return manifests


def assigned(manifests, task):
    shards=manifests[0]['shards'][task::manifests[0]['tasks']]
    memberships=[set(s['relative_shard'] for s in m['shards']) for m in manifests]
    return shards, memberships


def partial_path(root, family, evaluation, task):
    return root/family/evaluation/'work'/f'shared_rescore_{task:03d}.npz'


def sketch_path(root,family,evaluation,task):
    return root/family/evaluation/'work'/f'shared_sketch_{task:03d}.npz'


def sketch(args,manifests):
    shards,memberships=assigned(manifests,args.task)
    models=manifests[0]['models'];dim=manifests[0]['sketch_dim']
    width=sum(x['width'] for x in models);offsets=np.cumsum([0]+[x['width'] for x in models])
    totals=[dict(sketch=torch.zeros((dim,width),dtype=torch.float64,device=args.device),
                 sums=torch.zeros(width,dtype=torch.float64,device=args.device),
                 squares=torch.zeros(width,dtype=torch.float64,device=args.device),
                 support=torch.zeros(width,dtype=torch.int64,device=args.device),
                 ones=torch.zeros(dim,dtype=torch.float64,device=args.device)) for _ in manifests]
    rows=[0]*len(manifests);contributed=[[] for _ in manifests];started=time.monotonic();unique_rows=0
    for si,shard in enumerate(shards):
        parts=[torch.zeros((dim,model['width']),device=args.device) for model in models]
        sums=torch.zeros(width,dtype=torch.float64,device=args.device);squares=torch.zeros_like(sums)
        support=torch.zeros(width,dtype=torch.int64,device=args.device)
        ones=torch.zeros(dim,dtype=torch.float64,device=args.device)
        caches=[core.Slots(model,shard) for model in models]
        for start in range(0,shard['rows'],args.batch_rows):
            end=min(start+args.batch_rows,shard['rows'])
            b,s=core.row_hash(np.arange(shard['offset']+start,shard['offset']+end,dtype=np.uint64),dim)
            buckets=torch.as_tensor(b,device=args.device);signs=torch.as_tensor(s,device=args.device)
            ones.index_add_(0,buckets,signs.double())
            for j,cache in enumerate(caches):
                ids,values=cache.batch(start,end,args.device);sl=slice(offsets[j],offsets[j+1])
                core.accumulate_sketch(parts[j],sums[sl],squares[sl],support[sl],ids,values,buckets,signs)
        result=dict(sketch=torch.cat(parts,dim=1),sums=sums,squares=squares,support=support,ones=ones)
        for e,members in enumerate(memberships):
            if shard['relative_shard'] in members:
                rows[e]+=shard['rows'];contributed[e].append(shard['relative_shard'])
                for key in totals[e]:totals[e][key]+=result[key]
        unique_rows+=shard['rows']
        if si%5==0 or si==len(shards)-1:
            print(f'Shared sketch {args.family}/{args.task}: {si+1}/{len(shards)} shards, {unique_rows:,} unique rows, {sum(rows):,} distribution rows; {time.monotonic()-started:.1f}s',flush=True)
    for e,m in enumerate(manifests):
        core.save(sketch_path(args.root,args.family,m['evaluation_split'],args.task),
                  **{k:v.cpu().numpy() for k,v in totals[e].items()},rows=rows[e],shards=np.array(contributed[e]),
                  manifest_id=m['id'],implementation_sha256=base.digest(__file__),batch_rows=args.batch_rows)


def candidates(args,manifests):
    for e,m in enumerate(manifests):
        combined={};rows=0;covered=[]
        for task in range(m['tasks']):
            shards,memberships=assigned(manifests,task)
            expected=[s for s in shards if s['relative_shard'] in memberships[e]]
            with core.checked(sketch_path(args.root,args.family,m['evaluation_split'],task),m) as p:
                assert str(p['implementation_sha256'])==base.digest(__file__)
                assert int(p['rows'])==sum(s['rows'] for s in expected)
                assert list(p['shards'])==[s['relative_shard'] for s in expected]
                rows+=int(p['rows']);covered.extend(p['shards'])
                for key in ('sketch','sums','squares','support','ones'):
                    if key in combined:combined[key]+=p[key]
                    else:combined[key]=p[key]
        assert rows==m['rows'] and len(covered)==len(set(covered))==len(m['shards'])
        assert set(covered)=={s['relative_shard'] for s in m['shards']}
        norm,live=core.normalized_sketch(*(torch.as_tensor(combined[k],device=args.device) for k in ('sketch','sums','squares','ones')),rows)
        out=args.root/args.family/m['evaluation_split']
        provenance=dict(rows=rows,manifest_id=m['id'],sketch_implementation_sha256=base.digest(__file__),
                        accumulation='float32 per shard; float64 partition and distribution reduction')
        core.save(out/'sketch.npz',**combined,normalized=norm.cpu().numpy(),live=live.cpu().numpy(),**provenance)
        core.save(out/'moments.npz',**{k:v for k,v in combined.items() if k!='sketch'},**provenance)
        offsets=np.cumsum([0]+[model['width'] for model in m['models']]);result={}
        for i,j in base.directions(m):
            a,b=slice(offsets[i],offsets[i+1]),slice(offsets[j],offsets[j+1])
            result[f'{i}_{j}']=core.candidates_for(norm[:,a],norm[:,b],live[a],live[b],m['candidate_k']).cpu().numpy().astype(np.int32)
        core.save(out/'candidates.npz',**result,manifest_id=m['id'],sketch_implementation_sha256=base.digest(__file__))
        print('Generated shared-read candidates:',args.family,m['evaluation_split'],flush=True)


def rescore(args, manifests):
    shards, memberships=assigned(manifests,args.task)
    fingerprint=base.digest(__file__)
    if all(partial_path(args.root,args.family,m['evaluation_split'],args.task).exists() for m in manifests):
        for m, members in zip(manifests,memberships):
            with core.checked(partial_path(args.root,args.family,m['evaluation_split'],args.task),m) as p:
                assert str(p['implementation_sha256'])==fingerprint
                assert int(p['rows'])==sum(s['rows'] for s in shards if s['relative_shard'] in members)
        print('Already complete',args.family,args.task);return
    candidates=[];dots=[]
    for m in manifests:
        with core.checked(args.root/args.family/m['evaluation_split']/'candidates.npz',m) as c:
            cand={f'{i}_{j}':torch.as_tensor(c[f'{i}_{j}'].astype(np.int64),device=args.device) for i,j in base.directions(m)}
        candidates.append(cand)
        dots.append({k:torch.zeros(v.shape,device=args.device,dtype=torch.float64) for k,v in cand.items()})
    models=manifests[0]['models']; rows=[0]*len(manifests); contributed=[[] for _ in manifests]
    started=time.monotonic();total_rows=0
    for si,shard in enumerate(shards):
        evaluations=[e for e,members in enumerate(memberships) if shard['relative_shard'] in members]
        caches=[core.Slots(model,shard) for model in models]
        for start in range(0,shard['rows'],args.batch_rows):
            end=min(start+args.batch_rows,shard['rows'])
            batches=[cache.batch(start,end,args.device) for cache in caches]
            for j,model in enumerate(models):
                dense=torch.zeros((end-start,model['width']),device=args.device)
                dense.scatter_add_(1,*batches[j])
                for e in evaluations:
                    for i,jj in base.directions(manifests[e]):
                        if jj==j:
                            key=f'{i}_{j}'
                            core.accumulate_edges(dots[e][key],candidates[e][key],*batches[i],dense)
                del dense
        for e in evaluations:
            rows[e]+=shard['rows'];contributed[e].append(shard['relative_shard'])
        total_rows+=shard['rows']
        if si%5==0 or si==len(shards)-1:
            print(f'Shared rescore {args.family}/{args.task}: {si+1}/{len(shards)} shards, {total_rows:,} unique rows, {sum(rows):,} distribution rows; {time.monotonic()-started:.1f}s',flush=True)
    for e,m in enumerate(manifests):
        core.save(partial_path(args.root,args.family,m['evaluation_split'],args.task),
                  **{k:v.cpu().numpy() for k,v in dots[e].items()},rows=rows[e],shards=np.array(contributed[e]),
                  manifest_id=m['id'],implementation_sha256=fingerprint,batch_rows=args.batch_rows)


def reduce_one(root, family, manifests, e):
    m=manifests[e];out=root/family/m['evaluation_split'];pairs=base.directions(m)
    offsets=np.cumsum([0]+[model['width'] for model in m['models']])
    totals={};covered=[];rows=0
    for task in range(m['tasks']):
        shards,memberships=assigned(manifests,task)
        expected=[s for s in shards if s['relative_shard'] in memberships[e]]
        with core.checked(partial_path(root,family,m['evaluation_split'],task),m) as p:
            assert str(p['implementation_sha256'])==base.digest(__file__)
            assert list(p['shards'])==[s['relative_shard'] for s in expected]
            assert int(p['rows'])==sum(s['rows'] for s in expected)
            rows+=int(p['rows']);covered.extend(p['shards'])
            for i,j in pairs:
                key=f'{i}_{j}'
                if key in totals:totals[key]+=p[key]
                else:totals[key]=p[key]
    assert rows==m['rows']
    assert len(covered)==len(set(covered))==len(m['shards'])
    assert set(covered)=={s['relative_shard'] for s in m['shards']}
    top={};all_candidates={}
    with core.checked(out/'moments.npz',m) as moments,core.checked(out/'candidates.npz',m) as c:
        assert int(moments['rows'])==rows
        for i,j in pairs:
            key=f'{i}_{j}';a,b=slice(offsets[i],offsets[i+1]),slice(offsets[j],offsets[j+1])
            score=core.pearson_edges(totals[key],c[key],moments['sums'][a],moments['squares'][a],
                                     moments['sums'][b],moments['squares'][b],rows)
            ids,values=core.best_five(c[key],score,m['save_k'])
            top[key]=ids,values;all_candidates[key]=c[key],score.astype(np.float32)
            provenance=dict(evaluation_split=m['evaluation_split'],rows=rows,manifest_id=m['id'],
                            rescore_implementation_sha256=base.digest(__file__))
            core.save(out/f'{m["models"][i]["split"]}_to_{m["models"][j]["split"]}_top5.npz',
                      target_feature_ids=ids,pearson=values,source_feature_ids=np.arange(len(ids),dtype=np.int32),**provenance)
            core.save(out/f'{key}_candidates_rescored.npz',target_feature_ids=c[key],pearson=score,**provenance)
    results=[];source_index=m['evaluation_index']
    for j,model in enumerate(m['models']):
        if j==source_index:continue
        forward,reverse=f'{source_index}_{j}',f'{j}_{source_index}'
        graph,source,target,diag=base.match_graph(top[forward],top[reverse],m['threshold'])
        all_graph,all_source,_,_=base.match_graph(all_candidates[forward],all_candidates[reverse],m['threshold'])
        assert len(all_source)>=len(source)
        split=model['split'];save_npz(_paper_location(out/f'graph_to_{split}.npz'),graph)
        core.save(out/f'matching_to_{split}.npz',source=source,destination=target,
                  manifest_id=m['id'],evaluation_split=m['evaluation_split'])
        width=m['models'][source_index]['width']
        row=dict(family=family,source_split=m['evaluation_split'],comparison_split=split,
                 evaluation_split=m['evaluation_split'],rows=rows,threshold=m['threshold'],matches=len(source),
                 source_features=width,proportion=len(source)/width,all32_matches=len(all_source),all32_edges=all_graph.nnz,
                 certified_on_retained_graph=True,candidate_limited=True,**diag)
        results.append(row);print(json.dumps(row),flush=True)
    core.atomic_json(out/'summary.json',dict(complete=True,manifest_id=m['id'],comparisons=results,
        rescore_implementation_sha256=base.digest(__file__),coverage_audited=True,
        rescore_partitioning='full-corpus shard index modulo task count; exact membership and coverage verified'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['sketch','candidates','rescore','reduce'])
    p.add_argument('--root',type=Path,default=base.OUTPUT)
    p.add_argument('--family',choices=base.FAMILIES,default='gemini')
    p.add_argument('--task',type=int,default=0)
    p.add_argument('--batch-rows',type=int,default=8192)
    p.add_argument('--device',default='cuda')
    args=p.parse_args();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    manifests=load(args.root,args.family)
    if args.stage in ('sketch','candidates','rescore'):globals()[args.stage](args,manifests)
    else:
        for e in range(len(manifests)):reduce_one(args.root,args.family,manifests,e)

if __name__=='__main__':main()
