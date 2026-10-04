"""Nemotron width matching and reusable Gemini/Nemotron common-corpus sketches.

The shared row hash uses ORIGINAL Gemini row offsets, including gaps. Gemini's
common sketch is obtained by subtracting its six nonshared shards. Never compare
Nemotron with the uncorrected full-Gemini sketch or renumber the common rows.
"""
from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from scripts import match_gemini_width_sketch as core

GEMINI = _paper_path('full_experiments/results/gemini_cross_width_sketch512_candidates32_top5')
DEFAULT_OUT = _paper_path('full_experiments/results/nemotron_cross_width_sketch512_candidates32_top5')
NEM_CACHE = core.CACHE/'nemotron_all_experiment_saes_post_topk'
KEYS = ('sketch','sums','squares','support','ones')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def alignment(gemini_shards, nemotron_shards):
    targets={s['relative_shard']:int(s['rows']) for s in nemotron_shards}
    assert len(targets)==len(nemotron_shards)
    assert len({s['relative_shard'] for s in gemini_shards})==len(gemini_shards)
    assert targets.keys() <= {s['relative_shard'] for s in gemini_shards}
    common=[];excluded=[]
    for shard in gemini_shards:
        if shard['relative_shard'] in targets:
            assert shard['rows']==targets[shard['relative_shard']]
            common.append(dict(shard))
        else:excluded.append(dict(shard))
    return common,excluded


def write_manifest(path, value):
    value['id']=digest(value)
    if path.exists():assert core.read_json(path)==value, 'Manifest differs; use a fresh output directory'
    core.atomic_json(path,value)
    return value


def prepare(args):
    g=core.read_json(GEMINI/'manifest.json');n=core.read_json(NEM_CACHE/'plan.json')
    assert n['complete'] and core.read_json(GEMINI/'summary.json')['complete']
    assert g['source_code_sha256']==hashlib.sha256(_paper_path(core.__file__).read_bytes()).hexdigest()
    common,excluded=alignment(g['shards'],n['source_shards'])
    assert len(common)==923 and sum(s['rows'] for s in common)==89227558
    models=[]
    for width,k in zip(core.WIDTHS,core.TOPKS):
        name=f'nemotron_m{width}_k{k}';model=next(x for x in n['models'] if x['name']==name)
        models.append(dict(width=width,top_k=k,name=name,cache=str(NEM_CACHE),sha256=model['sha256']))
    space=dict(shards=common,hash='scripts.aligned_activations.row_hash / SplitMix64',
               sketch_dim=512,original_gemini_manifest_id=g['id'])
    m=dict(family='nemotron',models=models,shards=common,rows=sum(s['rows'] for s in common),tasks=args.tasks,
           sketch_dim=512,candidate_k=32,save_k=5,metric='signed activation Pearson',candidate_limited=True,
           sample_space_id=digest(space),sample_space=space,original_gemini_directory=str(GEMINI),original_gemini_manifest_id=g['id'],
           gemini_excluded_shards=excluded,sketch_dtype='float32',moments_and_candidate_products_dtype='float64',
           source_code_sha256=hashlib.sha256(_paper_path(__file__).read_bytes()).hexdigest(),
           core_code_sha256=hashlib.sha256(_paper_path(core.__file__).read_bytes()).hexdigest())
    m=write_manifest(args.output/'manifest.json',m)
    excluded_m=dict(models=g['models'],shards=excluded,rows=sum(s['rows'] for s in excluded),tasks=1,
                    sketch_dim=512,parent_manifest_id=m['id'])
    write_manifest(args.output/'gemini_excluded/manifest.json',excluded_m)
    print(f'Prepared {m["rows"]:,} common rows; {len(excluded)} Gemini-only shards; sample space {m["sample_space_id"]}',flush=True)


def audit(args,m):
    import pyarrow.parquet as pq
    gplan=core.read_json(_paper_path(core.read_json(GEMINI/'manifest.json')['models'][0]['cache'])/'plan.json')
    nplan=core.read_json(NEM_CACHE/'plan.json')
    grow={s['relative_shard']:s for s in gplan['source_shards']}
    nrow={s['relative_shard']:s for s in nplan['source_shards']}
    hashes=[]
    for i,shard in enumerate(m['shards']):
        relative=shard['relative_shard'];arrays=[];columns=[]
        for plan,expected in [(gplan,grow[relative]),(nplan,nrow[relative])]:
            folder=_paper_path(plan['embeddings_dir'])/relative
            path=folder/'metadata.parquet';stat=path.stat()
            assert stat.st_size==expected['metadata_size_bytes'] and stat.st_mtime_ns==expected['metadata_mtime_ns']
            values=pq.read_table(path,columns=['row_idx'])['row_idx'].to_numpy()
            assert len(values)==shard['rows'];arrays.append(values)
            columns.append(core.read_json(folder/'shard_meta.json')['text_column'])
        assert columns[0]==columns[1],relative
        assert np.array_equal(*arrays),relative
        hashes.append(hashlib.sha256(np.asarray(arrays[0],dtype='<i8').tobytes()).hexdigest())
        if i%100==0:print(f'Alignment audit {i+1}/{len(m["shards"])}',flush=True)
    core.atomic_json(args.output/'alignment_audit.json',dict(complete=True,manifest_id=m['id'],rows=m['rows'],shards=len(hashes),
                     source_row_hashes=hashes,checks='same dataset/config/split/shard, row_idx order and text_column; original metadata sizes/timestamps verified'))


def audited(args,m):
    audit=core.read_json(args.output/'alignment_audit.json')
    assert audit['complete'] and audit['manifest_id']==m['id']


def aggregate(folder,m):
    combined={};rows=0
    for task in range(m['tasks']):
        with core.checked(folder/'work'/f'sketch_{task:03d}.npz',m) as d:
            expected=sum(s['rows'] for s in m['shards'][task::m['tasks']])
            assert int(d['rows'])==expected
            rows+=int(d['rows'])
            for key in KEYS:
                value=d[key].astype(np.int64 if key=='support' else np.float64)
                if key not in combined:combined[key]=value
                else:combined[key]+=value
    assert rows==m['rows']
    return combined,rows


def subtract_excluded(full,excluded):
    result={key:full[key]-excluded[key] for key in KEYS}
    assert np.all(result['support']>=0)
    for key in ('sums','squares'):
        tol=64*np.finfo(np.float64).eps*np.maximum(np.abs(full[key]),1)
        assert np.all(result[key]>=-tol)
        result[key]=np.maximum(result[key],0)
    dead=result['support']==0
    result['sums'][dead]=0;result['squares'][dead]=0;result['sketch'][:,dead]=0
    return result


def export(args,m,family,combined,rows,method):
    assert rows==m['rows']
    normalized,live=core.normalized_sketch(*(torch.as_tensor(combined[k],device=args.device) for k in ('sketch','sums','squares','ones')),rows)
    offsets=np.cumsum([0]+[x['width'] for x in m['models']])
    core.save(args.output/f'{family}_common_sketch.npz',**combined,normalized=normalized.cpu().numpy(),live=live.cpu().numpy(),
              rows=rows,widths=np.array([x['width'] for x in m['models']]),feature_offsets=offsets,
              sample_space_id=m['sample_space_id'],manifest_id=m['id'],family=family,construction=method)
    return normalized,live,offsets


def candidates(args,m):
    audited(args,m)
    combined,rows=aggregate(args.output,m)
    core.save(args.output/'moments.npz',**{k:v for k,v in combined.items() if k!='sketch'},rows=rows,manifest_id=m['id'])
    normalized,live,offsets=export(args,m,'nemotron',combined,rows,'direct common-corpus accumulation with original Gemini row hashes')
    output={}
    for i,j in itertools.permutations(range(len(m['models'])),2):
        a,b=slice(offsets[i],offsets[i+1]),slice(offsets[j],offsets[j+1])
        output[f'{i}_{j}']=core.candidates_for(normalized[:,a],normalized[:,b],live[a],live[b],m['candidate_k']).cpu().numpy().astype(np.int32)
        print(f'Candidates {m["models"][i]["width"]} -> {m["models"][j]["width"]}',flush=True)
    core.save(args.output/'candidates.npz',**output,manifest_id=m['id'])


def gemini_excluded(args,m):
    audited(args,m)
    sub=argparse.Namespace(**vars(args));sub.output=args.output/'gemini_excluded';sub.task=0
    core.sketch(sub,core.read_json(sub.output/'manifest.json'))


def gemini_export(args,m):
    audited(args,m)
    g=core.read_json(GEMINI/'manifest.json')
    assert g['id']==m['original_gemini_manifest_id']
    full,fullrows=aggregate(GEMINI,g)
    excluded,excludedrows=aggregate(args.output/'gemini_excluded',core.read_json(args.output/'gemini_excluded/manifest.json'))
    combined=subtract_excluded(full,excluded)
    export(args,m,'gemini',combined,fullrows-excludedrows,'original full-Gemini task sketches minus six Gemini-only shards; float64 subtraction of float32 sketch accumulators')


def compatibility(args,m):
    with np.load(_paper_location(args.output/'gemini_common_sketch.npz')) as g,np.load(_paper_location(args.output/'nemotron_common_sketch.npz')) as n:
        assert str(g['sample_space_id'])==str(n['sample_space_id'])==m['sample_space_id']
        assert int(g['rows'])==int(n['rows'])==m['rows']
        np.testing.assert_array_equal(g['ones'],n['ones'])
        np.testing.assert_array_equal(g['feature_offsets'],n['feature_offsets'])
        for d in (g,n):
            assert np.isfinite(d['normalized']).all()
            assert d['normalized'].shape==(512,sum(x['width'] for x in m['models']))
    core.atomic_json(args.output/'sketch_compatibility.json',dict(complete=True,manifest_id=m['id'],sample_space_id=m['sample_space_id'],
                     rows=m['rows'],hashes_and_ones_agree=True,centered_with_common_corpus_moments=True,
                     note='Compatible approximate Pearson sketches, not full-corpus rescored cross-family matches. Existing full-Gemini results remain on their original corpus.'))


def reduce(args,m):
    offsets=np.cumsum([0]+[x['width'] for x in m['models']]);summary=[]
    with core.checked(args.output/'moments.npz',m) as moments,core.checked(args.output/'candidates.npz',m) as c:
        partials=[core.checked(core.task_path(args,'rescore',t),m) for t in range(m['tasks'])]
        for t,p in enumerate(partials):assert int(p['rows'])==sum(s['rows'] for s in m['shards'][t::m['tasks']])
        assert sum(int(p['rows']) for p in partials)==m['rows']
        for i,j in itertools.permutations(range(len(m['models'])),2):
            key=f'{i}_{j}';dots=sum((p[key] for p in partials),np.zeros(c[key].shape,dtype=np.float64))
            a,b=slice(offsets[i],offsets[i+1]),slice(offsets[j],offsets[j+1])
            scores=core.pearson_edges(dots,c[key],moments['sums'][a],moments['squares'][a],moments['sums'][b],moments['squares'][b],m['rows'])
            ids,values=core.best_five(c[key],scores,m['save_k']);sw,tw=m['models'][i]['width'],m['models'][j]['width']
            path=args.output/f'nemotron_{sw}_to_{tw}_top5.npz'
            core.save(path,target_feature_ids=ids,pearson=values,source_width=sw,target_width=tw,source_feature_ids=np.arange(sw,dtype=np.int32),
                      rows=m['rows'],candidate_limited=True,manifest_id=m['id'])
            valid=np.isfinite(values[:,0])
            summary.append(dict(source_width=sw,target_width=tw,valid_sources=int(valid.sum()),total_sources=sw,
                mean_best_valid=float(values[valid,0].mean()) if valid.any() else None,mean_best_all_zero_filled=float(np.nan_to_num(values[:,0]).mean()),output=str(path)))
        for p in partials:p.close()
    core.atomic_json(args.output/'summary.json',dict(complete=True,manifest_id=m['id'],candidate_limited=True,comparisons=summary))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','audit','sketch','candidates','rescore','reduce','gemini-excluded','gemini-export','compatibility'])
    parser.add_argument('--output',type=Path,default=DEFAULT_OUT);parser.add_argument('--tasks',type=int,default=16)
    parser.add_argument('--task',type=int,default=0);parser.add_argument('--batch-rows',type=int,default=2048);parser.add_argument('--device',default='cuda')
    args=parser.parse_args();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    if args.stage=='prepare':prepare(args);return
    m=core.read_json(args.output/'manifest.json')
    assert m['source_code_sha256']==hashlib.sha256(_paper_path(__file__).read_bytes()).hexdigest()
    assert m['core_code_sha256']==hashlib.sha256(_paper_path(core.__file__).read_bytes()).hexdigest()
    assert 0<=args.task<m['tasks']
    if args.stage in ('sketch','rescore'):
        audited(args,m)
        path=core.task_path(args,args.stage,args.task)
        if path.exists():
            with core.checked(path,m) as d:assert int(d['rows'])==sum(s['rows'] for s in m['shards'][args.task::m['tasks']])
            print('Already complete:',path,flush=True);return
        getattr(core,args.stage)(args,m)
    else:globals()[args.stage.replace('-','_')](args,m)

if __name__=='__main__':main()
