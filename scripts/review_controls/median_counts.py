"""Exact Figure-10 pooled-positive-median prevalence on every distribution.

One frozen threshold per SAE, calibrated on its family's full corpus. A sampled
bracket accelerates exact median selection; an exact rank check is mandatory.
All source values and feature IDs are verified against cache payload digests.
"""
from project_paths import resource_path as _paper_path, resource_location as _paper_location
import argparse,hashlib,json,mmap,os,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
ROOT=_paper_path(__file__).resolve().parents[2]
RESULTS=ROOT/'full_experiments/results'
OUT=RESULTS/'sparsity_seed_controls/prevalence_median'
SPLITS=['main','wikipedia','no_wikipedia','random1','random2']
def digest(path):return hashlib.sha256(_paper_path(path).read_bytes()).hexdigest()
def dump(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2)+'\n')
def floor32(value):
 t=np.float32(value)
 return np.nextafter(t,np.float32(-np.inf)) if float(t)>value else t

def compute(task):
 family=['gemini','nemotron'][task//5];split=SPLITS[task%5];folder=OUT/'counts'/f'{family}_{split}';folder.mkdir(parents=True,exist_ok=True)
 source=RESULTS/'split_sae_activation_sketch512_candidates32_top5'/family
 manifests={s:json.loads((source/s/'manifest.json').read_text()) for s in SPLITS};m=manifests['main'];model=next(x for x in m['models'] if x['split']==split);root=_paper_path(model['cache'])
 plan=json.loads((root/'plan.json').read_text());assert plan['complete'] and plan['spec_id']==model['cache_spec_id']
 checkpoint=next(x for x in plan['models'] if x['name']==model['name']);assert checkpoint['sha256']==model['sha256'];stat=_paper_path(checkpoint['path']).stat();assert stat.st_size==checkpoint['size_bytes'] and stat.st_mtime_ns==checkpoint['mtime_ns']
 shards=m['shards'];datasets=sorted({s['relative_shard'].split('/')[0] for s in shards});ds_index={d:i for i,d in enumerate(datasets)};shard_counts=np.zeros(len(datasets),np.int64);positive=np.zeros((len(datasets),16384),np.int64);above=np.zeros_like(positive)
 known=None;calibration={}
 if split=='main':
  source_stats=RESULTS/'main_sae_activation_statistics_zero_median/per_feature'/f'{family}_m16384_k64.npz'
  with np.load(_paper_location(source_stats)) as z:
   known=float(z['thresholds'][1]);reference=z['counts'].copy();prov=json.loads(str(z['provenance']));assert prov['model']['sha256']==model['sha256'];calibration=prov['calibration']
  upper=floor32(known);lower=upper
 else:
  seed=20260925+task;rng=np.random.default_rng(seed);selected=np.sort(rng.choice(m['rows'],10000,replace=False))
  def sample_shard(s):
   a,b=np.searchsorted(selected,[s['offset'],s['offset']+s['rows']])
   if b==a:return np.empty(0,np.float32)
   values=np.load(_paper_location(root/'shards'/s['relative_shard']/model['name']/'data.npy'),mmap_mode='r');values._mmap.madvise(mmap.MADV_RANDOM)
   v=np.asarray(values.reshape(s['rows'],64)[selected[a:b]-s['offset']]).ravel();return v[v>0]
  with ThreadPoolExecutor(max_workers=4) as pool:sample=np.concatenate(list(pool.map(sample_shard,shards)))
  lower,upper=np.quantile(sample,[.47,.53],method='nearest').astype(np.float32)
  calibration=dict(sample_rows=10000,seed=seed,sample_positive_events=len(sample),sample_median=float(np.median(sample.astype(np.float64))),sample_quantiles=[.47,.53],bracket_lower=float(lower),bracket_upper=float(upper));del sample
  dump(folder/'calibration_bracket.json',calibration)
 assert 0<lower<=upper
 print(f'{family}/{split}: bracket {lower:.9g}, {upper:.9g}; known median {known}',flush=True)
 brackets=[];bracket_count=0;below=0;manifest_hash=hashlib.sha256();start=time.monotonic()
 for i,s in enumerate(shards):
  ds=ds_index[s['relative_shard'].split('/')[0]];directory=root/'shards'/s['relative_shard'];path=directory/'complete.json';payload=path.read_bytes();manifest_hash.update(payload);complete=json.loads(payload)
  assert complete['complete'] and complete['spec_id']==plan['spec_id'];assert complete['source']['relative_shard']==s['relative_shard'];o=complete['outputs'][model['name']];assert o['shape']==[s['rows'],16384] and o['top_k']==64
  arrays={}
  for key,dtype in [('data','float32'),('indices','int32')]:
   entry=o['components'][key];p=directory/entry['file'];assert p.stat().st_size==entry['size_bytes'];a=np.load(_paper_location(p));assert str(a.dtype)==dtype and list(a.shape)==entry['shape'];assert hashlib.sha256(memoryview(a).cast('B')).hexdigest()==entry['payload_sha256'],str(p);arrays[key]=a
  v,ids=arrays['data'],arrays['indices'];assert len(v)==len(ids)==s['rows']*64 and np.all(np.isfinite(v)) and v.min()>=0 and ids.min()>=0 and ids.max()<16384
  pos=v>0;assert int(pos.sum())==o['positive_nnz'];positive[ds]+=np.bincount(ids[pos],minlength=16384);above[ds]+=np.bincount(ids[v>upper],minlength=16384)
  if known is None:
   mask=(v>=lower)&(v<=upper);bracket_count+=int(mask.sum());below+=int(((v>0)&(v<lower)).sum())
   if lower!=upper:brackets.append((ds,v[mask].copy(),ids[mask].copy()))
  shard_counts[ds]+=s['rows']
  if (i+1)%25==0 or i+1==len(shards):print(f'{family}/{split}: {i+1}/{len(shards)} shards; {time.monotonic()-start:.1f}s',flush=True)
 assert int(shard_counts.sum())==m['rows'];n=int(positive.sum())
 if known is None:
  ranks=((n-1)//2-below,n//2-below);assert 0<=ranks[0]<=ranks[1]<bracket_count,'Exact median outside sampled bracket; widen bracket and rerun'
  assert below+bracket_count+int(above.sum())==n
  if lower==upper:median=lo=hi=float(lower)
  else:
   values=np.concatenate([v for ds,v,ids in brackets]);assert len(values)==bracket_count;values.partition(ranks);lo,hi=float(values[ranks[0]]),float(values[ranks[1]]);median=(lo+hi)/2;del values
   for ds,v,ids in brackets:above[ds]+=np.bincount(ids[v.astype(np.float64)>median],minlength=16384)
  calibration.update(method='Exact pooled median of all positive full-corpus post-top-k activations',positive_events=n,below_bracket_events=below,bracket_events=bracket_count,central_lower=lo,central_upper=hi,median_positive=median,median_rank_bracket_verified=True)
 else:median=known;assert np.array_equal(positive.sum(0),reference[0]);assert np.array_equal(above.sum(0),reference[1])
 assert np.all(above<=positive) and int(above.sum())<=n//2
 counts=[];zeros=[];rows=[]
 for evaluation in SPLITS:
  em=manifests[evaluation];expected_shards={s['relative_shard'] for s in shards if s['relative_shard'].split('/')[0] in em['datasets']};assert expected_shards=={s['relative_shard'] for s in em['shards']}
  selected=[ds_index[d] for d in em['datasets']];nn=int(shard_counts[selected].sum());assert nn==em['rows'];p=positive[selected].sum(0);c=above[selected].sum(0)
  with np.load(_paper_location(source/evaluation/'moments.npz')) as z:
   assert str(z['manifest_id'])==em['id'];model_index=next(j for j,x in enumerate(em['models']) if x['split']==split);assert np.array_equal(p,z['support'][model_index*16384:(model_index+1)*16384])
  assert np.all(c<=p) and p.max()<=nn;counts.append(c);zeros.append(p);rows.append(nn)
 counts=np.array(counts);rows=np.array(rows)
 provenance=dict(family=family,source_split=split,model=model,checkpoint=checkpoint,threshold_definition='Strictly greater than the exact median of all positive activation events across all features and full-family-corpus examples; same fixed threshold on every evaluation split',calibration=calibration,plan_sha256=digest(root/'plan.json'),split_manifests_sha256={s:digest(source/s/'manifest.json') for s in SPLITS},concatenated_shard_completion_sha256=manifest_hash.hexdigest(),source_payloads_verified=True,zero_counts_match_all_existing_split_moments=True,full_main_counts_match_figure10=split=='main',script_sha256=digest(__file__),slurm_job=os.environ.get('SLURM_JOB_ID'))
 np.savez_compressed(_paper_location(folder/'counts.npz'),counts=counts,positive_counts=np.array(zeros),rows=rows,rates=counts/rows[:,None],evaluation_splits=np.array(SPLITS),threshold=np.array(median),datasets=np.array(datasets),dataset_rows=shard_counts,dataset_counts=above,dataset_positive_counts=positive,provenance=json.dumps(provenance))
 dump(folder/'provenance.json',provenance);dump(folder/'COMPLETE.json',dict(complete=True,threshold=median,positive_events=n,events_above_median=int(above.sum()),counts_sha256=digest(folder/'counts.npz')))
 print(f'COMPLETE {family}/{split}: median={median:.10g}, {above.sum():,}/{n:,} events above',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task',type=int,required=True);a=p.parse_args();assert 0<=a.task<10;compute(a.task)
