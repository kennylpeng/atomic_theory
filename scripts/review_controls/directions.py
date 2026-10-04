"""Fixed-sparsity and independent-seed decoder controls; signed cosine >= 0.7."""
from project_paths import resource_path as _paper_path, resource_location as _paper_location
import argparse, csv, hashlib, json, os, sys
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz
from scipy.sparse.csgraph import maximum_bipartite_matching
ROOT=_paper_path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.stability.persistent_matching import persistent_matching, validate_witness
from scripts.control_models import FIXED, ALTERNATE, checkpoint_path as trainpath
from scripts.stability.validate_split_cardinality_outputs import certify_maximum
OUT=ROOT/'full_experiments/results/sparsity_seed_controls'
BASELINE=ROOT/'full_experiments/results/persistent_stability_signed_0.7'
MODELS=_paper_path('/resources/models_dir')
SCHEDULE={512:32,1024:32,2048:32,4096:32,8192:64,16384:64,32768:64,65536:128,131072:128}
def dump(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,default=str)+'\n')
def sha(path):
 h=hashlib.sha256()
 with open(_paper_location(path),'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def mainpath(f,w):return MODELS/f'{f}_m{w}_k{SCHEDULE[w]}'
class Experiment:
 def __init__(self,family):
  self.family=family;self.folder=OUT/family;self.folder.mkdir(parents=True,exist_ok=True);self.vectors={};self.metadata={};self.rows=[]
  self.original=json.loads((BASELINE/family/'manifest.json').read_text())
  for p in self.original['source_files']:
   stat=_paper_path(p['path']).stat();assert stat.st_size==p['size'] and stat.st_mtime_ns==p['mtime_ns'],p
 def load(self,path):
  import torch
  import torch.nn.functional as F
  path=str(path)
  if path not in self.vectors:
   checkpoint=torch.load(_paper_location(path),map_location='cpu',weights_only=False,mmap=True)
   weights=checkpoint['model_state_dict']['W_dec'].detach().float()
   assert torch.isfinite(weights).all() and torch.all(weights.norm(dim=1)>0)
   self.vectors[path]=F.normalize(weights,dim=1).clone()
   stat=_paper_path(path).stat()
   self.metadata[path]=dict(path=path,sha256=sha(path),size_bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,config=checkpoint['config'],shape=list(weights.shape))
   dump(self.folder/'checkpoints.json',self.metadata)
  return self.vectors[path]
 def graph(self,source,target,tag,reuse=None):
  import torch
  folder=self.folder/'graphs';folder.mkdir(exist_ok=True);path=folder/f'{tag}.npz'
  if reuse is not None:
   graph=load_npz(_paper_location(reuse)).tocsr();dump(folder/f'{tag}.json',dict(reused=str(reuse),sha256=sha(reuse),source=str(source),target=str(target)))
   save_npz(_paper_location(path),graph);return graph
  a=self.load(source);b=self.load(target)
  expected=dict(source_sha256=self.metadata[str(source)]['sha256'],target_sha256=self.metadata[str(target)]['sha256'],threshold=0.7,similarity='signed float32 cosine; TF32 disabled')
  meta=folder/f'{tag}.json'
  if path.exists() and meta.exists() and json.loads(meta.read_text())==expected:return load_npz(_paper_location(path)).tocsr()
  print('Computing',tag,tuple(a.shape),tuple(b.shape),flush=True)
  target_gpu=b.cuda();rr=[];cc=[]
  for begin in range(0,len(a),256):
   scores=a[begin:begin+256].cuda()@target_gpu.T
   r,c=torch.where(scores>=float(np.float32(.7)))
   rr.append(r.cpu().numpy()+begin);cc.append(c.cpu().numpy())
  rows=np.concatenate(rr);cols=np.concatenate(cc)
  graph=csr_matrix((np.ones(len(rows),dtype=bool),(rows,cols)),shape=(len(a),len(b)))
  save_npz(_paper_location(path),graph);dump(meta,expected)
  del target_gpu,scores;torch.cuda.empty_cache();return graph
 def persistence(self,tag,width,targets,graphs):
  selected,dest,stats=persistent_matching(graphs);validate_witness(graphs,selected,dest)
  np.savez_compressed(_paper_location(self.folder/f'{tag}_witness_{width}.npz'),source=selected,**{f'target_{w}':d for w,d in zip(targets,dest)})
  row=dict(family=self.family,experiment=tag,width=width,target_widths=targets,count=len(selected),proportion=len(selected)/width,**stats)
  self.rows.append(row);dump(self.folder/'results.json',self.rows);print(row,flush=True)
 def fixed(self):
  widths=[512,4096,32768,65536,131072];graphs={}
  for treatment in ('schedule','fixed128'):
   paths={w:(trainpath(self.family,w,128,FIXED[self.family][w]) if treatment=='fixed128' and w in FIXED[self.family] else mainpath(self.family,w)) for w in widths}
   for i,w in enumerate(widths[:-1]):
    targets=widths[i+1:];gg=[]
    for v in targets:
     original=treatment=='schedule' or w>=65536
     gg.append(self.graph(paths[w],paths[v],f'{treatment}_{w}_{v}',BASELINE/self.family/f'graph_{w}_to_{v}.npz' if original else None))
    self.persistence(treatment+'_five_widths',w,targets,gg)
    if len(targets)>1:self.persistence(treatment+'_through_65536',w,targets[:-1],gg[:-1])
   for w,p in paths.items():self.load(p)
 def seeds(self):
  alternatives={w:trainpath('gemini',w,SCHEDULE[w],seed) for w,seed in ALTERNATE.items()}
  for w,p in alternatives.items():
   baseline=mainpath('gemini',w);a=self.load(baseline);b=self.load(p)
   c1=self.metadata[str(baseline)]['config'];c2=self.metadata[str(p)]['config']
   assert c1['seed']!=c2['seed'], 'Independent seeds required'
   for key in ('act_size','dict_size','top_k','num_tokens','aux_penalty','top_k_aux','batch_size','lr','beta1','beta2','input_unit_norm','n_batches_to_dead'):
    if key in c1 or key in c2:assert c1.get(key)==c2.get(key),(key,c1.get(key),c2.get(key))
   graph=self.graph(baseline,p,f'seed_same_width_{w}')
   dst=maximum_bipartite_matching(graph,perm_type='column');src=np.flatnonzero(dst>=0);dst=dst[src]
   certify_maximum(graph,src,dst)
   np.savez_compressed(_paper_location(self.folder/f'seed_same_width_{w}_assignment.npz'),source=src,destination=dst)
   self.rows.append(dict(family='gemini',experiment='seed_same_width',width=w,count=len(src),proportion=len(src)/w,edges=graph.nnz,certificate='matching_equal_vertex_cover'))
   targets=[v for v in SCHEDULE if v>w]
   if targets:
    gg=[self.graph(p,mainpath('gemini',v),f'seed_source_{w}_{v}') for v in targets]
    self.persistence('alternate_seed_source_full_schedule',w,targets,gg)
    old=json.loads((BASELINE/'gemini'/'summary.json').read_text());row=next(r for r in old if r['width']==w)
    self.rows.append(dict(family='gemini',experiment='original_seed_source_full_schedule',width=w,target_widths=targets,count=row['persistent_count'],proportion=row['proportion']))
  widths=[512,4096,32768]
  if True:
   for tag,paths in [('original_seed_three_widths',{w:mainpath('gemini',w) for w in widths}),('alternate_seed_three_widths',alternatives)]:
    for i,w in enumerate(widths[:-1]):
     targets=widths[i+1:];gg=[self.graph(paths[w],paths[v],f'{tag}_{w}_{v}',BASELINE/'gemini'/f'graph_{w}_to_{v}.npz' if tag.startswith('original') else None) for v in targets]
     self.persistence(tag,w,targets,gg)
  dump(self.folder/'results.json',self.rows)
if __name__=='__main__':
 import torch
 p=argparse.ArgumentParser();p.add_argument('--family',choices=['gemini','nemotron'],required=True);p.add_argument('--seeds-only',action='store_true');args=p.parse_args()
 torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')));torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert torch.cuda.is_available();e=Experiment(args.family)
 if args.seeds_only and (e.folder/'results.json').exists():e.rows=json.loads((e.folder/'results.json').read_text());e.rows=[r for r in e.rows if 'seed' not in r['experiment']]
 if not args.seeds_only:e.fixed()
 if args.family=='gemini':e.seeds()
 dump(e.folder/'COMPLETE.json',dict(family=args.family,slurm_job=os.environ.get('SLURM_JOB_ID'),torch_version=torch.__version__,gpu=torch.cuda.get_device_name(),alternate_seed_widths=list(ALTERNATE) if args.family=='gemini' else [],script_sha256=sha(_paper_path(__file__))))
