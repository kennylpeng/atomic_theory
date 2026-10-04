"""Figure 2(a) analog: exhaustive same-seed independent-intersection SAE persistence.

Decoder signed cosine and continuous post-TopK activation Pearson, both >= .7.
All six final checkpoints share a generating dictionary and held-out sample stream.
"""
from project_paths import resource_path as _paper_path, resource_location as _paper_location
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.sparse import csr_matrix, save_npz
import torch

from common import ROOT, Distribution, Spec, TopK, atomic_json
REPO = ROOT.parents[3]
sys.path.insert(0, str(REPO))
from scripts.stability.persistent_matching import persistent_matching, validate_witness

BUDGET = 1_024_000_000
THRESHOLD = 0.7
OUT = ROOT / 'results/persistence_1024M_t0.7'


def fingerprint(path):
    s = path.stat()
    return dict(path=str(path.relative_to(REPO)), size=s.st_size, mtime_ns=s.st_mtime_ns)


def code_hash():
    paths = [_paper_path(__file__), ROOT/'code/common.py', ROOT.parent.parent/'code/distributions.py',
             ROOT.parent.parent/'code/core.py', REPO/'scripts/stability/persistent_matching.py']
    return hashlib.sha256(b''.join(p.read_bytes() for p in paths)).hexdigest()


def checkpoint(c):
    return ROOT/'models'/c['name']/f'n{BUDGET}'/'model.pt'


def prepare():
    configs = json.loads((ROOT/'configs/sweep.json').read_text())
    groups = {}
    for c in configs:
        key = (c['data']['kind'], c['data']['alpha'], c['top_k'], c['seed'])
        groups.setdefault(key, []).append(c)
    assert len(configs) == 468 and len(groups) == 78
    jobs = []
    for index, (key, cs) in enumerate(sorted(groups.items())):
        cs.sort(key=lambda c:c['width'])
        kind, alpha, k, seed = key
        assert [c['width'] for c in cs] == [128,256,512,1024,2048,cs[0]['data']['M']]
        for c in cs:
            assert all(c[q] == cs[0][q] for q in ['data','eval_seed','test_samples','batch_size','seed','top_k'])
            assert c['budgets'][-1] == BUDGET and c['test_samples'] == 400_000
            assert (ROOT/'results'/c['name']/'complete.json').exists(), c['name']
        jobs.append(dict(index=index, name=f'{kind}_a{alpha:g}_k{k}_s{seed}', kind=kind,
                         alpha=alpha, k=k, seed=seed, configs=cs,
                         checkpoints=[fingerprint(checkpoint(c)) for c in cs]))
    OUT.mkdir(parents=True,exist_ok=True)
    doc=dict(training_examples=BUDGET, threshold=THRESHOLD, evaluation_examples=400_000,
             method='Exhaustive signed cosine and Pearson graphs; exact common-source-subset matching into every larger width.',
             jobs=jobs)
    path=OUT/'manifest.json'
    if path.exists():
        assert json.loads(path.read_text()) == doc, 'Existing analysis manifest differs'
    else: atomic_json(path,doc)
    print(f'{len(jobs)} groups, {len(configs)} final checkpoints: {path}',flush=True)


def moments(z):
    z=z.astype(np.float64,copy=False)
    sums=np.asarray(z.sum(axis=0)).ravel()
    squared=np.asarray(z.multiply(z).sum(axis=0)).ravel()
    centered=squared-sums*sums/z.shape[0]
    tolerance=np.finfo(np.float64).eps*64*np.maximum(squared,1e-300)
    valid=centered>tolerance
    return dict(sum=sums, sumsq=squared, centered_ss=centered, valid=valid)


def pearson_scores(a,b,ma=None,mb=None):
    assert a.shape[0]==b.shape[0]
    a=a.astype(np.float64,copy=False);b=b.astype(np.float64,copy=False)
    ma=moments(a) if ma is None else ma
    mb=moments(b) if mb is None else mb
    score=(a.T@b).toarray()
    score-=np.outer(ma['sum'],mb['sum'])/a.shape[0]
    denom=np.sqrt(np.maximum(ma['centered_ss'],0)[:,None]*np.maximum(mb['centered_ss'],0)[None,:])
    valid=ma['valid'][:,None]&mb['valid'][None,:]
    np.divide(score,denom,out=score,where=valid)
    score[~valid]=np.nan
    return np.clip(score,-1,1)


@torch.inference_mode()
def collect_shared(models,dist,seed,n,batch):
    rng=np.random.default_rng(seed)
    ids=[np.empty((n,m.k),np.int32) for m in models]
    vals=[np.empty((n,m.k),np.float32) for m in models]
    row_hash=hashlib.sha256()
    for start in range(0,n,batch):
        stop=min(start+batch,n)
        x,truth,v=dist.batch(rng,stop-start)
        row_hash.update(truth.tobytes());row_hash.update(v.tobytes())
        for j,m in enumerate(models):
            values,indices,_=m.encode(x)
            ids[j][start:stop]=indices.cpu().numpy()
            vals[j][start:stop]=values.cpu().numpy()
    arrays=[]
    for m,i,v in zip(models,ids,vals):
        z=csr_matrix((v.astype(np.float64).ravel(),
                      (np.repeat(np.arange(n,dtype=np.int32),m.k),i.ravel())),
                     shape=(n,m.decoder.shape[0]))
        z.eliminate_zeros();z.sort_indices();arrays.append(z)
    return arrays,row_hash.hexdigest()


def solve(graphs,widths,metric,directory):
    rows=[]
    for i,width in enumerate(widths[:-1]):
        target_widths=widths[i+1:]
        layers=[graphs[i,j] for j in range(i+1,len(widths))]
        selected,destinations,stats=persistent_matching(layers)
        validate_witness(layers,selected,destinations)
        np.savez_compressed(_paper_location(directory/f'{metric}_witness_{width}.npz'),source=selected,
                            **{f'target_{w}':v for w,v in zip(target_widths,destinations)})
        rows.append(dict(metric=metric,width=width,persistent_count=len(selected),
                         proportion=len(selected)/width,target_widths=target_widths,**stats))
    return rows


def run(index,device):
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','4')))
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    job=json.loads((OUT/'manifest.json').read_text())['jobs'][index]
    directory=OUT/job['name'];directory.mkdir(exist_ok=True)
    version=code_hash()
    complete=directory/'summary.json'
    if complete.exists():
        old=json.loads(complete.read_text())
        assert old['code_hash']==version and old['checkpoints']==job['checkpoints']
        print('ALREADY COMPLETE',job['name'],flush=True);return
    started=time.monotonic()
    cs=job['configs'];models=[];hashes=[]
    dist=Distribution(Spec(**cs[0]['data']),device)
    for c,expected in zip(cs,job['checkpoints']):
        path=checkpoint(c);assert fingerprint(path)==expected
        saved=torch.load(_paper_location(path),map_location='cpu',weights_only=False)
        assert saved['config']==c and saved['examples']==BUDGET
        model=TopK(c['data']['d'],c['width'],c['top_k'],False)
        model.load_state_dict(saved['model']);model.to(device).eval();models.append(model)
        h=hashlib.sha256()
        for key,tensor in sorted(saved['model'].items()):
            h.update(key.encode());h.update(tensor.numpy().tobytes())
        hashes.append(h.hexdigest())
        truth=np.load(_paper_location(ROOT/'models'/c['name']/'truth.npz'))
        np.testing.assert_array_equal(truth['dictionary'],dist.A.cpu().numpy())
        np.testing.assert_array_equal(truth['permutation'],dist.permutation)
        del saved
    print('LOADED',job['name'],'seconds',round(time.monotonic()-started,1),flush=True)
    widths=[c['width'] for c in cs]
    decoder=[torch.nn.functional.normalize(m.decoder.detach(),dim=1) for m in models]
    cosine={};diagnostics=[]
    for i in range(len(widths)-1):
        for j in range(i+1,len(widths)):
            score=(decoder[i]@decoder[j].T).cpu().numpy()
            graph=csr_matrix(score>=np.float32(THRESHOLD))
            cosine[i,j]=graph;save_npz(_paper_location(directory/f'cosine_{widths[i]}_to_{widths[j]}.npz'),graph)
            diagnostics.append(dict(metric='decoder_cosine',source=widths[i],target=widths[j],
                                    edges=graph.nnz,near_threshold=int((np.abs(score-THRESHOLD)<1e-6).sum())))
    rows=solve(cosine,widths,'decoder_cosine',directory)
    print('COSINE_COMPLETE',job['name'],'seconds',round(time.monotonic()-started,1),flush=True)
    z,row_hash=collect_shared(models,dist,cs[0]['eval_seed']+1,cs[0]['test_samples'],cs[0]['batch_size'])
    print('ACTIVATIONS_COLLECTED',job['name'],'seconds',round(time.monotonic()-started,1),flush=True)
    ms=[moments(a) for a in z]
    np.savez_compressed(_paper_location(directory/'activation_moments.npz'),
                        **{f'w{w}_{key}':value for w,m in zip(widths,ms) for key,value in m.items()})
    pearson={}
    for i in range(len(widths)-1):
        for j in range(i+1,len(widths)):
            score=pearson_scores(z[i],z[j],ms[i],ms[j])
            graph=csr_matrix(score>=THRESHOLD)
            pearson[i,j]=graph;save_npz(_paper_location(directory/f'pearson_{widths[i]}_to_{widths[j]}.npz'),graph)
            diagnostics.append(dict(metric='activation_pearson',source=widths[i],target=widths[j],
                                    edges=graph.nnz,near_threshold=int((np.abs(score-THRESHOLD)<1e-6).sum())))
            print('PEARSON',job['name'],widths[i],widths[j],'edges',graph.nnz,
                  'seconds',round(time.monotonic()-started,1),flush=True)
    rows+=solve(pearson,widths,'activation_pearson',directory)
    for row in rows:row.update(kind=job['kind'],alpha=job['alpha'],k=job['k'],seed=job['seed'])
    summary=dict(complete=True,name=job['name'],code_hash=version,checkpoints=job['checkpoints'],
                 model_state_sha256=hashes,training_examples=BUDGET,threshold=THRESHOLD,
                 evaluation_examples=cs[0]['test_samples'],evaluation_seed=cs[0]['eval_seed']+1,
                 evaluation_batch_size=cs[0]['batch_size'],evaluation_rows_sha256=row_hash,
                 constant_features={str(w):int((~m['valid']).sum()) for w,m in zip(widths,ms)},
                 exhaustive=True,matching='maximum common subset into every larger width',
                 diagnostics=diagnostics,rows=rows,seconds=time.monotonic()-started,
                 environment=dict(torch=torch.__version__,device=device,
                                  gpu=torch.cuda.get_device_name() if device=='cuda' else None,
                                  job=os.environ.get('SLURM_JOB_ID')))
    atomic_json(complete,summary)
    print('COMPLETE',job['name'],'seconds',round(summary['seconds'],1),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','run'])
    p.add_argument('--index',type=int,default=0);p.add_argument('--device',default='cuda')
    a=p.parse_args()
    prepare() if a.stage=='prepare' else run(a.index,a.device)
