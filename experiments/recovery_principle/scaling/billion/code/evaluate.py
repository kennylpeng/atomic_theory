"""Exact projected geometry in vectorized form; consistent report criteria."""
import numpy as np
import torch
from common import collect,activation_recovery,prefix,distinct_triple


def geometry(A,B,kind,t=.9):
    # Float64 reduces cancellation in the analytic projection formula.
    A=np.asarray(A,dtype=np.float64);B=np.asarray(B,dtype=np.float64)
    A=A/np.linalg.norm(A,axis=1,keepdims=True)
    B=B/np.linalg.norm(B,axis=1,keepdims=True)
    dots=A@B.T
    out={'atom_similarity':np.abs(dots).max(1),'atom_argmax':np.abs(dots).argmax(1)}
    if kind=='flat':
        out['recovered']=out['atom_similarity']>=t
        return out
    n=len(A)//3
    p=dots[::3]
    cp=(A.reshape(n,3,-1)[:,1:]*A[::3,None]).sum(2)
    numerator=dots.reshape(n,3,-1)[:,1:]-cp[:,:,None]*p[:,None,:]
    bn=np.sqrt(np.maximum(0,1-p*p))
    cn=np.sqrt(np.maximum(0,1-cp*cp))
    score=np.abs(numerator)/np.maximum(cn[:,:,None]*bn[:,None,:],1e-15)
    score=np.where(bn[:,None,:]>1e-6,score,0)
    score=np.clip(score,0,1)
    parents=np.abs(p)>=t;children=score>=t
    families=np.array([distinct_triple(np.flatnonzero(parents[i]),np.flatnonzero(children[i,0]),np.flatnonzero(children[i,1])) for i in range(n)])
    out.update(parent_similarity=np.abs(p).max(1),projected_child_similarity=score.max(2),
               parent_recovered=parents.any(1),child_recovered=children.any(2),recovered=families)
    return out


@torch.no_grad()
def evaluate(model,dist,c):
    model.eval()
    r=geometry(dist.A.cpu().numpy(),model.decoder.detach().cpu().numpy(),c['data']['kind'])
    vy,vz,vs=collect(model,dist,c['eval_seed'],c['validation_samples'],c['batch_size'])
    ty,tz,ts=collect(model,dist,c['eval_seed']+1,c['test_samples'],c['batch_size'])
    r.update(activation_recovery(vy,vz,ty,tz,targets=(.8,)))
    f=r['activation_distinct_test_f1_0.8']>=.8
    if c['data']['kind']=='hierarchical':f=f.reshape(-1,3).all(1)
    r['activation_recovered']=f
    q=r['recovered']
    summary={'geometric_prefix':prefix(q,.1),'geometric_count':int(q.sum()),'geometric_fraction':float(q.mean()),
             'activation_prefix':prefix(f,.1),'activation_count':int(f.sum()),'activation_fraction':float(f.mean()),
             'validation':vs,'test':ts,'test_features_with_20_positives':int((r['test_positives']>=20).sum())}
    if c['data']['kind']=='hierarchical':
        fa=r['activation_distinct_test_f1_0.8'].reshape(-1,3)>=.8
        summary.update(parent_geometric_fraction=float(r['parent_recovered'].mean()),
                       child_geometric_fraction=float(r['child_recovered'].mean()),
                       parent_activation_fraction=float(fa[:,0].mean()),child_activation_fraction=float(fa[:,1:].mean()))
    model.train()
    return r,summary
