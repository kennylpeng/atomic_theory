"""Deduplicated alpha/SAE-k grid; generating K remains eight."""
import json
from common import ROOT
BUDGETS=[1000000*2**i for i in range(11)]

def config(kind,M,d,width,k,alpha,seed,stage='main',budgets=None):
    data={'kind':kind,'M':M,'d':d,'K':8,'alpha':alpha,'dictionary_seed':100+seed,'dictionary':'gaussian','permutation':'identity','permutation_seed':700+seed}
    if alpha<=1:data['allow_outside_theorem']=True
    return dict(name=f'{stage}_{kind}_M{M}_w{width}_k{k}_a{alpha}_s{seed}',data=data,width=width,top_k=k,seed=seed,data_seed=10000+seed,eval_seed=20000+seed,budgets=budgets or BUDGETS,batch_size=1024,lr=.001,aux=.25,validation_samples=200000,test_samples=400000,checkpoint_seconds=600)

def grid():
    return [config(kind,M,512,w,k,a,s) for s in range(3) for kind,M in [('flat',4096),('hierarchical',3072)] for w in [128,256,512,1024,2048,M] for a in [.8,1.2,1.6,2.,2.4] for k in ([1,2,4,6,8,10,12,16,32] if a==1.6 else [8])]

if __name__=='__main__':
    rows=grid();assert len(rows)==468 and len({r['name'] for r in rows})==468
    (ROOT/'configs').mkdir(parents=True, exist_ok=True)
    (ROOT/'configs/sweep.json').write_text(json.dumps(rows,indent=2)+'\n')
    pilots=[config(kind,M,32,M,k,.8,0,stage='pilot',budgets=[1024,2048,4096]) for kind,M in [('flat',24),('hierarchical',24)] for k in [1,8,16]]
    for c in pilots:c.update(validation_samples=1024,test_samples=2048,batch_size=128)
    (ROOT/'configs/pilot.json').write_text(json.dumps(pilots,indent=2)+'\n')
    print(len(rows),'trajectories',len(rows)*len(BUDGETS),'milestones',sum(c['budgets'][-1] for c in rows),'examples')
