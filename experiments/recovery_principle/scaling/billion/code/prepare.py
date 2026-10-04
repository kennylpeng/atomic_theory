"""Import every compatible evaluated milestone, preserving optimizer and RNG."""
from project_paths import resource_path as _paper_path, resource_location as _paper_location
import json,shutil
import torch
from common import ROOT,BASE,atomic_json,digest

def signature(c):
    return {k:v for k,v in c.items() if k not in ('name','budgets','continuation','checkpoint_seconds')}

def main():
    torch.set_num_threads(2)
    sources={}
    for sr in [BASE/'scaling',BASE/'scaling/leveling']:
        for p in sorted((sr/'results').glob('*/config.json')):
            c=json.loads(p.read_text()); key=json.dumps(signature(c),sort_keys=True)
            sources.setdefault(key,[]).append((sr,c))
    imported=[]
    for c in json.loads((ROOT/'configs/sweep.json').read_text()):
        matches=sources.get(json.dumps(signature(c),sort_keys=True),[])
        if not matches:continue
        md=ROOT/'models'/c['name'];rd=ROOT/'results'/c['name']
        md.mkdir(parents=True,exist_ok=True);rd.mkdir(parents=True,exist_ok=True)
        if (rd/'config.json').exists():assert json.loads((rd/'config.json').read_text())==c
        atomic_json(rd/'config.json',c)
        for n in c['budgets']:
            found=None
            for sr,old in matches:
                mp=sr/'models'/old['name']/f'n{n}/model.pt';rp=sr/'results'/old['name']/f'n{n}'
                if mp.exists() and (rp/'summary.json').exists():found=(sr,old,mp,rp)
            if found is None:break # only import a contiguous evaluated prefix
            sr,old,mp,rp=found
            dest=md/f'n{n}';res=rd/f'n{n}';dest.mkdir(exist_ok=True);res.mkdir(exist_ok=True)
            provenance={'checkpoint':str(mp.resolve()),'source_config_hash':digest(old)}
            if not (dest/'model.pt').exists():
                cp=torch.load(_paper_location(mp),map_location='cpu',weights_only=False)
                assert cp['config']==old and cp['examples']==n
                cp['config']=c;cp['continuation']=provenance
                tmp=dest/'model.tmp';torch.save(cp,_paper_location(tmp));tmp.replace(dest/'model.pt');del cp
            shutil.copy2(sr/'models'/old['name']/'truth.npz',md/'truth.npz')
            shutil.copy2(rp/'recovery.npz',res/'recovery.npz')
            summary=json.loads((rp/'summary.json').read_text());summary.update(config_hash=digest(c),continuation=provenance)
            atomic_json(res/'summary.json',summary)
            imported.append({'name':c['name'],'examples':n,**provenance})
    atomic_json(ROOT/'results/imported.json',imported)
    print('Imported',len(imported),'milestones across',len({v['name'] for v in imported}),'trajectories',flush=True)

if __name__=='__main__':main()
