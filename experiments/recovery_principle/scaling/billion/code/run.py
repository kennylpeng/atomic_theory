"""One training trajectory with exact cumulative-example checkpoints and resumption."""
from project_paths import resource_path as _paper_path, resource_location as _paper_location
import argparse,json,os,time,hashlib
from pathlib import Path
import numpy as np
import torch
from common import ROOT,BASE,Distribution,Spec,TopK,atomic_json,digest
from evaluate import evaluate


def batch_plan(seen,target,batch_size):
    """Short boundary batches hit exact budgets without skipping examples."""
    while seen<target:
        size=min(batch_size,target-seen)
        yield size
        seen+=size


def save_checkpoint(path,model,opt,c,seen,step,rng):
    state={'model':model.state_dict(),'optimizer':opt.state_dict(),'config':c,'examples':seen,'step':step,
           'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),
           'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    tmp=path.with_suffix('.tmp');torch.save(state,_paper_location(tmp));tmp.replace(path)


def run(c,device):
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','4')))
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    out=ROOT/'results'/c['name'];md=ROOT/'models'/c['name']
    out.mkdir(parents=True,exist_ok=True);md.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==c
    atomic_json(out/'config.json',c)
    if (out/'complete.json').exists():print('ALREADY_COMPLETE',c['name']);return
    sources={str(p.relative_to(BASE)):p.read_text() for p in sorted((ROOT/'code').glob('*.py'))}
    sources.update({str(p.relative_to(BASE)):p.read_text() for p in sorted((BASE/'code').glob('*.py'))})
    atomic_json(out/'source_snapshot.json',sources)
    torch.manual_seed(c['seed'])
    dist=Distribution(Spec(**c['data']),device)
    truth=md/'truth.npz'
    if truth.exists():
        saved=np.load(_paper_location(truth));np.testing.assert_array_equal(saved['dictionary'],dist.A.cpu().numpy())
    else:np.savez_compressed(_paper_location(truth),dictionary=dist.A.cpu().numpy(),permutation=dist.permutation)
    model=TopK(c['data']['d'],c['width'],c['top_k'],False).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=c['lr'],betas=(.9,.99))
    rng=np.random.default_rng(c['data_seed']);seen=step=0
    saved_budgets=[n for n in c['budgets'] if (md/f'n{n}'/'model.pt').exists()]
    if saved_budgets:
        last=max(saved_budgets);saved=torch.load(_paper_location(md/f'n{last}'/'model.pt'),map_location=device,weights_only=False)
        assert saved['config']==c
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
        seen=saved['examples'];step=saved['step'];rng.bit_generator.state=saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng'].cpu())
        if device=='cuda':torch.cuda.set_rng_state_all([v.cpu() for v in saved['cuda_rng']])
        del saved
    rolling=md/'latest.pt'
    if rolling.exists():
        saved=torch.load(_paper_location(rolling),map_location=device,weights_only=False)
        assert saved['config']==c
        if saved['examples']>seen:
            model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
            seen=saved['examples'];step=saved['step'];rng.bit_generator.state=saved['numpy_rng']
            torch.set_rng_state(saved['torch_rng'].cpu())
            if device=='cuda':torch.cuda.set_rng_state_all([v.cpu() for v in saved['cuda_rng']])
        del saved
    started=time.time();last_save=time.monotonic()
    print('START',c['name'],'resume_examples',seen,flush=True)
    for budget in c['budgets']:
        result=out/f'n{budget}';modeldir=md/f'n{budget}'
        result.mkdir(exist_ok=True);modeldir.mkdir(exist_ok=True)
        if (result/'summary.json').exists():
            assert (modeldir/'model.pt').exists()
            continue
        if budget<seen:
            # Normally impossible: each checkpoint is evaluated before training on.
            raise RuntimeError('Earlier checkpoint needs evaluation; run repair rather than overwrite newer state')
        for size in batch_plan(seen,budget,c['batch_size']):
            x,_,_=dist.batch(rng,size)
            recon,acts,_,_,pre=model(x)
            loss=(recon-x).square().mean()
            with torch.no_grad():
                active=(acts>0).any(0);model.inactive.add_(1);model.inactive[active]=0
            aux=torch.zeros((),device=device);dead=model.inactive>=256
            if c['aux'] and bool(dead.any()):
                dp=pre[:,dead];v,i=dp.topk(min(64,dp.shape[1]),dim=1)
                az=torch.zeros_like(dp).scatter(1,i,v)
                aux=c['aux']*((az@model.decoder[dead])-(x-recon).detach()).square().mean()
            opt.zero_grad(set_to_none=True);(loss+aux).backward();model.constrain()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            with torch.no_grad():model.decoder.copy_(torch.nn.functional.normalize(model.decoder,dim=1))
            seen+=size;step+=1
            if time.monotonic()-last_save>=c.get('checkpoint_seconds',600):
                save_checkpoint(rolling,model,opt,c,seen,step,rng)
                last_save=time.monotonic()
            if step%1000==0:
                with (out/'training.jsonl').open('a') as f:f.write(json.dumps({'examples':seen,'step':step,'mse':float(loss),'aux':float(aux),'dead':int(dead.sum())})+'\n')
        assert seen==budget
        if not (modeldir/'model.pt').exists():save_checkpoint(modeldir/'model.pt',model,opt,c,seen,step,rng)
        before=rng.bit_generator.state
        recovery,summary=evaluate(model,dist,c)
        assert rng.bit_generator.state==before, 'Evaluation changed the training stream'
        np.savez_compressed(_paper_location(result/'recovery.npz'),**recovery)
        summary.update(examples=seen,step=step,config_hash=digest(c),seconds_this_session=time.time()-started,
                       gpu=torch.cuda.get_device_name() if device=='cuda' else 'cpu',
                       job=os.environ.get('SLURM_JOB_ID'),code_hash=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest())
        atomic_json(result/'summary.json',summary)
        print('CHECKPOINT',c['name'],budget,'prefix',summary['geometric_prefix'],'F1_prefix',summary['activation_prefix'],flush=True)
    atomic_json(out/'complete.json',{'config_hash':digest(c),'budgets':c['budgets'],'examples':seen})
    if rolling.exists():rolling.unlink()
    print('COMPLETE',c['name'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--index',type=int,default=0);p.add_argument('--device',default='cuda')
    a=p.parse_args();run(json.loads(a.manifest.read_text())[a.index],a.device)
