"""Shared synthetic TopK model, evaluation collection and serialization."""

import hashlib
import json
import numpy as np
import torch
from torch import nn
from scipy import sparse

class TopK(nn.Module):
    """Per-example ReLU TopK, tied initialization, unit decoder rows.

    Matches trainer/sae.py's forward map. Bias is frozen at zero for the
    primary paper-objective condition; optional auxiliary dead-feature loss.
    """
    def __init__(self,d,width,k,bias=False):
        super().__init__()
        self.k=k
        self.encoder=nn.Parameter(torch.empty(d,width))
        nn.init.kaiming_uniform_(self.encoder)
        self.decoder=nn.Parameter(nn.functional.normalize(self.encoder.detach().T.clone(),dim=1))
        self.bias=nn.Parameter(torch.zeros(d),requires_grad=bias)
        self.register_buffer('inactive',torch.zeros(width,dtype=torch.long))

    def encode(self,x):
        pre=torch.relu((x-self.bias)@self.encoder)
        vals,idx=pre.topk(min(self.k,pre.shape[1]),dim=1)
        return vals,idx,pre

    def forward(self,x):
        vals,idx,pre=self.encode(x)
        acts=torch.zeros_like(pre).scatter(1,idx,vals)
        return acts@self.decoder+self.bias,acts,vals,idx,pre

    @torch.no_grad()
    def constrain(self):
        unit=nn.functional.normalize(self.decoder,dim=1)
        if self.decoder.grad is not None:
            self.decoder.grad.sub_((self.decoder.grad*unit).sum(1,keepdim=True)*unit)
        self.decoder.copy_(unit)

def digest(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()

def atomic_json(path,data):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2)+'\n')
    temp.replace(path)

@torch.no_grad()
def collect(model,dist,seed,n,batch):
    rng=np.random.default_rng(seed)
    true_ids=[]; learned_ids=[]; learned_values=[]
    mse=energy=0.
    for start in range(0,n,batch):
        x,ids,_=dist.batch(rng,min(batch,n-start))
        recon,_,vals,idx,_=model(x)
        mse+=float((recon-x).square().sum())
        energy+=float(x.square().sum())
        true_ids.append(ids)
        learned_ids.append(idx.cpu().numpy())
        learned_values.append(vals.cpu().numpy())
    ti=np.concatenate(true_ids); li=np.concatenate(learned_ids); lv=np.concatenate(learned_values)
    y=sparse.csr_matrix((np.ones(ti.size,np.float32),(np.repeat(np.arange(n),ti.shape[1]),ti.ravel())),shape=(n,dist.spec.M))
    z=sparse.csr_matrix((lv.ravel(),(np.repeat(np.arange(n),li.shape[1]),li.ravel())),shape=(n,model.decoder.shape[0]))
    z.eliminate_zeros(); z.sort_indices(); y.sort_indices()
    return y,z,{'mse_per_example':mse/n,'relative_mse':mse/max(energy,1e-30),'mean_l0':z.nnz/n}
