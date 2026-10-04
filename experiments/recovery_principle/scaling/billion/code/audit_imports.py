"""Prove imported checkpoints preserve all training state exactly."""
from project_paths import resource_path as _paper_path, resource_location as _paper_location
import json
import torch
from common import ROOT,atomic_json

def equal(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b

if __name__=='__main__':
    torch.set_num_threads(2)
    rows=json.loads((ROOT/'results/imported.json').read_text())
    for r in rows:
        source=torch.load(_paper_location(r['checkpoint']),map_location='cpu',weights_only=False,mmap=True)
        dest=torch.load(_paper_location(ROOT/'models'/r['name']/f"n{r['examples']}/model.pt"),map_location='cpu',weights_only=False,mmap=True)
        for key in ['model','optimizer','numpy_rng','torch_rng','cuda_rng','examples','step']:equal(source[key],dest[key])
    atomic_json(ROOT/'results/import_audit.json',{'checked':len(rows),'errors':[]})
    print('Exact training-state checks passed:',len(rows))
