#!/usr/bin/env python3
"""Compare portable SAE inference to the original forward pass on actual embeddings."""
from project_paths import resource_path as _paper_path, resource_location as _paper_location
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=_paper_path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))
from atomic_features.sae import SAE
from atomic_features.bundles import write_json
from trainer.sae import TopKSAE
from project_paths import get_path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,default=ROOT/'releases/0.1.0')
    args=p.parse_args();models=args.release/'models'
    catalog=json.loads((models/'models.json').read_text())
    if len(catalog)!=26:raise ValueError('Expected 26 SAEs')
    for record in catalog:
        portable=SAE(models/record['checkpoint'])
        if portable.W_enc.shape!=(record['act_size'],record['dict_size']):raise ValueError('Incorrect model shape')
    checks=[]
    torch.set_num_threads(1)
    for family in ('gemini','nemotron'):
        name=f'{family}_m512_k32'
        state=torch.load(_paper_location(_paper_path(get_path('models_dir'))/name),map_location='cpu',weights_only=True,mmap=True)
        cfg=dict(state['config']);cfg['device']='cpu'
        reference=TopKSAE(cfg);reference.load_state_dict(state['model_state_dict']);reference.eval()
        inputs=np.load(_paper_location(args.release/'evaluation/wordfreq'/f'embeddings_{family}.npy'),mmap_mode='r',allow_pickle=False)[:32].astype(np.float32)
        with torch.no_grad():expected=reference(torch.from_numpy(inputs))['feature_acts'].numpy()
        portable=SAE(models/name)
        actual=portable.encode(inputs,batch_size=32,backend='torch').toarray()
        np.testing.assert_array_equal(actual,expected)
        numpy_acts=portable.encode(inputs,batch_size=32).toarray()
        if not np.array_equal(numpy_acts>0,expected>0):raise ValueError('Sample NumPy support differs')
        np.testing.assert_allclose(numpy_acts,expected,atol=2e-6,rtol=2e-5)
        checks.append(dict(model=name,rows=32,torch_exact=True,numpy_max_abs_error=float(np.max(np.abs(numpy_acts-expected))),numpy_support_equal=True))
    report=dict(status='passed',model_shapes_checked=26,comparisons=checks)
    write_json(args.release/'verification/model-parity.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
