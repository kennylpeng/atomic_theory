#!/usr/bin/env python3
"""Inventory training shard metadata and NPY headers without reading dense tensors."""
from project_paths import resource_path as _paper_path, resource_location as _paper_location
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
ROOT=_paper_path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from atomic_features.bundles import sha256,write_json
from scripts.generate_model_configs import FULL_DATASETS
from project_paths import get_path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gemini',type=Path,default=_paper_path(get_path('gemini_embeddings_dir')))
    p.add_argument('--nemotron',type=Path,default=_paper_path(get_path('nemotron_embeddings_dir')))
    p.add_argument('--output',type=Path,default=ROOT/'generated/training-shards.json')
    args=p.parse_args();records=[];totals={}
    audit=json.loads((ROOT/'release/provenance/training-corpora.json').read_text())
    for family in ('gemini','nemotron'):
        root=getattr(args,family);counts=Counter();shards=0
        for dataset in FULL_DATASETS:
            print(f'Inspecting {family}/{dataset}',flush=True)
            for path in sorted((root/dataset).rglob('shard_meta.json')):
                meta=json.loads(path.read_text())
                tensor=path.parent/meta['embedding_file']
                a=np.load(_paper_location(tensor),mmap_mode='r',allow_pickle=False)
                if list(a.shape)!=[int(meta['rows']),int(meta['dim'])] or str(a.dtype)!=meta['dtype']:
                    raise ValueError(f'Shard/header mismatch: {path}')
                records.append(dict(family=family,dataset=dataset,path=path.relative_to(root).as_posix(),
                                    metadata_sha256=sha256(path),rows=int(meta['rows']),dimension=int(meta['dim']),dtype=meta['dtype'],
                                    embedding_path=tensor.relative_to(root).as_posix(),embedding_bytes=tensor.stat().st_size,
                                    settings={k:meta[k] for k in ('model_id','task_type','normalize_embeddings','text_column') if k in meta}))
                counts[dataset]+=int(meta['rows']);shards+=1
        if dict(counts)!=audit[family]['counts'] or shards!=audit[family]['shards']:
            raise ValueError(f'Corpus changed since paper audit: {family}')
        totals[family]=dict(rows=sum(counts.values()),shards=shards,by_dataset=dict(counts))
    write_json(args.output,dict(schema_version=1,scope='Hashes identify shard metadata only. Dense tensor content is not hashed or bundled; shapes/dtypes and file sizes are checked.',totals=totals,shards=records))
    print(json.dumps(totals,indent=2))


if __name__=='__main__':main()
