"""Compute the paper's absolute-cosine maximum-cardinality heatmap at t=.7.

Use the exclusive-match comparison's layout and color normalization.
Each matching is certified by an equal-size vertex cover. Cached pair results
are specific to abs(cosine) >= float32(.7); no signed-metric cache is reused.
"""
from project_paths import resource_path as _paper_path, resource_location as _paper_location
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import csr_matrix, save_npz
from scipy.sparse.csgraph import maximum_bipartite_matching
from split_16384_match_heatmaps import SPLITS, model_paths, load_vectors, plot_combined_heatmaps
from split_threshold_match_heatmaps import similarity_matrix
from validate_split_cardinality_outputs import certify_maximum
from project_paths import get_path

ROOT=_paper_path(__file__).resolve().parents[2]
OUT=ROOT/'full_experiments/results/split_matches/absolute_maximum_cardinality_0.7'
EXCLUSIVE_REFERENCE=ROOT/'full_experiments/results/split_matches/t1_0.7_t2_0.7/matches'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot-only', action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if not args.plot_only:
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False
        rows=[]
        for model in ('gemini','nemotron'):
            for rep in ('sae','kmeans','pca'):
                vectors=[load_vectors(p,rep) for _,p in model_paths(_paper_path(get_path('models_dir')),model,rep)]
                for i in range(5):
                    for j in range(i+1,5):
                        name=f'{model}_{rep}_{SPLITS[i]}_to_{SPLITS[j]}'
                        cache=OUT/'pairs'/name
                        cache.mkdir(parents=True,exist_ok=True)
                        meta=cache/'counts.json'
                        if meta.exists():
                            record=json.loads(meta.read_text())
                        else:
                            sim=similarity_matrix(vectors[i],vectors[j],'cuda',512)
                            np.abs(sim,out=sim)
                            graph=csr_matrix(sim>=np.float32(.7))
                            assignment=maximum_bipartite_matching(graph,perm_type='column')
                            src=np.flatnonzero(assignment>=0);dst=assignment[src]
                            certify_maximum(graph,src,dst)
                            assert np.all(sim[src,dst]>=np.float32(.7))
                            np.savez_compressed(_paper_location(cache/'assignment.npz'),source=src,destination=dst,absolute_cosine=sim[src,dst])
                            save_npz(_paper_location(cache/'graph.npz'),graph)
                            record=dict(matches=len(src),left_features=len(vectors[i]),right_features=len(vectors[j]),certified=True)
                            meta.write_text(json.dumps(record,indent=2)+'\n')
                            del sim,graph
                        for a,b,width in [(i,j,record['left_features']),(j,i,record['right_features'])]:
                            rows.append(dict(model=model,representation=rep,source_split=SPLITS[a],comparison_split=SPLITS[b],matches=record['matches'],source_features=width,proportion=record['matches']/width))
                        print(name,record['matches'],flush=True)
        with (OUT/'counts.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    rows=list(csv.DictReader((OUT/'counts.csv').open()))
    assert len(rows)==120
    matrices={};counts={};reference_max=0.
    for model in ('gemini','nemotron'):
        for rep in ('sae','kmeans','pca'):
            key=model,rep
            group=[r for r in rows if (r['model'],r['representation'])==key]
            widths={int(r['source_features']) for r in group};assert len(widths)==1
            counts[key]=widths.pop();matrix=np.zeros((5,5),dtype=int)
            for r in group:matrix[SPLITS.index(r['source_split']),SPLITS.index(r['comparison_split'])]=int(r['matches'])
            assert np.array_equal(matrix,matrix.T)
            assert np.all(matrix<=counts[key])
            matrices[key]=matrix
            stem='16384' if rep=='sae' else rep
            reference=np.genfromtxt(EXCLUSIVE_REFERENCE/f'{model}_{stem}_split_match_matrix.csv',delimiter=',',skip_header=1)[:,1:]
            reference_max=max(reference_max,float(np.nanmax(reference/counts[key])))
    for ext in ('png','pdf'):
        plot_combined_heatmaps(matrices,counts,list(SPLITS),OUT/f'combined_proportions.{ext}',color_max=reference_max)
    meta=dict(metric='maximum-cardinality one-to-one matching',similarity='absolute cosine',threshold=.7,comparison='>= float32(0.7)',color_map='viridis',color_min=0,color_max=reference_max,color_reference=str(EXCLUSIVE_REFERENCE),clipped_cells=sum(int(np.count_nonzero(matrices[k]/counts[k]>reference_max)) for k in matrices))
    (OUT/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta),flush=True)

if __name__=='__main__':main()
