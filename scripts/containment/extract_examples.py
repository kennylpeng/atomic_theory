from project_paths import resource_path as _paper_path, resource_location as _paper_location
import numpy as np,pandas as pd,json,pathlib,pyarrow.parquet as pq,sys
base=pathlib.Path('/resources/research_data_dir')
out=(_paper_path(__file__).resolve().parents[2] / "full_experiments/results/gemini_131k_hierarchical_families")
d=pd.read_csv(base/'parent_child_pairs/gemini_m131072_k128_gt_0p05_rlower_0p7/immediate_parent_child_pairs.csv')
r=np.load(_paper_location(base/'activation_caches/gemini_binned_random_examples_v1/final/gemini_m131072_k128/sample_rows.npy'),mmap_mode='r')
a=np.load(_paper_location(base/'activation_caches/gemini_binned_random_examples_v1/final/gemini_m131072_k128/sample_activations.npy'),mmap_mode='r')
def fetch(ids,name):
 picks={}
 for f in ids:
  picks[int(f)]=[(int(r[f,b,j]),float(a[f,b,j]),b) for b in range(6,0,-1) for j in range(min(3, int((r[f,b]>=0).sum())))]
 wanted=set(x[0] for xs in picks.values() for x in xs)
 cache=out/'texts.json';texts=json.load(open(_paper_location(cache))) if cache.exists() else {}
 missing=wanted-set(map(int,texts))
 cat=base/'activation_caches/gemini_binned_random_examples_v1/text_catalog/final';m=json.load(open(_paper_location(cat/'manifest.json')))
 for part in m['parts']:
  need=[x for x in missing if part['global_row_min']<=x<=part['global_row_max']]
  if not need: continue
  tab=pq.read_table(cat/part['file'],columns=['global_row','dataset','full_text'],filters=[('global_row','in',need)])
  for x in tab.to_pylist():texts[str(x['global_row'])]=x
 cache.write_text(json.dumps(texts))
 result={str(f):[dict(texts[str(row)],activation=act,bin=b) for row,act,b in xs] for f,xs in picks.items()}
 (out/(name+'.json')).write_text(json.dumps(result,indent=2))
 for f,xs in result.items():
  print(f,'children',int((d.parent_feature==int(f)).sum()),'::',' | '.join(x['full_text'][:150].replace('\n',' ') for x in xs[:4]))
if __name__=='__main__':
 if len(sys.argv)>1: ids=list(map(int,sys.argv[1].split(',')));name=sys.argv[2]
 else:
  g=d.groupby('parent_feature').agg(n=('child_feature','size'),support=('parent_support','first'))
  ids=g.sort_values('n',ascending=False).head(100).index.tolist();name='branching_parent_examples'
 fetch(ids,name)
