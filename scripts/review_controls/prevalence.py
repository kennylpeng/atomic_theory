"""Full-corpus prevalence versus exact signed direction persistence/matching.

Effect sizes are descriptive across features, not independent-trial p-values.
Witness membership need not be unique; edge eligibility is a sensitivity check.
"""
from project_paths import resource_path as _paper_path, resource_location as _paper_location
import csv, hashlib, itertools, json, sys
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=_paper_path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.stability.persistent_matching import validate_witness
from scripts.stability.validate_split_cardinality_outputs import certify_maximum
RESULTS=ROOT/'full_experiments/results';OUT=RESULTS/'sparsity_seed_controls/prevalence'
WIDTHS={512:32,1024:32,2048:32,4096:32,8192:64,16384:64,32768:64,65536:128}
SPLITS=['main','wikipedia','no_wikipedia','random1','random2']
PROVENANCE={}
def sha(path):
 h=hashlib.sha256(path.read_bytes()).hexdigest();PROVENANCE[str(path)]=h;return h
def read(path):sha(path);return json.loads(path.read_text())
def arr(path):sha(path);return np.load(_paper_location(path))
def write_csv(name,rows):
 with (OUT/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def auc(values,outcome):
 n=int(outcome.sum());m=len(outcome)-n
 if not n or not m:return None
 ranks=rankdata(values);return float((ranks[outcome].sum()-n*(n+1)/2)/(n*m))
def effect(values,outcome,mask=None):
 if mask is not None:values=values[mask];outcome=outcome[mask]
 pos=values[outcome];neg=values[~outcome]
 med1=float(np.median(pos)) if len(pos) else None;med0=float(np.median(neg)) if len(neg) else None
 return dict(features=len(values),selected=int(outcome.sum()),median_selected=med1,median_unselected=med0,median_ratio=med1/med0 if med0 and med1 is not None else None,auc=auc(values,outcome))
def bins(values):
 # Quantile boundaries respect ties, including the mass at zero.
 cut=np.unique(np.quantile(values,np.linspace(0,1,11)))
 return np.searchsorted(cut[1:-1],values,side='right')
def curves(values,outcome,context):
 q=bins(values);result=[]
 for b in np.unique(q):
  mask=q==b
  result.append(dict(**context,bin=int(b),n=int(mask.sum()),prevalence_median=float(np.median(values[mask])),selected=int(outcome[mask].sum()),selection_rate=float(outcome[mask].mean())))
 return result
def conditional_auc(own,other,outcome):
 q=bins(own);num=0.;den=0
 for b in np.unique(q):
  mask=q==b;y=outcome[mask];weight=int(y.sum())*int((~y).sum());a=auc(other[mask],y)
  if a is not None:num+=a*weight;den+=weight
 return num/den if den else None

def main():
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'per_feature').mkdir(exist_ok=True)
 effects=[];width_curves=[];cross=[];cross_curves=[];paired=[]
 for family in ('gemini','nemotron'):
  persist=RESULTS/'persistent_stability_signed_0.7'/family;manifest=read(persist/'manifest.json')
  fingerprints={p['path']:p for p in manifest['source_files']}
  for width,k in WIDTHS.items():
   name=f'{family}_m{width}_k{k}';p=RESULTS/'main_sae_activation_statistics_zero_median/per_feature'/f'{name}.npz'
   with arr(p) as z:
    rates=z['rates'].copy();counts=z['counts'].copy();n=int(z['rows']);prov=json.loads(str(z['provenance']));modes=z['threshold_modes'].tolist()
   assert modes==['zero','median_positive'];assert np.array_equal(rates,counts/n)
   fp=fingerprints[prov['model']['path']];assert fp['size']==prov['model']['size_bytes'] and fp['mtime_ns']==prov['model']['mtime_ns']
   targets=[w for w in list(WIDTHS)+[131072] if w>width]
   graphs=[]
   for w in targets:
    path=persist/f'graph_{width}_to_{w}.npz';sha(path);graphs.append(load_npz(_paper_location(path)))
   with arr(persist/f'witness_{width}.npz') as z:
    source=z['source'].copy();dest=[z[f'target_{w}'].copy() for w in targets]
   validate_witness(graphs,source,dest)
   selected=np.zeros(width,dtype=bool);selected[source]=True
   eligible=np.logical_and.reduce([np.diff(g.indptr)>0 for g in graphs]);assert np.all(eligible[selected])
   np.savez_compressed(_paper_location(OUT/'per_feature'/f'width_{name}.npz'),rates=rates,counts=counts,rows=n,selected=selected,eligible=eligible)
   for j,mode in enumerate(modes):
    for membership,outcome in [('witness',selected),('eligible_all_targets',eligible)]:
     context=dict(family=family,width=width,threshold=mode,membership=membership)
     for population,mask in [('all',np.ones(width,dtype=bool)),('positive_prevalence',rates[j]>0)]:
      effects.append(dict(**context,population=population,**effect(rates[j],outcome,mask)))
     width_curves+=curves(rates[j],outcome,context)
  splitroot=RESULTS/'split_sae_activation_sketch512_candidates32_top5'/family
  moments={};manifests={}
  for split in SPLITS:
   m=read(splitroot/split/'manifest.json');manifests[split]=m
   with arr(splitroot/split/'moments.npz') as z:
    assert str(z['manifest_id'])==m['id'];assert int(z['rows'])==m['rows'];support=z['support'].copy()
   assert len(support)==5*16384
   moments[split]={model['split']:support[i*16384:(i+1)*16384]/m['rows'] for i,model in enumerate(m['models'])}
   assert all(0<=v.min()<=v.max()<=1 for v in moments[split].values())
  # Exact agreement between independently aggregated full-corpus statistics.
  with arr(RESULTS/'main_sae_activation_statistics_zero_median/per_feature'/f'{family}_m16384_k64.npz') as z:
   assert np.array_equal(z['rates'][0],moments['main']['main'])
  matchroot=RESULTS/'split_matches/signed_maximum_cardinality_pca_absolute_0.7'
  meta=read(matchroot/'metadata.json');fp={_paper_path(p['path']).name:p for p in meta['source_files']}
  for m in manifests.values():
   for model in m['models']:
    original=fp[model['name']];stat=_paper_path(original['path']).stat()
    assert stat.st_size==original['size'] and stat.st_mtime_ns==original['mtime_ns']
    assert model['sha256']==next(x['sha256'] for x in manifests['main']['models'] if x['name']==model['name'])
  for a,b in itertools.combinations(SPLITS,2):
   path=matchroot/'pairs'/f'{family}_sae_{a}_to_{b}'
   sha(path/'graph.npz');g=load_npz(_paper_location(path/'graph.npz'))
   with arr(path/'assignment.npz') as z:src=z['source'].copy();dst=z['destination'].copy()
   certify_maximum(g,src,dst)
   # Both directions use the same maximum matching, indexed in their own SAE.
   for source_split,target_split,indices,graph in [(a,b,src,g),(b,a,dst,g.T.tocsr())]:
    own=moments[source_split][source_split];other=moments[target_split][source_split];both=np.minimum(own,other)
    selected=np.zeros(16384,dtype=bool);selected[indices]=True;eligible=np.diff(graph.indptr)>0
    context=dict(family=family,source_split=source_split,target_split=target_split,disjoint_pair={a,b} in ({'wikipedia','no_wikipedia'},{'random1','random2'}))
    np.savez_compressed(_paper_location(OUT/'per_feature'/f'split_{family}_{source_split}_to_{target_split}.npz'),own=own,other=other,both=both,selected=selected,eligible=eligible)
    for membership,outcome in [('witness',selected),('eligible_neighbor',eligible)]:
     for metric,values in [('own',own),('other',other),('minimum_both',both)]:
      c=dict(**context,membership=membership,prevalence=metric)
      for population,mask in [('all',np.ones(16384,dtype=bool)),('positive_both',both>0)]:
       cross.append(dict(**c,population=population,other_auc_within_own_deciles=conditional_auc(own[mask],other[mask],outcome[mask]),**effect(values,outcome,mask)))
      cross_curves+=curves(values,outcome,c)
   pa=moments[a][a][src];pb=moments[b][b][dst]
   # Partner-frequency summary is descriptive; no fabricated unmatched partners.
   paired.append(dict(family=family,split_a=a,split_b=b,matches=len(src),median_a_own=float(np.median(pa)),median_b_own=float(np.median(pb)),median_min_own=float(np.median(np.minimum(pa,pb))),positive_in_both_own=int(((pa>0)&(pb>0)).sum())))
 write_csv('width_effects.csv',effects);write_csv('width_bins.csv',width_curves);write_csv('cross_distribution_effects.csv',cross);write_csv('cross_distribution_bins.csv',cross_curves);write_csv('matched_partner_prevalence.csv',paired)
 fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
 for ax,family in zip(axes,('gemini','nemotron')):
  for w in WIDTHS:
   r=[x for x in width_curves if x['family']==family and x['width']==w and x['threshold']=='zero' and x['membership']=='witness']
   ax.plot([x['prevalence_median'] for x in r],[x['selection_rate'] for x in r],'.-',label=f'{w:,}')
  ax.set(xscale='log',xlabel='Feature prevalence (positive activation)',title=family.capitalize(),ylim=(0,1));ax.grid(alpha=.2)
 axes[0].set_ylabel('Fraction in persistent matching');axes[1].legend(title='Width',fontsize=8,ncol=2);fig.tight_layout()
 for ext in ('png','pdf'):fig.savefig(OUT/f'width_prevalence.{ext}',dpi=180)
 plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
 for ax,family in zip(axes,('gemini','nemotron')):
  for a,b in [('wikipedia','no_wikipedia'),('no_wikipedia','wikipedia'),('random1','random2'),('random2','random1')]:
   r=[x for x in cross_curves if x['family']==family and x['source_split']==a and x['target_split']==b and x['membership']=='witness' and x['prevalence']=='minimum_both']
   ax.plot([x['prevalence_median'] for x in r],[x['selection_rate'] for x in r],'.-',label=f'{a} → {b}')
  ax.set(xscale='log',xlabel='Minimum prevalence across both distributions',title=family.capitalize(),ylim=(0,1));ax.grid(alpha=.2)
 axes[0].set_ylabel('Fraction with one-to-one direction match');axes[1].legend(fontsize=8);fig.tight_layout()
 for ext in ('png','pdf'):fig.savefig(OUT/f'cross_distribution_prevalence.{ext}',dpi=180)
 plt.close(fig)
 PROVENANCE[str(_paper_path(__file__))]=hashlib.sha256(_paper_path(__file__).read_bytes()).hexdigest()
 (OUT/'provenance.json').write_text(json.dumps(PROVENANCE,indent=2)+'\n')
 (OUT/'COMPLETE.json').write_text(json.dumps(dict(width_effect_rows=len(effects),cross_effect_rows=len(cross),independent_full_corpus_count_check=True,matching_witnesses_validated=True),indent=2)+'\n')
 print('Completed prevalence analyses:',len(effects),'width effects;',len(cross),'cross-distribution effects.',flush=True)
if __name__=='__main__':main()
