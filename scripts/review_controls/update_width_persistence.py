"""Refresh width-prevalence effects from unchanged cached per-feature rates.

Cross-distribution matching and corpus-level counting are unaffected. The input
rate arrays are preserved; only membership in the independent persistence
intersection and its descriptive effects/plots change.
"""
from project_paths import resource_path as _paper_path, resource_location as _paper_location
from pathlib import Path
import csv,hashlib,json,sys
import numpy as np
from scipy.stats import mannwhitneyu
ROOT=_paper_path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.review_controls import prevalence,median_report,plot_median_paper


def main():
 effects=[];bins=[];hashes={};checked=0
 old_provenance=json.loads((prevalence.OUT/'provenance.json').read_text())
 for family in ('gemini','nemotron'):
  folder=prevalence.RESULTS/'persistent_stability_signed_0.7'/family
  rows={r['width']:r for r in json.loads((folder/'summary.json').read_text())}
  for w,k in prevalence.WIDTHS.items():
   path=prevalence.OUT/'per_feature'/f'width_{family}_m{w}_k{k}.npz'
   with np.load(_paper_location(path)) as z:arrays={key:z[key].copy() for key in z.files}
   with np.load(_paper_location(folder/f'witness_{w}.npz')) as z:ids=z['source'].copy()
   selected=np.zeros(w,bool);selected[ids]=True
   assert selected.sum()==rows[w]['persistent_count']
   assert np.all(arrays['eligible'][selected])
   np.testing.assert_array_equal(arrays['rates'],arrays['counts']/arrays['rows'])
   arrays['selected']=selected;np.savez_compressed(_paper_location(path),**arrays)
   for j,mode in enumerate(('zero','median_positive')):
    for membership,outcome in [('witness',selected),('eligible_all_targets',arrays['eligible'])]:
     context=dict(family=family,width=w,threshold=mode,membership=membership)
     for population,mask in [('all',np.ones(w,bool)),('positive_prevalence',arrays['rates'][j]>0)]:
      values=arrays['rates'][j];result=prevalence.effect(values,outcome,mask)
      if outcome[mask].any() and (~outcome[mask]).any():
       a=values[mask][outcome[mask]];b=values[mask][~outcome[mask]]
       assert np.isclose(result['auc'],mannwhitneyu(a,b,method='asymptotic').statistic/(len(a)*len(b)),rtol=1e-12)
      effects.append(dict(**context,population=population,**result));checked+=1
     bins+=prevalence.curves(arrays['rates'][j],outcome,context)
   for source in (path,folder/'manifest.json',folder/f'witness_{w}.npz'):
    hashes[str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
  print('Refreshed cached width prevalence:',family,flush=True)
 for name,data in [('width_effects.csv',effects),('width_bins.csv',bins)]:
  prevalence.write_csv(name,data)
 # Preserve hashes for unchanged corpus inputs; refresh only changed membership
 # and method sources that were already represented in the input provenance.
 for name in list(old_provenance):
  path=_paper_path(name)
  if name in hashes:old_provenance[name]=hashes[name]
  elif path==_paper_path(prevalence.__file__):old_provenance[name]=hashlib.sha256(path.read_bytes()).hexdigest()
 old_provenance.update(hashes)
 old_provenance[str(_paper_path(__file__))]=hashlib.sha256(_paper_path(__file__).read_bytes()).hexdigest()
 (prevalence.OUT/'provenance.json').write_text(json.dumps(old_provenance,indent=2)+'\n')
 complete=json.loads((prevalence.OUT/'COMPLETE.json').read_text());complete.update(width_persistence_method='independent_pairwise_intersection',width_auc_effects_independently_verified=checked,cross_distribution_results_reused_unchanged=True)
 (prevalence.OUT/'COMPLETE.json').write_text(json.dumps(complete,indent=2)+'\n')
 median_report.width()
 plot_median_paper.main()
 print('Updated and independently checked',checked,'width-prevalence effects.',flush=True)

if __name__=='__main__':main()
