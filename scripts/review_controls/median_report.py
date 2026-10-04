"""Compare zero and Figure-10 median prevalence with frozen direction outcomes."""
from project_paths import resource_path as _paper_path, resource_location as _paper_location
import argparse,csv,hashlib,itertools,json,sys
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=_paper_path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.review_controls.prevalence import effect,curves,conditional_auc,SPLITS,WIDTHS
from scripts.stability.validate_split_cardinality_outputs import certify_maximum
RESULTS=ROOT/'full_experiments/results';BASE=RESULTS/'sparsity_seed_controls';BASELINE=BASE/'prevalence';OUT=BASE/'prevalence_median'
LABELS={'main':'Full','wikipedia':'Wiki','no_wikipedia':'Non-Wiki','random1':'Random 1','random2':'Random 2'}
MODES=['zero','median_positive'];TITLES=['Activation > 0','Activation > SAE pooled positive median']
def readcsv(path):return list(csv.DictReader(path.open()))
def writecsv(path,rows):
 with path.open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def digest(path):return hashlib.sha256(_paper_path(path).read_bytes()).hexdigest()
def savefig(fig,name):
 for ext in ['png','pdf']:fig.savefig(OUT/f'{name}.{ext}',dpi=180)
 plt.close(fig)
def table(headers,rows):return ['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,row))+' |' for row in rows]
def f3(value):return f'{float(value):.3f}' if value not in ('',None) else 'undefined'
def style(ax):
 ax.set_xscale('symlog',linthresh=1e-8,linscale=.5);ax.set_xlim(left=0);ax.set_ylim(0,1);ax.grid(alpha=.2);ax.set_xlabel('Feature prevalence (fraction of examples)')
 ticks=[0,1e-7,1e-5,1e-3,1e-1];ax.set_xticks(ticks,[r'$0$',r'$10^{-7}$',r'$10^{-5}$',r'$10^{-3}$',r'$10^{-1}$'])
def refresh_width_readme(effects):
 path=OUT/'README.md'
 if not path.exists():return
 text=path.read_text();rows=[]
 for family in ('gemini','nemotron'):
  for w in WIDTHS:
   group=[r for r in effects if r['family']==family and int(r['width'])==w and r['membership']=='witness']
   z=next(r for r in group if r['threshold']=='zero' and r['population']=='all')
   m=next(r for r in group if r['threshold']=='median_positive' and r['population']=='all')
   a=next(r for r in group if r['threshold']=='median_positive' and r['population']=='positive_prevalence')
   rows.append([family,w,f3(z['auc']),f3(m['auc']),f3(a['auc']),a['features']])
 headers=['Model','Width','AUC: activation > 0','AUC: activation > pooled median','AUC: >median, positive-prevalence features only','Positive-prevalence features at >median']
 start=text.index('| Model | Width |');end=text.index('\n\n',start)
 text=text[:start]+'\n'.join(table(headers,rows))+text[end:]
 path.write_text(text)

def width():
 OUT.mkdir(parents=True,exist_ok=True);records=readcsv(BASELINE/'width_bins.csv');effects=readcsv(BASELINE/'width_effects.csv');thresholds=[]
 for family in ['gemini','nemotron']:
  for w,k in WIDTHS.items():
   path=RESULTS/'main_sae_activation_statistics_zero_median/per_feature'/f'{family}_m{w}_k{k}.npz'
   with np.load(_paper_location(path)) as z:
    t=float(z['thresholds'][1]);prov=json.loads(str(z['provenance']));assert t==prov['calibration']['median_positive'];rows=int(z['rows'])
    with np.load(_paper_location(BASELINE/'per_feature'/f'width_{family}_m{w}_k{k}.npz')) as old:assert np.array_equal(z['counts'],old['counts'])
   thresholds.append(dict(family=family,width=w,top_k=k,full_corpus_rows=rows,median_positive=t,figure10_source=str(path),figure10_source_sha256=digest(path)))
 writecsv(OUT/'width_thresholds.csv',thresholds);writecsv(OUT/'width_effects.csv',effects);writecsv(OUT/'width_bins.csv',records)
 colors=plt.get_cmap('viridis')(np.linspace(.05,.95,len(WIDTHS)))
 fig,axes=plt.subplots(2,2,figsize=(12,8),sharey=True)
 for i,family in enumerate(['gemini','nemotron']):
  for j,mode in enumerate(MODES):
   ax=axes[i,j]
   for color,w in zip(colors,WIDTHS):
    r=[x for x in records if x['family']==family and int(x['width'])==w and x['threshold']==mode and x['membership']=='witness']
    ax.plot([float(x['prevalence_median']) for x in r],[float(x['selection_rate']) for x in r],'.-',color=color,label=f'{w:,}')
   style(ax);ax.set_title(f'{family.capitalize()}: {TITLES[j]}')
   if j==0:ax.set_ylabel('Fraction in persistent direction matching')
 handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=8,title='SAE width');fig.tight_layout(rect=(0,.10,1,1));savefig(fig,'width_prevalence_comparison')
 fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
 for ax,family in zip(axes,['gemini','nemotron']):
  for color,w in zip(colors,WIDTHS):
   r=[x for x in records if x['family']==family and int(x['width'])==w and x['threshold']=='median_positive' and x['membership']=='witness'];ax.plot([float(x['prevalence_median']) for x in r],[float(x['selection_rate']) for x in r],'.-',color=color,label=f'{w:,}')
  style(ax);ax.set_title(family.capitalize());ax.set_xlabel('Prevalence above SAE pooled positive median')
 axes[0].set_ylabel('Fraction in persistent direction matching');axes[1].legend(fontsize=8,ncol=2,title='Width');fig.tight_layout();savefig(fig,'width_prevalence_median')
 lines=['# Width persistence with the Figure 10 median threshold','',
 'Complete full-corpus results using the exact pooled-positive median and strict greater-than comparison from Figure 10. Direction matching is held fixed at signed cosine ≥ 0.7. The 131K SAE supplies a target but has no persistence outcome itself because there is no larger dictionary. An AUC of 0.5 indicates no prevalence rank separation between selected and unselected features.','']
 rows=[]
 for family in ['gemini','nemotron']:
  for w in WIDTHS:
   rr=[r for r in effects if r['family']==family and int(r['width'])==w and r['membership']=='witness']
   z=next(r for r in rr if r['threshold']=='zero' and r['population']=='all');m=next(r for r in rr if r['threshold']=='median_positive' and r['population']=='all');active=next(r for r in rr if r['threshold']=='median_positive' and r['population']=='positive_prevalence')
   rows.append([family,w,f3(z['auc']),f3(m['auc']),f3(active['auc'])])
 lines+=table(['Model','Width','AUC: >0','AUC: >median','AUC: >median, excluding zero-prevalence features'],rows)
 lines+=['','![Zero versus median threshold](width_prevalence_comparison.png)','',
 'The larger-width associations remain after excluding features that never exceed the threshold. The smallest widths remain exceptions: Gemini 512 is near chance and Nemotron 512 is reversed. Threshold values and their exact Figure 10 source files are in `width_thresholds.csv`; detailed effects and populations are in `width_effects.csv`.','']
 (OUT/'WIDTH_RESULTS.md').write_text('\n'.join(lines)+'\n')
 refresh_width_readme(effects)
 print('Figure-10 width thresholds verified; width plots ready.',flush=True)
 return effects,thresholds

def full():
 width_effects,width_thresholds=width();(OUT/'per_feature').mkdir(exist_ok=True);effects=[];curve_rows=[];thresholds=[];sources={};positive_rates={};median_rates={}
 for family in ['gemini','nemotron']:
  positive_rates[family]={s:{} for s in SPLITS};median_rates[family]={s:{} for s in SPLITS}
  for split in SPLITS:
   folder=OUT/'counts'/f'{family}_{split}';complete=json.loads((folder/'COMPLETE.json').read_text());assert complete['complete'];assert digest(folder/'counts.npz')==complete['counts_sha256']
   with np.load(_paper_location(folder/'counts.npz')) as z:
    assert z['evaluation_splits'].tolist()==SPLITS;assert np.array_equal(z['rates'],z['counts']/z['rows'][:,None]);assert np.all(z['counts']<=z['positive_counts']);t=float(z['threshold']);full_rows=int(z['rows'][0]);prov=json.loads(str(z['provenance']))
    assert prov['calibration']['median_rank_bracket_verified'] and t==prov['calibration']['median_positive'];assert prov['source_payloads_verified'] and prov['zero_counts_match_all_existing_split_moments']
    for j,e in enumerate(SPLITS):positive_rates[family][e][split]=z['positive_counts'][j]/z['rows'][j];median_rates[family][e][split]=z['rates'][j].copy()
    assert np.array_equal(z['dataset_counts'].sum(0),z['counts'][0]);assert np.array_equal(z['dataset_positive_counts'].sum(0),z['positive_counts'][0])
   thresholds.append(dict(family=family,source_sae=split,median_positive=t,full_corpus_rows=full_rows,positive_events=complete['positive_events'],events_above_median=complete['events_above_median'],fraction_positive_above_median=complete['events_above_median']/complete['positive_events']))
   sources[str(folder/'counts.npz')]=digest(folder/'counts.npz')
  for a,b in itertools.combinations(SPLITS,2):
   match=RESULTS/'split_matches/signed_maximum_cardinality_pca_absolute_0.7/pairs'/f'{family}_sae_{a}_to_{b}';g=load_npz(_paper_location(match/'graph.npz'));sources[str(match/'graph.npz')]=digest(match/'graph.npz')
   with np.load(_paper_location(match/'assignment.npz')) as z:src=z['source'];dst=z['destination']
   certify_maximum(g,src,dst);sources[str(match/'assignment.npz')]=digest(match/'assignment.npz')
   for source,target,ids,graph in [(a,b,src,g),(b,a,dst,g.T.tocsr())]:
    selected=np.zeros(16384,bool);selected[ids]=True;eligible=np.diff(graph.indptr)>0
    with np.load(_paper_location(BASELINE/'per_feature'/f'split_{family}_{source}_to_{target}.npz')) as previous:assert np.array_equal(selected,previous['selected']) and np.array_equal(eligible,previous['eligible'])
    for mode,rates in [('zero',positive_rates),('median_positive',median_rates)]:
     own=rates[family][source][source];other=rates[family][target][source];both=np.minimum(own,other)
     if mode=='zero':
      with np.load(_paper_location(BASELINE/'per_feature'/f'split_{family}_{source}_to_{target}.npz')) as previous:assert np.array_equal(own,previous['own']) and np.array_equal(other,previous['other'])
     np.savez_compressed(_paper_location(OUT/'per_feature'/f'{family}_{source}_to_{target}_{mode}.npz'),own=own,other=other,both=both,selected=selected,eligible=eligible)
     context=dict(family=family,source_split=source,target_split=target,threshold=mode,disjoint_pair={a,b} in ({'wikipedia','no_wikipedia'},{'random1','random2'}))
     for membership,y in [('witness',selected),('eligible_neighbor',eligible)]:
      for metric,values in [('own',own),('other',other),('minimum_both',both)]:
       c=dict(**context,membership=membership,prevalence=metric)
       for population,mask in [('all',np.ones(16384,bool)),('positive_both',both>0)]:effects.append(dict(**c,population=population,other_auc_within_own_deciles=conditional_auc(own[mask],other[mask],y[mask]) if mask.any() else None,**effect(values,y,mask)))
       curve_rows+=curves(values,y,c)
 writecsv(OUT/'split_thresholds.csv',thresholds);writecsv(OUT/'cross_distribution_effects.csv',effects);writecsv(OUT/'cross_distribution_bins.csv',curve_rows)
 # Full equality with previous zero-threshold effects ensures outcome/mask continuity.
 old=readcsv(BASELINE/'cross_distribution_effects.csv');key=lambda r:tuple(str(r[k]) for k in ['family','source_split','target_split','membership','prevalence','population']);index={key(r):r for r in old}
 for r in effects:
  if r['threshold']=='zero':
   before=index[key(r)]
   for k in ['features','selected','median_selected','median_unselected','median_ratio','auc','other_auc_within_own_deciles']:
    if r[k] is None:assert before[k]==''
    else:assert np.isclose(r[k],float(before[k]),rtol=1e-13,atol=1e-15),(key(r),k)
 contrasts=[('wikipedia','no_wikipedia'),('no_wikipedia','wikipedia'),('random1','random2'),('random2','random1')]
 fig,axes=plt.subplots(2,2,figsize=(12,8),sharey=True)
 for i,family in enumerate(['gemini','nemotron']):
  for j,mode in enumerate(MODES):
   ax=axes[i,j]
   for a,b in contrasts:
    r=[x for x in curve_rows if x['family']==family and x['source_split']==a and x['target_split']==b and x['threshold']==mode and x['membership']=='witness' and x['prevalence']=='minimum_both'];ax.plot([x['prevalence_median'] for x in r],[x['selection_rate'] for x in r],'.-',label=f'{LABELS[a]} → {LABELS[b]}')
   style(ax);ax.set_title(f'{family.capitalize()}: {TITLES[j]}');ax.set_xlabel('Minimum prevalence in both distributions');ax.set_xlim(right=.03)
   if j==0:ax.set_ylabel('Fraction with one-to-one direction match')
 handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4);fig.tight_layout(rect=(0,.06,1,1));savefig(fig,'cross_distribution_prevalence_comparison')
 # Compact effect-size figure, without suppressing zero-prevalence features.
 fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
 for ax,family in zip(axes,['gemini','nemotron']):
  for j,mode in enumerate(MODES):
   rr=[next(r for r in effects if r['family']==family and r['source_split']==a and r['target_split']==b and r['threshold']==mode and r['population']=='all' and r['membership']=='witness' and r['prevalence']=='minimum_both') for a,b in contrasts]
   ax.bar(np.arange(4)+(j-.5)*.36,[r['auc'] for r in rr],width=.36,label=TITLES[j])
  ax.axhline(.5,color='gray',ls=':');ax.set_ylim(0,1);ax.set_title(family.capitalize());ax.set_xticks(range(4),['Wiki→Other','Other→Wiki','Random 1→2','Random 2→1'],rotation=20,ha='right');ax.grid(axis='y',alpha=.2)
 axes[0].set_ylabel('Prevalence AUC for direction matching');axes[1].legend(fontsize=8);fig.tight_layout();savefig(fig,'cross_distribution_auc_comparison')
 lines=['# Prevalence using the Figure 10 median activation threshold','',
 'For each SAE, the threshold is the exact pooled median of all strictly positive post-TopK activations across all features and examples in its full family evaluation corpus. Prevalence is the fraction of examples whose activation is **strictly greater** than that threshold. This reproduces the definition in `latex/sections/appendix.tex`, feature activation rates.','',
 'Main-width analyses reuse and verify the exact Figure 10 thresholds and full-corpus counts. For each split-trained 16K SAE, we compute its own pooled median over the full family corpus, then freeze that single threshold on every evaluation distribution. It is not recalibrated per feature or per evaluation split. This keeps prevalence comparable when measuring the same feature in two distributions. Calibration includes MIRACL, as in Figure 10. The primary disjoint distribution pairs contain the twelve English datasets and exclude MIRACL: Wiki/non-Wiki have 42,166,282 and 20,089,125 examples, and Random 1/2 have 37,336,279 and 24,919,128. They are dataset partitions, not equal-sized row halves. These are descriptive full-corpus analyses, not held-out evaluations.','',
 'The zero-threshold analyses are retained for comparison. Decoder graphs, signed cosine threshold 0.7, matching witnesses, and feature populations are held fixed. “Both” prevalence means the minimum of a source feature’s frequencies in its own and the target distributions, using that same source SAE.','',
 '## Width persistence','',
 'Width effects use the intersection of independently selected full-source pairwise maximum matchings, with fixed feature ordering and SciPy 1.15.3 tie-breaking. The 131K SAE has no persistence outcome because no larger dictionary is available. AUC is the probability that a feature selected in the persistent matching has greater prevalence than an unselected feature (half credit for ties); 0.5 indicates no rank separation.','']
 rows=[]
 for family in ['gemini','nemotron']:
  for w in WIDTHS:
   r=[x for x in width_effects if x['family']==family and int(x['width'])==w and x['membership']=='witness' and x['population']=='all'];z=next(x for x in r if x['threshold']=='zero');m=next(x for x in r if x['threshold']=='median_positive');active=next(x for x in width_effects if x['family']==family and int(x['width'])==w and x['membership']=='witness' and x['threshold']=='median_positive' and x['population']=='positive_prevalence');rows.append([family,w,f3(z['auc']),f3(m['auc']),f3(active['auc']),active['features']])
 lines+=table(['Model','Width','AUC: activation > 0','AUC: activation > pooled median','AUC: >median, positive-prevalence features only','Positive-prevalence features at >median'],rows)
 lines+=['','![Width prevalence comparison](width_prevalence_comparison.png)','',
 '## Cross-distribution matching','',
 'Primary comparisons use the disjoint Wikipedia/non-Wikipedia and random dataset partitions. Both directions are reported because prevalence is measured in the source SAE. The full CSV also contains all other split pairs. “Active in both” restricts to features with nonzero prevalence under the chosen threshold in both distributions.','']
 rows=[]
 for family in ['gemini','nemotron']:
  for a,b in contrasts:
   rr=[r for r in effects if r['family']==family and r['source_split']==a and r['target_split']==b and r['membership']=='witness' and r['prevalence']=='minimum_both'];z=next(r for r in rr if r['threshold']=='zero' and r['population']=='all');m=next(r for r in rr if r['threshold']=='median_positive' and r['population']=='all');active=next(r for r in rr if r['threshold']=='median_positive' and r['population']=='positive_both')
   rows.append([family,f'{LABELS[a]} → {LABELS[b]}',f3(z['auc']),f3(m['auc']),f3(active['auc']),f3(m['other_auc_within_own_deciles']),active['features']])
 lines+=table(['Model','Source → target','AUC: >0','AUC: >median','AUC: >median, active in both','Other-domain AUC within own-prevalence bins: >median','Features active in both'],rows)
 lines+=['','The within-bin comparison groups source features by their own-distribution prevalence, preserving ties, and averages target-distribution-prevalence AUCs with weights equal to the number of selected/unselected feature pairs. All AUCs are descriptive effect sizes; features and split comparisons are dependent. Neighbor-eligibility sensitivity and all denominators are included in the CSV files.','',
 '![Cross-distribution prevalence comparison](cross_distribution_prevalence_comparison.png)','',
 '![Cross-distribution AUC comparison](cross_distribution_auc_comparison.png)','',
 '## Thresholds and validation','']
 lines+=table(['Model','16K SAE training split','Frozen positive median','Positive events above threshold'],[[r['family'],LABELS[r['source_sae']],f"{r['median_positive']:.10g}",f"{100*r['fraction_positive_above_median']:.6f}%"] for r in thresholds])
 lines+=['','A small sample locates a bracket; the final threshold is an exact full-corpus order statistic with a checked global median rank. Every cached activation-value and feature-index payload is checked against its SHA-256 digest. Zero-threshold counts agree exactly with existing moments on all five evaluation distributions. Main-16K counts above the median agree exactly with the original Figure 10 counts.','',
 'Plots include all quantile bins, including bins with median prevalence zero, using a small linear segment near zero and logarithmic spacing above it. Quantile bins preserve ties and can contain unequal numbers of features. Full per-feature counts, dataset aggregates, thresholds, calibration diagnostics, checkpoint provenance and script hashes are under `counts/` and `per_feature/`.','',
 'From the source directory, run `python paper.py script --config /path/to/resources.json -- scripts/review_controls/median_counts.py --task INDEX` for every INDEX from 0 through 9, then `python paper.py run prevalence-association --config /path/to/resources.json`. To render only the width figures, use `python paper.py script --config /path/to/resources.json -- scripts/review_controls/median_report.py --width-only`.','']
 (OUT/'README.md').write_text('\n'.join(lines)+'\n');sources[str(_paper_path(__file__))]=digest(__file__);sources[str(ROOT/'scripts/review_controls/prevalence.py')]=digest(ROOT/'scripts/review_controls/prevalence.py')
 (OUT/'provenance.json').write_text(json.dumps(sources,indent=2)+'\n');(OUT/'COMPLETE.json').write_text(json.dumps(dict(complete=True,split_saes=10,cross_distribution_effect_rows=len(effects),previous_zero_effects_reproduced=True,main_thresholds_and_counts_match_figure10=True,all_raw_activation_payloads_verified=True,matching_witnesses_validated=True),indent=2)+'\n')
 print('Median-threshold comparison complete:',OUT,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--width-only',action='store_true');args=p.parse_args();width() if args.width_only else full()
