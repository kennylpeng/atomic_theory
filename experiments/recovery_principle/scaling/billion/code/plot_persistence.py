"""Validate completed persistence graphs and plot the two Section D sweeps."""
from project_paths import resource_path as _paper_path, resource_location as _paper_location
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.sparse import load_npz, coo_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching, connected_components
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter

# Cached reporting does not load PyTorch or training code.
import sys
ROOT = _paper_path(__file__).resolve().parents[1]
REPO = ROOT.parents[3]
sys.path.insert(0, str(REPO))
from scripts.stability.persistent_matching import validate_witness
BUDGET = 1_024_000_000
THRESHOLD = 0.7
OUT = ROOT / 'results/persistence_1024M_t0.7'
def code_hash():
    from persistence import code_hash as generation_hash
    return generation_hash()
def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)
from scripts.stability.persistent_matching import independent_assignments



def component_upper_bound(layers,candidates):
    """Sum independent per-component pairwise matching upper bounds."""
    sub=[g[candidates] for g in layers]
    left=[];right=[]
    for g in sub:
        c=g.tocsc()
        for col in np.flatnonzero(np.diff(c.indptr)>1):
            rows=c.indices[c.indptr[col]:c.indptr[col+1]]
            left.extend([int(rows[0])]*(len(rows)-1));right.extend(rows[1:])
    adjacency=coo_matrix((np.ones(len(left)),(left,right)),shape=(len(candidates),len(candidates))).tocsr()
    count,labels=connected_components(adjacency,directed=False)
    upper=0
    for label in range(count):
        rows=np.flatnonzero(labels==label)
        if len(rows)==1:upper+=1
        else:upper+=min(np.count_nonzero(maximum_bipartite_matching(g[rows],perm_type='column')>=0) for g in sub)
    return int(upper)


def load_validate():
    manifest=json.loads((OUT/'manifest.json').read_text())
    all_rows=[];validation=[];stream_hashes={}; missing=[]
    for job in manifest['jobs']:
        p=OUT/job['name']/'summary.json'
        if not p.exists():missing.append(job['name']);continue
        r=json.loads(p.read_text())
        assert r['complete'] and r['exhaustive']
        if 'matching_migration' in r:
            import hashlib
            assert r['matching_migration']['matching_code_sha256'] == hashlib.sha256((REPO/'scripts/stability/persistent_matching.py').read_bytes()).hexdigest()
            assert r['persistence_method'] == 'independent_pairwise_intersection'
        else:
            assert r['code_hash']==code_hash()
        assert r['checkpoints']==job['checkpoints'] and r['training_examples']==BUDGET
        assert r['threshold']==THRESHOLD and r['evaluation_examples']==400_000
        assert r['evaluation_seed']==job['configs'][0]['eval_seed']+1
        key=(job['kind'],job['alpha'],job['seed'])
        assert stream_hashes.setdefault(key,r['evaluation_rows_sha256'])==r['evaluation_rows_sha256']
        widths=[c['width'] for c in job['configs']]
        expected={(metric,width) for metric in ['decoder_cosine','activation_pearson'] for width in widths[:-1]}
        assert {(row['metric'],row['width']) for row in r['rows']}==expected and len(r['rows'])==10
        graphs={}
        for d in r['diagnostics']:
            prefix='cosine' if d['metric']=='decoder_cosine' else 'pearson'
            g=load_npz(_paper_location(p.parent/f'{prefix}_{d["source"]}_to_{d["target"]}.npz'))
            assert g.shape==(d['source'],d['target']) and g.nnz==d['edges']
            graphs[d['metric'],d['source'],d['target']]=g
        assert len(graphs)==30
        for row in r['rows']:
            assert all(row[k]==job[k] for k in ['kind','alpha','k','seed'])
            width=row['width'];metric=row['metric']
            targets=[w for w in widths if w>width]
            assert row['target_widths']==targets
            layers=[graphs[metric,width,t] for t in targets]
            witness=np.load(_paper_location(p.parent/f'{metric}_witness_{width}.npz'))
            selected=witness['source'];destinations=[witness[f'target_{t}'] for t in targets]
            validate_witness(layers,selected,destinations)
            assert len(selected)==row['persistent_count']
            assert row['proportion']==len(selected)/width and 0<=len(selected)<=width
            expected, assignments = independent_assignments(layers)
            np.testing.assert_array_equal(selected, expected)
            for target, assignment, destination in zip(targets, assignments, destinations):
                np.testing.assert_array_equal(destination, assignment[selected])
                if f'assignment_{target}' in witness:
                    np.testing.assert_array_equal(witness[f'assignment_{target}'], assignment)
            validation.append(dict(group=job['name'],metric=metric,width=width,count=len(selected),
                                   pairwise_counts=[int((a>=0).sum()) for a in assignments],
                                   certificate='pairwise_vertex_covers_and_source_intersection'))
            all_rows.append({k:row[k] for k in ['kind','alpha','k','seed','metric','width','persistent_count','proportion']})
    if missing:raise ValueError(f'{len(missing)} incomplete groups: {missing[:8]}')
    assert len(all_rows)==780 and len(stream_hashes)==30
    df=pd.DataFrame(all_rows)
    assert not df.duplicated(['kind','alpha','k','seed','metric','width']).any()
    df.to_csv(OUT/'counts.csv',index=False)
    atomic_json(OUT/'validation.json',dict(complete=True,groups=78,checkpoints=468,
                graphs=2340,persistence_results=len(validation),evaluation_streams=len(stream_hashes),
                results=validation))
    return df


def plots(df):
    plt.rcParams.update({'font.size':16,'axes.labelsize':16,'axes.titlesize':18,
                         'xtick.labelsize':15,'ytick.labelsize':15,
                         'legend.fontsize':15,'figure.titlesize':21,
                         'pdf.fonttype':42,'ps.fonttype':42})
    output=OUT/'plots';output.mkdir(exist_ok=True)
    stats=[]
    sweeps=[('alpha',[.8,1.2,1.6,2.,2.4],df['k']==8),
            ('k',[1,2,4,6,8,10,12,16,32],np.isclose(df['alpha'],1.6))]
    widths=[128,256,512,1024,2048]
    for sweep,values,fixed in sweeps:
        for quantity in ['count','proportion']:
            key='persistent_count' if quantity=='count' else 'proportion'
            fig,axes=plt.subplots(2,2,figsize=(11.8,8.9),sharex=True)
            for row,kind in enumerate(['flat','hierarchical']):
                for col,metric in enumerate(['decoder_cosine','activation_pearson']):
                    ax=axes[row,col]
                    for i,value in enumerate(values):
                        group=df[fixed & (df['kind']==kind) & (df['metric']==metric) & np.isclose(df[sweep],value)]
                        seeds=group.groupby('width')['seed'].apply(set)
                        assert set(seeds.index)==set(widths) and all(s=={0,1,2} for s in seeds)
                        summary=group.groupby('width')[key].agg(['mean','min','max']).reindex(widths)
                        label=(rf'$\alpha={value:g}$' if sweep=='alpha' else rf'$k={value:g}$')
                        color=plt.get_cmap('tab10')(i)
                        ax.plot(widths,summary['mean'],label=label,color=color,marker='o',
                                markersize=5,linewidth=2.1,linestyle='--' if sweep=='alpha' and value<=1 else '-')
                        ax.fill_between(widths,summary['min'].to_numpy(),summary['max'].to_numpy(),color=color,alpha=.10,linewidth=0)
                        for w,record in summary.iterrows():
                            stats.append(dict(sweep=sweep,value=value,kind=kind,metric=metric,quantity=quantity,
                                              width=w,mean=record['mean'],min=record['min'],max=record['max'],seeds=3))
                    ax.set_xscale('log',base=2);ax.set_xticks(widths,['128','256','512','1K','2K'])
                    ax.tick_params(axis='x',labelbottom=True)
                    ax.set_xlabel('Source SAE width')
                    ax.set_ylabel('Persistent features' if quantity=='count' else 'Persistent features (%)')
                    ax.set_title(f'{kind.title()} · '+('Decoder cosine' if col==0 else 'Activation Pearson'))
                    ax.spines[['top','right']].set_visible(False)
                    ax.grid(axis='y',alpha=.23);ax.set_axisbelow(True)
                    ax.set_ylim(0,1 if quantity=='proportion' else 2150)
                    ax.yaxis.set_major_formatter(PercentFormatter(1) if quantity=='proportion' else FuncFormatter(lambda v,p:f'{v:,.0f}'))
            suffix='by distribution' if sweep=='alpha' else 'by sparsity parameter'
            fig.suptitle(f'Simulated feature persistence {suffix}', y=.98)
            h,l=axes[0,0].get_legend_handles_labels()
            fig.legend(h,l,loc='lower center',bbox_to_anchor=(.5,.005),ncol=5 if sweep=='alpha' else 9,frameon=False,
                       columnspacing=1.0,handlelength=1.6)
            fig.tight_layout(rect=(0,.065,1,1),h_pad=1.4,w_pad=1.4)
            stem=output/f'persistence_{sweep}_{quantity}_1024M'
            for ext in ('pdf','png'):fig.savefig(stem.with_suffix('.'+ext),dpi=220)
            plt.close(fig);print(stem.with_suffix('.pdf'))
    pd.DataFrame(stats).to_csv(OUT/'plotted_values.csv',index=False)


def report():
    df=load_validate();plots(df)
    atomic_json(OUT/'summary.json',dict(complete=True,groups=78,checkpoints=468,persistence_results=780,
        threshold=THRESHOLD,training_examples=BUDGET,evaluation_examples=400_000,
        curves='Three-seed mean; shaded minimum–maximum range',
        metrics=['signed decoder cosine','signed continuous post-TopK activation Pearson'],
        method='Exhaustive threshold graphs; intersection of independently selected pairwise maximum matchings into every larger width',
        omitted_source_widths={'flat':4096,'hierarchical':3072},
        plotter_sha256=__import__('hashlib').sha256(_paper_path(__file__).read_bytes()).hexdigest()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plots-only',action='store_true',help='Redraw saved counts without recomputing or validating graphs')
    parser.add_argument('--results-dir',type=Path,help='Saved persistence result directory; requires --plots-only')
    args=parser.parse_args()
    if args.results_dir and not args.plots_only:
        parser.error('--results-dir requires --plots-only')
    if args.plots_only:
        if args.results_dir:OUT=args.results_dir.resolve()
        THRESHOLD=json.loads((OUT/'summary.json').read_text())['threshold']
        plots(pd.read_csv(OUT/'counts.csv'))
    else:
        report()

