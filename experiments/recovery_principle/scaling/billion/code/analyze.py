"""Refresh partial results without treating unfinished runs as zero recovery."""
from project_paths import resource_path as _paper_path, resource_location as _paper_location
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT


def main():
    configs=json.loads((ROOT/'configs/sweep.json').read_text());rows=[]
    for c in configs:
        for n in c['budgets']:
            rd=ROOT/'results'/c['name']/f'n{n}'
            if not (rd/'summary.json').exists():continue
            s=json.loads((rd/'summary.json').read_text())
            with np.load(_paper_location(rd/'recovery.npz')) as r:
                support=r['test_positives'];p=s['activation_prefix']
                prefix_support=support[:p*(3 if c['data']['kind']=='hierarchical' else 1)]
                rows.append(dict(name=c['name'],kind=c['data']['kind'],alpha=c['data']['alpha'],width=c['width'],k=c['top_k'],seed=c['seed'],examples=n,outside_theorem=c['data']['alpha']<=1,
                    **{key:s[key] for key in ['geometric_prefix','activation_prefix','geometric_count','activation_count']},
                    prefix_min_test_positives=int(prefix_support.min()) if len(prefix_support) else None,
                    prefix_features_under_20_positives=int((prefix_support<20).sum()),test_features_under_20_positives=int((support<20).sum())))
    df=pd.DataFrame(rows);df.to_csv(ROOT/'results/checkpoints.csv',index=False)
    complete=sum((ROOT/'results'/c['name']/'complete.json').exists() for c in configs)
    report=f'# Recovery sweep through 1,024M examples\n\n{complete}/468 trajectories complete; {len(rows)}/5,148 milestones evaluated.\n\nGenerating K=8 throughout. Alpha=0.8 is outside the theorem. All means below and plots use available seeds; consult checkpoints.csv for coverage. Missing runs are never scored as zero.\n'
    if len(df):
        final=df[df.examples==1024000000]
        if len(final):report+='\n## Final-budget means\n\n'+final.groupby(['kind','alpha','width','k'])[['geometric_prefix','activation_prefix','geometric_count','activation_count']].agg(['mean','count']).to_markdown()+'\n'
        if len(final)==468:
            from plot_width import plot_width
            plot_width(df)
            report+='\n## Prefix recovery versus width\n\n[Fixed SAE k=8, curves by alpha](plots/width_prefix_alpha_sweep_1024M.pdf) · [Fixed alpha=1.6, curves by SAE k](plots/width_prefix_k_sweep_1024M.pdf). Both figures show flat/hierarchical geometric and activation prefixes at 1,024M, using three-seed means and ranges. Generating K=8 stays fixed; width uses a log scale.\n\nProportion versions: [curves by alpha](plots/width_prefix_alpha_sweep_1024M_proportion.pdf) · [curves by SAE k](plots/width_prefix_k_sweep_1024M_proportion.pdf). Prefix lengths are divided by SAE width at each point (atoms per column for both distributions, with hierarchical family prefixes multiplied by 3 first); 10% prefix misses are still permitted, and ratios are not capped at 1.\n'
            report+='\n## Recovery counts versus width\n\nTotal recovered atom counts at all ranks: [alpha sweep](plots/width_count_alpha_sweep_1024M.pdf) · [k sweep](plots/width_count_k_sweep_1024M.pdf). Divided by SAE width: [alpha sweep](plots/width_count_alpha_sweep_1024M_proportion.pdf) · [k sweep](plots/width_count_k_sweep_1024M_proportion.pdf). Hierarchical counts include three atoms per recovered complete family and exclude partial families. No prefix or miss allowance applies.\n'
        for kind in ['flat','hierarchical']:
            for sweep in ['alpha','k']:
                sub=df[(df.kind==kind)&((df.k==8) if sweep=='alpha' else (df.alpha==1.6))]
                if not len(sub):continue
                for metric in ['geometric_prefix','activation_prefix','geometric_count','activation_count']:
                    fig,axes=plt.subplots(2,3,figsize=(14,8),sharex=True)
                    for ax,w in zip(axes.flat,[128,256,512,1024,2048,4096 if kind=='flat' else 3072]):
                        for value,g in sub[sub.width==w].groupby(sweep):
                            stats=g.groupby('examples')[metric].agg(['mean','min','max'])
                            x=stats.index.to_numpy()/1e6
                            ax.plot(x,stats['mean'],marker='.',label=f'{sweep}={value}')
                            ax.fill_between(x,stats['min'],stats['max'],alpha=.12)
                        ax.set(xscale='log',title=f'width {w}',xlabel='Training examples (millions)',ylabel=metric.replace('_',' '))
                        ax.grid(alpha=.2)
                    axes.flat[0].legend(fontsize=7)
                    fig.suptitle(f'{kind}: {sweep} sweep; available-seed mean and range')
                    fig.tight_layout()
                    for ext in ['png','pdf']:fig.savefig(ROOT/'plots'/f'{kind}_{sweep}_{metric}.{ext}',dpi=160)
                    plt.close(fig)
    (ROOT/'RESULTS.md').write_text(report)
    print(complete,'complete trajectories;',len(rows),'evaluated checkpoints')

if __name__=='__main__':main()
