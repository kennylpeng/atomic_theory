"""Shared count/proportion panels for the paper's stability figures."""
from project_paths import resource_path as _paper_path, resource_location as _paper_location
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter, MultipleLocator
sys.path.insert(0,str(_paper_path(__file__).resolve().parents[2]))
from scripts.plot_style import apply_plot_style


def apply_stability_style():
    """Readable typography when the two-panel plot is embedded in the paper."""
    apply_plot_style()
    plt.rcParams.update({
        'font.size': 18,
        'axes.labelsize': 20,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18,
    })


def apply_stability_grid(ax, *, proportion=False):
    """Count grids every 1,000; proportion grids every ten percentage points."""
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(MultipleLocator(.2 if proportion else 5000))
    ax.yaxis.set_minor_locator(MultipleLocator(.1 if proportion else 1000))
    ax.grid(axis='y', which='major', alpha=.25)
    ax.grid(axis='y', which='minor', linewidth=.5, alpha=.2)


def plot_stability_panels(rows,path,*,count_key='count',persistent=True,separate=False):
    apply_stability_style()
    widths=sorted({r['width'] for r in rows})
    labels=[str(m) if m<1024 else f'{m//1024}K' for m in widths]
    styles={'gemini':dict(color='#0072B2',marker='o',label='Gemini'),
            'nemotron':dict(color='#D55E00',marker='s',label='Nemotron')}
    noun='persistent' if persistent else 'matched'
    def draw(ax,key):
        for model in ('gemini','nemotron'):
            group=sorted([r for r in rows if r['model']==model],key=lambda r:r['width'])
            assert [r['width'] for r in group]==widths
            ax.plot([r['width'] for r in group],[r[key] for r in group],
                    **styles[model],linewidth=2.5,markersize=6)
        ax.set_xscale('log',base=2);ax.set_xticks(widths,labels)
        ax.set_xlabel('SAE width');apply_stability_grid(ax, proportion=key=='proportion');ax.legend(frameon=False)
        ax.spines[['top','right']].set_visible(False)
        if key==count_key:
            ax.set_ylabel(f'{noun.capitalize()} feature count');ax.set_ylim(bottom=0)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:,.0f}'))
        else:
            ax.set_ylabel(f'{noun} feature proportion');ax.set_ylim(0,1)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
    path=_paper_path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(13,5.0),layout='constrained')
    for ax,key in zip(axes,[count_key,'proportion']):draw(ax,key)
    for ext in ['png','pdf']:fig.savefig(path.with_suffix('.'+ext),dpi=220)
    plt.close(fig)
    if separate:
        for key in [count_key,'proportion']:
            fig,ax=plt.subplots(figsize=(6.5,5.0),layout='constrained');draw(ax,key)
            for ext in ['png','pdf']:fig.savefig(path.parent/f'{key}.{ext}',dpi=220)
            plt.close(fig)
