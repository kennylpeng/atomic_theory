"""Render the validated asymmetric split-SAE activation heatmaps."""
from project_paths import resource_path as _paper_path, resource_location as _paper_location
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT=_paper_path(__file__).resolve().parents[2]
OUTPUT=ROOT/'full_experiments/results/split_sae_activation_sketch512_candidates32_top5'
SPLITS=('main','wikipedia','no_wikipedia','random1','random2')
LABELS=('full','wiki','no_wiki','rand1','rand2')


def main():
    rows=list(csv.DictReader((OUTPUT/'counts.csv').open()))
    assert len(rows)==40
    plt.rcParams.update({'font.size':12,'axes.labelsize':13,'axes.titlesize':15,
                         'xtick.labelsize':12,'ytick.labelsize':12,
                         'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(10.6,4.8),layout='constrained')
    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#f2f2f2')
    vmax=min(1.,np.ceil(max(float(r['proportion']) for r in rows)*10)/10)
    for family,ax in zip(('gemini','nemotron'),axes):
        matrix=np.full((5,5),np.nan)
        for r in rows:
            if r['family']==family:
                matrix[SPLITS.index(r['source_split']),SPLITS.index(r['comparison_split'])]=float(r['proportion'])
        im=ax.imshow(np.ma.masked_invalid(matrix),cmap=cmap,vmin=0,vmax=vmax)
        ax.set_xticks(range(5),LABELS,rotation=35,ha='right')
        ax.set_yticks(range(5),LABELS)
        ax.set_title(f'{family.capitalize()} SAE activations',pad=9)
        for i in range(5):
            for j in range(5):
                if i==j:continue
                value=matrix[i,j];red,green,blue,_=cmap(im.norm(value))
                color='black' if .2126*red+.7152*green+.0722*blue>.55 else 'white'
                ax.text(j,i,f'{value:.1%}',ha='center',va='center',fontsize=12,color=color)
    axes[0].set_ylabel('Source SAE and\nevaluation distribution')
    fig.supxlabel('Comparison SAE training distribution',fontsize=13)
    cbar=fig.colorbar(im,ax=axes,fraction=.033,pad=.025,format=PercentFormatter(1))
    cbar.set_label('Matched features\n(Pearson ≥ 0.7)',fontsize=13,labelpad=10)
    for ext in ('png','pdf'):
        fig.savefig(OUTPUT/f'activation_split_heatmaps_t0.7.{ext}',dpi=240,bbox_inches='tight',pad_inches=.06)
    plt.close(fig)
    path=OUTPUT/'summary.json';summary=json.loads(path.read_text())
    summary['plot_script']='scripts/stability/plot_split_activation_heatmaps.py'
    summary['plot_script_sha256']=hashlib.sha256(_paper_path(__file__).read_bytes()).hexdigest()
    path.write_text(json.dumps(summary,indent=2)+'\n')
    print('Rendered PDF and PNG with complete axis labels and shared color scale.')

if __name__=='__main__':main()
