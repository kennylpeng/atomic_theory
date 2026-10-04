"""Shared compact typography for the matched food and music figures."""
from project_paths import resource_path as _paper_path, resource_location as _paper_location
from pathlib import Path
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def render(groups, root, root_id, output_stem):
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42})
    width, height = 20, 7.0
    fig = plt.figure(figsize=(width,height), facecolor='white')
    ax = fig.add_axes([0,0,1,1])
    ax.set(xlim=(0,width),ylim=(0,height)); ax.axis('off')
    left, span = .24, width-.48
    step = span/len(groups)
    centers = [left+(i+.5)*step for i in range(len(groups))]
    middle = sum(centers)/len(centers)
    ax.text(middle,6.72,root,fontsize=30,ha='center',va='center')
    ax.text(middle,6.40,f'f{root_id}',fontsize=11,ha='center',va='center',color='#555')
    ax.plot([middle,middle],[6.27,6.08],color='black',lw=1)
    ax.plot([centers[0],centers[-1]],[6.08,6.08],color='black',lw=1)
    for i,(p,name,children) in enumerate(groups):
        cx=centers[i]
        ax.plot([cx,cx],[6.08,5.89],color='black',lw=1)
        ax.text(cx,5.69,name,fontsize=19,ha='center',va='center')
        ax.text(cx,5.39,f'f{p}',fontsize=10.5,ha='center',va='center',color='#555')
        stem=left+i*step+.05
        text_x=stem+.23
        ys=[4.91-j*.865 for j in range(len(children))]
        ax.plot([cx,cx,stem,stem],[5.26,5.13,5.13,ys[-1]-.09],color='black',lw=1)
        for (f,label),top in zip(children,ys):
            wrapped=textwrap.fill(label,22,break_long_words=False,break_on_hyphens=False)
            lines=wrapped.count('\n')+1
            ax.plot([stem,text_x-.07],[top-.09,top-.09],color='black',lw=1)
            ax.text(text_x,top,wrapped,fontsize=18,ha='left',va='top',linespacing=1.12)
            id_y=top-lines*.28-.075
            ax.text(text_x,id_y,f'f{f}',fontsize=10.5,ha='left',va='top',color='#555')
    node_count=1+sum(1+len(cs) for _,_,cs in groups)
    ax.text(.24,.16,f'Gemini 131K SAE · {node_count} selected features · Lines are retained parent–child relationships; labels summarize sampled texts.',fontsize=9,color='#555',va='center')
    for ext in ['png','svg','pdf']:
        fig.savefig(_paper_path(str(output_stem)+'.'+ext),dpi=190,facecolor='white',bbox_inches='tight',pad_inches=.08)
    plt.close(fig)

if __name__=='__main__':
    import json
    out=(_paper_path(__file__).resolve().parents[2] / "full_experiments/results/gemini_131k_hierarchical_families")
    for source,root,rid,stem in [('food_illustration_selection.json','Food',10077,'food_hierarchy_illustration'),('music_medium_selection.json','Music',33809,'music_hierarchy_medium')]:
        render(json.loads((out/source).read_text()),root,rid,out/stem)
    print('Rendered both figures with 18-point leaf labels, 19-point branch labels, and feature ID labels.')
