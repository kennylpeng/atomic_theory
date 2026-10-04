#!/usr/bin/env python
"""Render the paper's two split heatmaps with only SAE and PCA columns.

Reads existing numerical results without recomputing experiments or modifying
any original figures. Both criteria use absolute cosine for PCA and signed
cosine for SAE. Recompute exclusive PCA inputs with split_16384_match_heatmaps.py
--representations pca --t1 .7 --t2 .7 --device cpu
--out-dir full_experiments/results/split_matches/exclusive_pca_absolute_0.7/matches
--plot-dir full_experiments/results/split_matches/exclusive_pca_absolute_0.7/plots.
"""
from project_paths import resource_path as _paper_path, resource_location as _paper_location
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = _paper_path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scripts.plot_style import apply_plot_style

BASE = ROOT / 'full_experiments/results/split_matches'
OUT = BASE / 'paper_sae_pca_0.7'
SPLITS = ('main', 'wikipedia', 'no_wikipedia', 'random1', 'random2')
PAPER_SPLIT_LABELS = ('full', 'wiki', 'no_wiki', 'rand1', 'rand2')
MODELS = ('gemini', 'nemotron')
REPS = ('sae', 'pca')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matrices(rows):
    indexed = {(r['model'], r['representation'], r['source_split'], r['comparison_split']): r for r in rows}
    assert len(indexed) == len(rows) == 80
    data, widths = {}, {}
    for model in MODELS:
        for rep in REPS:
            key = model, rep
            matrix = np.zeros((5, 5), dtype=np.int64)
            for i, left in enumerate(SPLITS):
                for j, right in enumerate(SPLITS):
                    if i == j:
                        continue
                    row = indexed[model, rep, left, right]
                    width = int(row['source_features'])
                    assert key not in widths or widths[key] == width
                    widths[key] = width
                    matrix[i,j] = int(row['matches'])
                    assert np.isclose(float(row['proportion']), matrix[i,j]/width)
            assert np.array_equal(matrix, matrix.T)
            assert np.all((matrix >= 0) & (matrix <= widths[key]))
            data[key] = matrix
    return data, widths


def plot(data, widths, path, color_max, *, single_row=False):
    apply_plot_style()
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})
    if single_row:
        fig, axes = plt.subplots(1, 4, figsize=(19., 5.3), sharex=True, sharey=True)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(11., 10.), sharex=True, sharey=True)
    split_labels = PAPER_SPLIT_LABELS if single_row else SPLITS
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('#f2f2f2')
    for i, model in enumerate(MODELS):
        for j, rep in enumerate(REPS):
            ax = axes[j * len(MODELS) + i] if single_row else axes[i,j]
            values = data[model, rep] / widths[model, rep]
            image = ax.imshow(np.ma.array(values, mask=np.eye(5, dtype=bool)), cmap=cmap, vmin=0, vmax=color_max)
            if single_row:
                ax.set_title(f'{rep.upper()} ({model})', fontweight='normal', fontsize=18)
            elif i == 0:
                ax.set_title(rep.upper(), fontweight='bold', fontsize=18)
            ax.set_xticks(range(5))
            ax.set_yticks(range(5))
            ax.tick_params(axis='both', labelsize=18)
            if single_row or i == 1:
                ax.set_xticklabels(split_labels, rotation=35, ha='right')
            else:
                ax.tick_params(axis='x', labelbottom=False)
            if j == 0 and (not single_row or i == 0):
                ax.set_yticklabels(split_labels)
            else:
                ax.tick_params(axis='y', labelleft=False)
            for a in range(5):
                for b in range(5):
                    if a == b:
                        continue
                    value = values[a,b]
                    r,g,blue,_ = image.cmap(image.norm(value))
                    color = 'black' if .2126*r + .7152*g + .0722*blue > .55 else 'white'
                    ax.text(b,a,f'{value:.2f}',ha='center',va='center',color=color,fontsize=18)
    if single_row:
        fig.subplots_adjust(left=.11,right=.94,bottom=.32,top=.88,wspace=.08)
    else:
        fig.text(.025,.70,'Gemini',rotation=90,va='center',ha='center',fontsize=18,fontweight='bold')
        fig.text(.025,.29,'Nemotron',rotation=90,va='center',ha='center',fontsize=18,fontweight='bold')
        fig.subplots_adjust(left=.23,right=.88,bottom=.18,top=.92,wspace=.06,hspace=.05)
    colorbar = fig.colorbar(image,ax=axes,fraction=.015 if single_row else .035,
                           pad=.02 if single_row else .035)
    colorbar.set_label('matching feature proportion',fontsize=18)
    colorbar.ax.tick_params(labelsize=18)
    for ext in ('pdf','png'):
        save_options = dict(bbox_inches='tight', pad_inches=.02) if single_row else {}
        fig.savefig(path.with_suffix('.'+ext),dpi=220, **save_options)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cardinality = BASE / 'signed_maximum_cardinality_pca_absolute_0.7'
    exclusive = BASE / 't1_0.7_t2_0.7/matches'
    exclusive_inputs = {
        'sae': exclusive/'split_sae_match_counts.csv',
        'pca': BASE/'exclusive_pca_absolute_0.7/matches/split_pca_match_counts.csv',
    }
    inputs = [cardinality/'counts.csv',cardinality/'metadata.json', *exclusive_inputs.values()]
    hashes = {str(p.relative_to(ROOT)):sha256(p) for p in inputs}
    maximum_rows = [r for r in csv.DictReader((cardinality/'counts.csv').open()) if r['representation'] in REPS]
    exclusive_rows = []
    for rep in REPS:
        for row in csv.DictReader(exclusive_inputs[rep].open()):
            if rep == 'pca':
                assert row['similarity'] == 'absolute cosine'
            assert float(row['t1']) == float(row['t2']) == .7
            width = (3072 if row['model']=='gemini' else 4096) if rep=='pca' else 16384
            exclusive_rows.append(dict(model=row['model'],representation=rep,source_split=row['left_split'],
                comparison_split=row['right_split'],matches=int(row['matches']),source_features=width,
                proportion=int(row['matches'])/width))
    metadata = dict(representations=list(REPS),source_sha256=hashes,figures={})
    for name,rows in (('maximum_cardinality',maximum_rows),('exclusive',exclusive_rows)):
        data,widths = matrices(rows)
        color_max = (json.loads((cardinality/'metadata.json').read_text())['color_max'] if name=='maximum_cardinality'
                     else max(float(m.max()/widths[key]) for key,m in data.items()))
        plot(data,widths,OUT/f'{name}_proportions',color_max,
             single_row=name=='maximum_cardinality')
        with (OUT/f'{name}_counts.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        metadata['figures'][name]=dict(color_min=0,color_max=color_max,rows=len(rows),
            similarity=dict(sae='signed cosine',pca='absolute cosine'),
            threshold=.7,comparison='>=' if name=='maximum_cardinality' else '>')
        print(f'{name}: validated {len(rows)} unchanged SAE/PCA entries; PDF and PNG rendered',flush=True)
    assert hashes == {str(p.relative_to(ROOT)):sha256(p) for p in inputs}
    (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__':
    main()
