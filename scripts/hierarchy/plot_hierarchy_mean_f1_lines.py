#!/usr/bin/env python3
"""Figure 2(c): three domain panels of pooled mean held-out F1, plus fixed k=128."""
from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from export_hierarchy_regret_latex import LEVEL_SPECS, WIDTHS, MODELS
from hierarchy_probe_config import HIERARCHY_DATA_DIR

ROOT = _paper_path(__file__).resolve().parents[2]
OUT = ROOT / 'full_experiments/plots/hierarchy_mean_f1_lines'
RESULTS = ROOT / 'full_experiments/results/hierarchy_recovery_k128'
FIXED_WIDTHS = [512, 4096, 32768, 65536, 131072]
SCHEDULE = dict(zip(WIDTHS, [32,32,32,32,64,64,64,128,128]))
EXPECTED_COUNTS = [52, 432, 12, 174, 8, 24, 66, 262, 552, 86]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f, delimiter='\t'))


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def inputs(variant):
    widths = list(WIDTHS) if variant == 'original' else FIXED_WIDTHS
    rows, sources = [], []
    datasets = list(dict.fromkeys(s.dataset for s in LEVEL_SPECS))
    for dataset in datasets:
        for model in MODELS:
            for width in widths:
                k = SCHEDULE[width] if variant == 'original' else 128
                name = f'{model}_m{width}_k{k}'
                base = HIERARCHY_DATA_DIR / 'probe_results'
                if variant == 'k128' and width < 65536:
                    base = (ROOT / 'full_experiments/results/prefix_recovery_k128') if dataset == 'wordfreq' else RESULTS
                    assert (base / dataset / name / 'checkpoint.json').exists(), f'Incomplete probe: {dataset}/{name}'
                path = base / dataset / name / 'summary.tsv'
                table = read(path)
                assert table and all(r['dataset'] == dataset and r['model'] == model and
                    int(r['width']) == width and int(r['top_k']) == k for r in table), path
                rows.extend(table)
                sources.append(dict(path=str(path), sha256=digest(path), rows=len(table),
                                    reused=variant == 'original' or dataset == 'wordfreq' or width >= 65536))
    return widths, rows, sources


def summarize(widths, rows):
    trajectories = defaultdict(dict)
    for row in rows:
        key = row['dataset'], row['model'], row['category_id']
        width = int(row['width'])
        assert width not in trajectories[key], (key, width)
        trajectories[key][width] = row
        for split in ['train', 'test']:
            tp, fp, fn = [int(row[f'{split}_{k}']) for k in ['tp','fp','fn']]
            assert abs(float(row[f'{split}_f1']) - 2*tp/(2*tp+fp+fn)) < 1e-12
            assert tp+fn == int(row[f'{split}_pos'])
    for key, trajectory in trajectories.items():
        assert set(trajectory) == set(widths), key
        assert len({tuple(r[k] for k in ['positive_count','train_pos','test_pos']) for r in trajectory.values()}) == 1, key
    output = []
    for spec, expected_count in zip(LEVEL_SPECS, EXPECTED_COUNTS):
        for width in widths:
            selected = [r for r in rows if r['dataset'] == spec.dataset and r['level'] == spec.level and int(r['width']) == width]
            assert len(selected) == expected_count, (spec, width, len(selected), expected_count)
            assert {r['model'] for r in selected} == set(MODELS)
            output.append(dict(domain=spec.domain, level=spec.level, label=spec.label, width=width,
                               category_model_count=len(selected), mean_test_f1=float(np.mean([float(r['test_f1']) for r in selected]))))
    return output


def verify_original_table(means):
    table = ROOT / 'full_experiments/plots/hierarchy_mean_f1_by_level.tex'
    lines = [line for line in table.read_text().splitlines() if line.startswith(('Prefix &','Cities &','GBIF &'))]
    assert len(lines) == len(LEVEL_SPECS)
    for line, spec in zip(lines, LEVEL_SPECS):
        values = re.findall(r'0\.\d{5}', line)
        assert len(values) == len(WIDTHS)
        for width, value in zip(WIDTHS, values):
            mean = next(r['mean_test_f1'] for r in means if r['domain'] == spec.domain and r['level'] == spec.level and r['width'] == width)
            assert f'{mean:.5f}' == value, (spec, width, mean, value)
    return dict(path=str(table), sha256=digest(table), exact_rounded_cells=90)


def draw(variant, widths, means):
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 15,
                        'legend.fontsize': 10.5, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(11.7, 3.9), sharex=True, sharey=True, layout='constrained')
    cmap = plt.get_cmap('viridis')
    for ax, domain in zip(axes, ['Prefix','Cities','GBIF']):
        specs = [s for s in LEVEL_SPECS if s.domain == domain]
        colors = [cmap(.2), cmap(.8)] if len(specs) == 2 else [cmap(i / 5) for i in range(6)]
        for spec, color in zip(specs, colors):
            vals = [next(r['mean_test_f1'] for r in means if r['domain'] == domain and r['level'] == spec.level and r['width'] == w) for w in widths]
            ax.plot(widths, vals, color=color, marker='o', markersize=4.2, lw=2, label=spec.label)
        ax.set_title(domain)
        ax.set_xscale('log', base=2)
        ax.set_xlim(min(widths)/2**.25, max(widths)*2**.25)
        ax.set_ylim(0, 1)
        ax.set_yticks(np.linspace(0,1,6))
        ticks = [512,2048,8192,32768,131072] if variant == 'original' else widths
        ax.set_xticks(ticks, [f'{x:,}' for x in ticks], rotation=40, ha='right', fontsize=10)
        ax.set_xlabel('SAE width')
        ax.grid(axis='y', alpha=.25)
        ax.spines[['top','right']].set_visible(False)
        ax.legend(loc='lower right', frameon=False, ncol=2 if domain == 'GBIF' else 1,
                  columnspacing=.8, handlelength=1.5, handletextpad=.4)
    axes[0].set_ylabel('Mean held-out test F1')
    stem = f'hierarchy_mean_f1_{variant}'
    if variant == 'k128':
        fig.suptitle('Fixed k=128', fontsize=15)
    for suffix in ['pdf','png']:
        fig.savefig(OUT / f'{stem}.{suffix}', dpi=220, bbox_inches='tight')
    plt.close(fig)


def run(variant):
    widths, rows, sources = inputs(variant)
    means = summarize(widths, rows)
    validation = verify_original_table(means) if variant == 'original' else None
    if variant == 'k128':
        baseline_rows = read(OUT / 'original_source_rows.tsv')
        baseline_by_id = {(r['dataset'],r['model'],r['category_id'],int(r['width'])):r for r in baseline_rows}
        for row in rows:
            baseline = baseline_by_id[(row['dataset'],row['model'],row['category_id'],int(row['width']))]
            assert all(row[k] == baseline[k] for k in ['positive_count','train_pos','test_pos'])
        assert (RESULTS / 'validation/COMPLETE.json').exists()
        write(RESULTS / 'all_summaries.tsv', rows)
    write(OUT / f'{variant}_source_rows.tsv', rows)
    write(OUT / f'{variant}_mean_f1.tsv', means)
    draw(variant, widths, means)
    meta = dict(variant=variant, widths=widths, models=list(MODELS), sources=sources,
                source_row_count=len(rows), curve_row_count=len(means), table_validation=validation,
                protocol='Unweighted mean held-out test F1 over category-model pairs at each hierarchy level; identical categories and original splits across widths.',
                script_sha256=digest(_paper_path(__file__)))
    (OUT / f'{variant}_provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(f'{variant}: {len(rows)} source rows, {len(means)} mean F1 cells; all checks passed.', flush=True)
    if variant == 'k128':
        (RESULTS / 'COMPLETE.json').write_text(json.dumps(dict(widths=widths,models=list(MODELS),source_row_count=len(rows),curve_row_count=len(means)),indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', choices=['original','k128','both'], default='both')
    args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    for variant in (['original','k128'] if args.variant == 'both' else [args.variant]):
        run(variant)
    (OUT / 'README.md').write_text('''# Hierarchical recovery: three-panel Figure 2(c)

Each figure places Prefix, Cities, and GBIF side by side. Curves show unweighted
mean held-out test F1 over the same category-model pairs as the Figure 2(c)
table, pooling Gemini and Nemotron. Each category has its own feature and activation
threshold selected on train rows and evaluated unchanged on test rows.

`hierarchy_mean_f1_original` uses the original nine-width sparsity schedule.
Its 90 values reproduce the source table to all five stored decimal places.
`hierarchy_mean_f1_k128` holds k=128 across five widths: 512, 4096, 32768, 65536,
and 131072. It uses the additional trained checkpoints at the first three widths,
the completed prefix probes, and reused original k=128 results at the last two widths.
Cities uses the separate country-question and continent-question embeddings.
GBIF retains the existing taxonomy membership and species eligibility rules.
The original and fixed-k figures have the same categories, splits, averaging, colors,
and y-axis limits. No categories are filtered by F1.

Each figure is exported as PNG and PDF, with source TSVs, mean TSVs, and hash provenance.

After preparing the resources, run from the source directory:
```bash
python paper.py script --config /path/to/resources.json -- scripts/hierarchy/plot_hierarchy_mean_f1_lines.py --variant original
python paper.py script --config /path/to/resources.json -- scripts/hierarchy/run_hierarchy_recovery_k128.py --validate
python paper.py run fixed-k-probes --config /path/to/resources.json
python paper.py run hierarchy-k128 --config /path/to/resources.json
```
''')


if __name__ == '__main__':
    main()
