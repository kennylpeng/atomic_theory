#!/usr/bin/env python3
"""Compare unique a-prefix probe decoder directions on one signed cosine scale."""
from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

ROOT = _paper_path(__file__).resolve().parents[2]


def load_matrix(path: Path, features: dict[str, list[str]]):
    with path.open(newline='') as handle:
        rows = list(csv.reader(handle))
    columns = rows[0][1:]
    if [row[0] for row in rows[1:]] != columns:
        raise ValueError(f'Row/column order differs in {path}')
    ids = [label.split(' ', 1)[0] for label in columns]
    if len(ids) != len(set(ids)) or set(ids) != set(features):
        raise ValueError(f'Unexpected or duplicate features in {path}')
    matrix = np.array([[float(value) for value in row[1:]] for row in rows[1:]])
    if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T) or not np.allclose(matrix.diagonal(), 1):
        raise ValueError(f'Invalid cosine matrix in {path}')
    labels = ['/'.join(sorted(features[feature])) for feature in ids]
    order = sorted(range(len(labels)), key=lambda i: labels[i])
    return matrix[np.ix_(order, order)], [labels[i] for i in order]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'full_experiments/results/a_prefix_probe_decoder_cosines')
    parser.add_argument('--out-stem', type=Path, default=ROOT / 'full_experiments/plots/a_prefix_probe_decoder_cosines')
    args = parser.parse_args()
    summary = json.loads((args.data_dir / 'summary.json').read_text())
    panels = []
    for result in summary['results']:
        matrix, labels = load_matrix(args.data_dir / f'gemini_{result["width"]}_cosines.csv', result['features'])
        values = matrix[np.triu_indices(len(matrix), k=1)]
        if not np.isclose(np.median(values), result['median_cosine'], atol=1e-10):
            raise ValueError('Median differs from saved analysis')
        panels.append((result, matrix, labels))
    if len(panels) != 2 or panels[0][2] != panels[1][2]:
        raise ValueError('Expected two matrices with matching prefix groups')
    off_diagonal = np.concatenate([m[np.triu_indices(len(m), 1)] for _, m, _ in panels])
    limit = float(np.ceil(np.max(np.abs(off_diagonal)) * 10) / 10)
    norm = Normalize(vmin=-limit, vmax=limit)
    cmap = plt.get_cmap('RdBu_r').copy()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig = plt.figure(figsize=(10.8, 5.1))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.045], left=0.055, right=0.94,
                           bottom=0.13, top=0.90, wspace=0.16)
    for i, (result, matrix, labels) in enumerate(panels):
        ax = fig.add_subplot(grid[i])
        # Self-cosines are 1 and saturate at the shared scale's maximum color.
        im = ax.imshow(matrix, cmap=cmap, norm=norm, interpolation='nearest')
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=10)
        ax.set_yticks(range(len(labels)), labels, fontsize=10)
        ax.tick_params(length=0, pad=5)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f'SAE width {result["width"]:,}', fontsize=12, pad=10)
    bar = fig.colorbar(im, cax=fig.add_subplot(grid[2]))
    bar.set_label('Decoder cosine similarity', fontsize=11, labelpad=10)
    bar.set_ticks(np.linspace(-limit, limit, 5))
    bar.ax.tick_params(labelsize=10)
    args.out_stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(args.out_stem.with_suffix('.' + extension), dpi=220, facecolor='white',
                    bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    args.out_stem.with_suffix('.json').write_text(json.dumps({
        'vmin': -limit, 'vmax': limit, 'colormap': 'RdBu_r', 'diagonal': 'self-cosine 1, saturated at maximum red',
        'prefix_order': panels[0][2], 'off_diagonal_min': float(off_diagonal.min()),
        'off_diagonal_max': float(off_diagonal.max()), 'input_directory': str(args.data_dir),
    }, indent=2) + '\n')
    print(f'Saved {args.out_stem}.png/.pdf/.svg; shared scale [{-limit}, {limit}].')
    print(f'Observed off-diagonal range: [{off_diagonal.min():.6f}, {off_diagonal.max():.6f}].')


if __name__ == '__main__':
    main()
