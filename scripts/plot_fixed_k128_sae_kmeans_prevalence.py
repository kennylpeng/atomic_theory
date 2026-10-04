"""Fixed-k=128 counterpart of the SAE/KMeans prevalence appendix figure."""
from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import numpy as np

from scripts import plot_sae_kmeans_prevalence as original
from scripts import train_kmeans_full_corpus as common

ROOT = common.ROOT
SOURCE = ROOT / 'full_experiments/results/fixed_k128_sae_prevalence'
OUT = ROOT / 'full_experiments/plots/sae_kmeans_prevalence_fixed_k128_iter100'
WIDTHS = (512, 4096, 32768, 65536, 131072)
KMEANS_WIDTHS = original.KMEANS_WIDTHS
FAMILIES = ('gemini', 'nemotron')


def render(source=SOURCE, output=OUT):
    curves, records = {}, []
    for family in FAMILIES:
        manifest = common.read(source / family / 'manifest.json')
        rows = manifest['rows']
        for method, widths in (('SAE', WIDTHS), ('KMeans', KMEANS_WIDTHS)):
            for width in widths:
                if method == 'SAE':
                    if width in manifest['compute_widths']:
                        path = source / 'per_feature' / f'{family}_m{width}_k128.npz'
                        done = common.read(source / 'completed' / f'{family}_m{width}_k128.json')
                        assert done['complete'] and common.digest(path) == done['sha256']
                    else:
                        reused = next(r for r in manifest['reused'] if r['width'] == width)
                        path = _paper_path(reused['path'])
                        assert common.digest(path) == reused['sha256']
                    with np.load(_paper_location(path), allow_pickle=False) as z:
                        assert int(z['rows']) == rows
                        column = z['threshold_modes'].tolist().index('median_positive')
                        counts, rates = z['counts'][column], z['rates'][column]
                        threshold = float(z['thresholds'][column])
                        provenance = json.loads(str(z['provenance']))
                        assert provenance['family'] == family and provenance['width'] == width
                        assert provenance['top_k'] == 128 and provenance['calibration']['median_rank_bracket_verified']
                        assert threshold == provenance['calibration']['median_positive'] > 0
                else:
                    path = original.kmeans_prevalence_path(family, width)
                    with np.load(_paper_location(path), allow_pickle=False) as z:
                        assert int(z['evaluation_rows']) == rows
                        counts, rates = z['counts'], z['prevalence']
                        saved_density = z['count_density']
                        np.testing.assert_array_equal(z['log10_grid'], original.GRID)
                    assert int(counts.sum()) == rows
                    threshold = None
                assert counts.dtype == np.int64 and counts.shape == (width,)
                assert np.all((counts >= 0) & (counts <= rows))
                np.testing.assert_array_equal(rates, counts/rows)
                density = original.count_density(rates)
                if method == 'KMeans':
                    np.testing.assert_allclose(density, saved_density, rtol=1e-12, atol=1e-10)
                curves[family, method, width] = density
                records.append(dict(family=family, method=method, width=width, top_k=128 if method == 'SAE' else 1,
                    rows=rows, threshold=threshold, median_prevalence=float(np.median(rates)),
                    zero_rate_features=int((rates == 0).sum()), positive_rate_features=int((rates > 0).sum()),
                    source=str(path), source_sha256=common.digest(path)))
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 18, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    colors = dict(zip(original.WIDTHS, plt.get_cmap('viridis')(np.linspace(.05, .95, len(original.WIDTHS)))))
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10), sharex=True, sharey=True)
    upper = max(float(y.max()) for y in curves.values())*1.12
    root_ticks = MaxNLocator(nbins=5).tick_values(0, np.sqrt(upper))
    ticks = root_ticks[(root_ticks >= 0) & (root_ticks <= np.sqrt(upper))]**2
    for row, family in enumerate(FAMILIES):
        for col, (method, widths) in enumerate((('SAE', WIDTHS), ('KMeans', KMEANS_WIDTHS))):
            ax = axes[row, col]
            for width in widths:
                ax.plot(original.GRID, curves[family, method, width], color=colors[width], linewidth=2.7)
            ax.set_yscale('function', functions=(lambda y: np.sqrt(np.maximum(y, 0)), np.square))
            ax.set_ylim(0, upper)
            ax.set_yticks(ticks)
            ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
            ax.set_xlim(original.GRID[0], original.GRID[-1])
            ax.set_xticks(np.arange(-8, 1, 2))
            title = f'SAE, fixed $k=128$ ({family})' if method == 'SAE' else f'KMeans ({family})'
            ax.set_title(title, fontsize=22, pad=13)
            ax.tick_params(labelsize=17)
            ax.grid(alpha=.18)
            if col == 0:
                ax.set_ylabel('Features per\nlog₁₀ prevalence unit', fontsize=20)
            if row == 1:
                ax.set_xlabel('log₁₀ prevalence', fontsize=20, labelpad=8)
    legend_widths = sorted(set(WIDTHS) | set(KMEANS_WIDTHS))
    fig.legend([Line2D([], [], color=colors[w], linewidth=2.7) for w in legend_widths],
        [f'{w:,}' for w in legend_widths], title='Number of features / clusters',
        loc='lower center', bbox_to_anchor=(.54, .005), ncol=3,
        fontsize=17, title_fontsize=18, frameon=False, columnspacing=1.5)
    fig.subplots_adjust(left=.125, right=.99, bottom=.225, top=.945, hspace=.35, wspace=.13)
    stem = 'sae_kmeans_prevalence_fixed_k128'
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(output / f'{stem}.{ext}', dpi=220, bbox_inches='tight', pad_inches=.08)
    plt.close(fig)
    import csv
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    np.savez_compressed(_paper_location(output / 'density_curves.npz'), log10_prevalence=original.GRID,
                        **{f'{f}_{m}_k{w}': d for (f, m, w), d in curves.items()})
    common.write(output / 'provenance.json', dict(complete=True, top_k=128, sae_widths=list(WIDTHS),
        kmeans_widths=list(KMEANS_WIDTHS), kmeans_iterations=100, inputs=records,
        created_utc=datetime.now(timezone.utc).isoformat(), source_code_sha256=common.digest(__file__),
        original_plot_code_sha256=common.digest(original.__file__),
        definition='SAE: activation strictly above the exact pooled positive median; KMeans: nearest-centroid membership',
        plotting='Count-scaled log10 KDE, bandwidth 0.12, shared axes with square-root y-scale',
        colors='Same width colors as the original nine-width figure; legend is the union of displayed widths'))
    print('Wrote fixed-k=128 comparison:', output, flush=True)
    return records


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=SOURCE)
    p.add_argument('--output', type=Path, default=OUT)
    a = p.parse_args()
    render(a.source, a.output)
