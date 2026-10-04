"""Paper appendix: SAE median-threshold prevalence beside full-corpus KMeans."""
from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import numpy as np
from scipy.integrate import trapezoid
from scipy.stats import gaussian_kde

ROOT = _paper_path(__file__).resolve().parents[1]
SAE = ROOT / 'full_experiments/results/main_sae_activation_statistics_zero_median'
KMEANS = ROOT / 'full_experiments/results/kmeans_raw_full_corpus_iter100/prevalence'
OUT = ROOT / 'full_experiments/plots/sae_kmeans_prevalence_iter100'
WIDTHS = tuple(512 * 2**i for i in range(9))
TOPKS = (32, 32, 32, 32, 64, 64, 64, 128, 128)
KMEANS_512 = ROOT / 'full_experiments/results/kmeans_512_full_corpus_iter100'
# All four widths are part of the current paper. Missing 512-cluster results
# must fail at input validation, rather than silently render an older figure.
KMEANS_WIDTHS = (512, 4096, 16384, 131072)
FAMILIES = ('gemini', 'nemotron')
GRID = np.linspace(-8.5, 0, 600)
BANDWIDTH = .12


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def kmeans_prevalence_path(family, width):
    if width == 512:
        from scripts.kmeans_512_iter100 import validate
        record = validate(KMEANS_512, family)
        return _paper_path(record['prevalence_path'])
    return KMEANS / f'{family}_k{width}_prevalence.npz'


def count_density(rates):
    positive = rates[rates > 0]
    logs = np.log10(positive)
    if len(logs) > 1 and logs.std(ddof=1) > 0:
        kde = gaussian_kde(logs, bw_method=BANDWIDTH / logs.std(ddof=1))
        np.testing.assert_allclose(np.sqrt(kde.covariance[0, 0]), BANDWIDTH)
        density = kde(GRID) * len(logs)
    elif len(logs):
        density = np.exp(-.5*((GRID-logs[0])/BANDWIDTH)**2)*len(logs)/(BANDWIDTH*np.sqrt(2*np.pi))
    else:
        density = np.zeros_like(GRID)
    assert np.isfinite(density).all() and np.all(density >= 0)
    np.testing.assert_allclose(trapezoid(density, GRID), len(logs), rtol=1e-4, atol=1e-10)
    return density


def collect():
    assert json.loads((SAE / 'COMPLETE.json').read_text())['complete']
    assert json.loads((SAE / 'validation.json').read_text())['passed']
    summary = json.loads((KMEANS / 'summary.json').read_text())
    assert summary['complete'] and summary['iterations'] == 100
    curves, records = {}, []
    for family in FAMILIES:
        family_rows = {r['evaluation_rows'] for r in summary['models'] if r['family'] == family}
        assert len(family_rows) == 1
        n = family_rows.pop()
        for method, widths in (('SAE', WIDTHS), ('KMeans', KMEANS_WIDTHS)):
            for width in widths:
                if method == 'SAE':
                    topk = TOPKS[WIDTHS.index(width)]
                    path = SAE / 'per_feature' / f'{family}_m{width}_k{topk}.npz'
                    with np.load(_paper_location(path), allow_pickle=False) as z:
                        assert int(z['rows']) == n
                        modes = z['threshold_modes'].tolist()
                        column = modes.index('median_positive')
                        assert modes[column] == 'median_positive'
                        counts, rates = z['counts'][column], z['rates'][column]
                        threshold = float(z['thresholds'][column])
                        provenance = json.loads(str(z['provenance']))
                        assert provenance['family'] == family and provenance['width'] == width
                        assert provenance['calibration']['median_rank_bracket_verified']
                        assert threshold == provenance['calibration']['median_positive'] > 0
                        assert int(counts.sum()) == int(z['row_l0_statistics'][column, 0])
                else:
                    path = kmeans_prevalence_path(family, width)
                    with np.load(_paper_location(path), allow_pickle=False) as z:
                        assert int(z['evaluation_rows']) == n
                        counts, rates = z['counts'], z['prevalence']
                        assert int(counts.sum()) == n
                        saved_grid, saved_density = z['log10_grid'], z['count_density']
                    threshold = None
                assert counts.dtype == np.int64 and counts.shape == rates.shape == (width,)
                assert np.all((counts >= 0) & (counts <= n))
                np.testing.assert_array_equal(rates, counts/n)
                density = count_density(rates)
                if method == 'KMeans':
                    np.testing.assert_array_equal(saved_grid, GRID)
                    np.testing.assert_allclose(density, saved_density, rtol=1e-12, atol=1e-10)
                curves[family, method, width] = density
                records.append(dict(family=family, method=method, width=width, rows=n,
                    activation_threshold=threshold, positive_rate_features=int((rates > 0).sum()),
                    zero_rate_features=int((rates == 0).sum()), median_prevalence=float(np.median(rates)),
                    density_integral=float(trapezoid(density, GRID)),
                    source=str(path.relative_to(ROOT)), source_sha256=digest(path)))
    return curves, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    curves, records = collect()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 18, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    colors = dict(zip(WIDTHS, plt.get_cmap('viridis')(np.linspace(.05, .95, len(WIDTHS)))))
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10), sharex=True, sharey=True)
    upper = max(float(y.max()) for y in curves.values()) * 1.12
    root_ticks = MaxNLocator(nbins=5).tick_values(0, np.sqrt(upper))
    ticks = root_ticks[(root_ticks >= 0) & (root_ticks <= np.sqrt(upper))]**2
    for row, family in enumerate(FAMILIES):
        for col, (method, widths) in enumerate((('SAE', WIDTHS), ('KMeans', KMEANS_WIDTHS))):
            ax = axes[row, col]
            for width in widths:
                ax.plot(GRID, curves[family, method, width], color=colors[width], linewidth=2.7,
                        label=f'{width:,}')
            ax.set_yscale('function', functions=(lambda y: np.sqrt(np.maximum(y, 0)), np.square))
            ax.set_ylim(0, upper)
            ax.set_yticks(ticks)
            ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
            ax.set_xlim(GRID[0], GRID[-1])
            ax.set_xticks(np.arange(-8, 1, 2))
            ax.set_title(f'{method} ({family})', fontsize=22, pad=13)
            ax.tick_params(labelsize=17)
            ax.grid(alpha=.18)
            if col == 0:
                ax.set_ylabel('Features per\nlog₁₀ prevalence unit', fontsize=20)
            if row == 1:
                ax.set_xlabel('log₁₀ prevalence', fontsize=20, labelpad=8)
    fig.legend(axes[0, 0].lines, [f'{width:,}' for width in WIDTHS],
               title='Number of features / clusters', loc='lower center', bbox_to_anchor=(.54, .005),
               ncol=5, fontsize=17, title_fontsize=18, frameon=False, columnspacing=1.5)
    fig.subplots_adjust(left=.125, right=.99, bottom=.225, top=.945, hspace=.35, wspace=.13)
    stem = 'sae_kmeans_prevalence'
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(args.output / f'{stem}.{ext}', dpi=220, bbox_inches='tight', pad_inches=.08)
    plt.close(fig)
    with (args.output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (args.output / 'density_curves.npz').open('wb') as stream:
        np.savez_compressed(_paper_location(stream), log10_prevalence=GRID,
                            **{f'{f}_{m}_k{w}': d for (f, m, w), d in curves.items()})
    (args.output / 'provenance.json').write_text(json.dumps(dict(
        complete=True, created_utc=datetime.now(timezone.utc).isoformat(), source_code_sha256=digest(_paper_path(__file__)),
        panels='Rows: Gemini, Nemotron. Columns: SAE above pooled positive median, KMeans membership.',
        sae_widths=WIDTHS, kmeans_widths=KMEANS_WIDTHS, kmeans_iterations=100,
        bandwidth_log10=BANDWIDTH, scaling='Number of positive-rate features; square-root y axis; shared axes',
        zero_rates='Omitted from logarithmic axes; retained in summary counts',
        validation='Saved counts divided by identical family corpus sizes; median threshold provenance; '
                   'KDE bandwidth and count-scaled areas; KMeans curves equal the validated original plot.',
        inputs=records), indent=2) + '\n')
    print(f'Wrote {args.output / (stem + ".pdf")}; validated {len(records)} curves.', flush=True)


if __name__ == '__main__':
    main()
