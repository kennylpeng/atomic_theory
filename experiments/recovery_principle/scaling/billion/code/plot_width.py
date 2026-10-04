"""Final-budget width scaling, with paired seed means and min/max bands."""
from project_paths import resource_path as _paper_path, resource_location as _paper_location
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator, NullLocator

ROOT = _paper_path(__file__).resolve().parents[1]
BUDGET = 1_024_000_000


def plot_width(df, root=ROOT, proportion=False, measure="prefix"):
    plt.rcParams.update({'font.size': 16, 'axes.labelsize': 16, 'axes.titlesize': 18,
                         'xtick.labelsize': 15, 'ytick.labelsize': 15,
                         'legend.fontsize': 15, 'figure.titlesize': 21,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    if measure not in ("prefix", "count"):
        raise ValueError(measure)
    final = df.loc[df['examples'] == BUDGET].copy()
    if final.empty:
        raise ValueError('No final 1,024M checkpoints available')
    keys = ['kind', 'alpha', 'width', 'k', 'seed']
    if final.duplicated(keys).any():
        raise ValueError('Duplicate final checkpoints')
    plots = root / 'plots'
    plots.mkdir(parents=True, exist_ok=True)
    summaries = []
    for sweep, values, fixed in [
        ('alpha', [0.8, 1.2, 1.6, 2.0, 2.4], final['k'] == 8),
        ('k', [1, 2, 4, 6, 8, 10, 12, 16, 32], np.isclose(final['alpha'], 1.6)),
    ]:
        sub = final.loc[fixed]
        colors = plt.get_cmap('tab10').colors
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        for row, kind in enumerate(['flat', 'hierarchical']):
            widths = [128, 256, 512, 1024, 2048, 4096 if kind == 'flat' else 3072]
            atoms_per_source_unit = 1 if kind == 'flat' else 3
            for col, metric in enumerate([f'geometric_{measure}', f'activation_{measure}']):
                ax = axes[row, col]
                for i, value in enumerate(values):
                    group = sub.loc[(sub['kind'] == kind) & np.isclose(sub[sweep], value)]
                    seeds = group.groupby('width')['seed'].apply(lambda x: set(x))
                    if set(seeds.index) != set(widths) or any(s != {0, 1, 2} for s in seeds):
                        raise ValueError(f'Incomplete three-seed grid: {kind}, {sweep}={value}')
                    stats = group.groupby('width')[metric].agg(['mean', 'min', 'max', 'count']).reindex(widths)
                    stats[['mean', 'min', 'max']] *= atoms_per_source_unit
                    if proportion:
                        stats[['mean', 'min', 'max']] = stats[['mean', 'min', 'max']].div(stats.index.to_numpy(), axis=0)
                    label = (rf'$\alpha={value:g}$' if sweep == 'alpha' else rf'$k={value:g}$')
                    ax.plot(widths, stats['mean'], color=colors[i], marker='o', markersize=5,
                            linewidth=2.1, linestyle='--' if sweep == 'alpha' and value <= 1 else '-', label=label)
                    ax.fill_between(widths, stats['min'].to_numpy(), stats['max'].to_numpy(),
                                    color=colors[i], alpha=0.10, linewidth=0)
                    summary = stats.reset_index().assign(sweep=sweep, value=value, kind=kind,
                                                         metric=metric, examples=BUDGET, units='atoms',
                                                         atoms_per_source_unit=atoms_per_source_unit,
                                                         denominator=widths if proportion else 1)
                    summaries.append(summary)
                ax.set_xscale('log', base=2)
                ax.xaxis.set_major_locator(FixedLocator(widths))
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
                ax.xaxis.set_minor_locator(NullLocator())
                ax.tick_params(axis='x', labelrotation=25)
                ax.set_ylim(bottom=0)
                if proportion:
                    ax.set_ylim(top=max(1.03, ax.get_ylim()[1]))
                    ax.yaxis.set_major_locator(MultipleLocator(.2))
                    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:.1f}'))
                ax.set_xlabel('SAE width')
                quantity = 'Prefix length' if measure == 'prefix' else 'Recovered atoms'
                ax.set_ylabel(f'{quantity} / SAE width' if proportion
                              else ('Prefix length (atoms)' if measure == 'prefix' else quantity))
                ax.set_title(f'{kind.capitalize()} · {"Geometry" if col == 0 else "Activations"}')
                ax.grid(alpha=0.2)
        heading = 'Simulated prefix recovery' if measure == 'prefix' else 'Simulated total atom recovery'
        suffix = 'by distribution' if sweep == 'alpha' else 'by sparsity parameter'
        fig.suptitle(f'{heading} {suffix}', y=.98)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .005),
                   ncol=5 if sweep == 'alpha' else 9, frameon=False,
                   columnspacing=1.0, handlelength=1.6)
        fig.tight_layout(rect=(0, .065, 1, 1), h_pad=1.4, w_pad=1.4)
        stem = f'width_{measure}_{sweep}_sweep_1024M' + ('_proportion' if proportion else '')
        for ext in ['pdf', 'png']:
            fig.savefig(plots / f'{stem}.{ext}', dpi=180)
        plt.close(fig)
        print(plots / f'{stem}.pdf')
    filename = f'width_{measure}_1024M' + ('_proportion' if proportion else '') + '.csv'
    pd.concat(summaries, ignore_index=True).to_csv(root / 'results' / filename, index=False)
    if not proportion:
        plot_width(df, root, proportion=True, measure=measure)
        if measure == "prefix":
            plot_width(df, root, measure="count")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=ROOT / 'results/checkpoints.csv')
    parser.add_argument('--measure', choices=['all', 'count'], default='all',
                        help='Generate all width figures, or only recovery-count figures')
    args = parser.parse_args()
    plot_width(pd.read_csv(args.csv), measure="count" if args.measure == "count" else "prefix")
