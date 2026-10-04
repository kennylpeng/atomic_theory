"""Iteration-labeled rendering of the existing five-distribution activation matrix."""
import numpy as np

from scripts.match_all_split_kmeans_activations import FAMILIES, SPLITS, LABELS


def plot(output, records, threshold, iterations):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'font.size': 11, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6), layout='constrained')
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('#eeeeee')
    vmax = min(1., max(.4, np.ceil(max(row['proportion'] for row in records)*10)/10))
    for ax, family in zip(axes, FAMILIES):
        matrix = np.full((5, 5), np.nan)
        counts = np.full((5, 5), np.nan)
        for row in records:
            if row['family'] == family:
                i, j = SPLITS.index(row['source_split']), SPLITS.index(row['comparison_split'])
                assert np.isnan(matrix[i, j])
                matrix[i, j], counts[i, j] = row['proportion'], row['matches']
        assert np.isfinite(matrix).sum() == 20 and np.isnan(np.diag(matrix)).all()
        im = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=0, vmax=vmax)
        ax.set_xticks(range(5), LABELS, rotation=35, ha='right')
        ax.set_yticks(range(5), LABELS)
        ax.set_title(f'{family.capitalize()} · 16,384-cluster KMeans', pad=12, fontsize=14)
        for i in range(5):
            for j in range(5):
                if i == j:
                    ax.text(j, i, '—', ha='center', va='center', color='#999999')
                    continue
                r, g, b, _ = cmap(im.norm(matrix[i, j]))
                color = 'black' if .2126*r + .7152*g + .0722*b > .55 else 'white'
                ax.text(j, i, f'{int(counts[i,j]):,}\n({matrix[i,j]:.1%})',
                        ha='center', va='center', color=color, fontsize=10.5)
    fig.suptitle(f'Iteration {iterations} · continued from iteration 20', fontsize=15)
    axes[0].set_ylabel('Source model training distribution\nand evaluation distribution')
    fig.supxlabel('Comparison model training distribution', fontsize=12)
    bar = fig.colorbar(im, ax=axes, fraction=.032, pad=.025, format=PercentFormatter(1))
    bar.set_label(f'Matched clusters (Pearson ≥ {threshold:g})')
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(output / f'activation_split_heatmaps_t{threshold:g}.{extension}',
                    dpi=240, bbox_inches='tight', pad_inches=.08)
    plt.close(fig)
    return dict(color_min=0, color_max=vmax, colormap='viridis', iterations=iterations,
                diagonal='omitted self-comparisons')
