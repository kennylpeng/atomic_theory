#!/usr/bin/env python3
"""Final dates figure: embedding and month-feature PCAs above the feature heatmap."""
from project_paths import resource_path as _paper_path, resource_location as _paper_location
import calendar
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import numpy as np

RESULTS = _paper_path(__file__).resolve().parent / 'results'


def main():
    feature_dir = RESULTS / 'gemini_smaller_dictionaries/m131072'
    with np.load(_paper_location(RESULTS / 'gemini_date_embeddings_pca/gemini_date_embeddings_pca3.npz')) as data:
        dates = data['texts']
        embedding_pca = data['coordinates'][:, :2]
    with np.load(_paper_location(feature_dir / 'gemini_month_features_pca2.npz')) as data:
        np.testing.assert_array_equal(dates, data['dates'])
        feature_pca = data['coordinates']
        pca_features = data['features']
    with np.load(_paper_location(feature_dir / 'matched_feature_activations.npz')) as data:
        np.testing.assert_array_equal(dates, data['dates'])
        features = data['matched_features']
        np.testing.assert_array_equal(features, pca_features)
        matrix = data['activations'].astype(np.float64).T
    months = [calendar.month_name[i] for i in range(1, 13)]
    month_ids = np.array([months.index(str(date).split()[0]) for date in dates])
    lengths = np.bincount(month_ids, minlength=12)
    np.testing.assert_array_equal(month_ids, np.repeat(np.arange(12), lengths))
    edges = np.r_[0, np.cumsum(lengths)] - 0.5
    palette = plt.get_cmap('twilight_shifted')(np.arange(12) / 12)
    resultant = matrix @ np.exp(2j * np.pi * np.arange(len(dates)) / len(dates))
    centers = np.mod(np.angle(resultant), 2 * np.pi)
    centers[np.abs(resultant) <= 1e-12 * matrix.sum(axis=1)] = np.inf
    order = np.argsort(centers, kind='stable')
    matrix = matrix[order]

    with plt.rc_context({'font.size': 11, 'axes.titlesize': 14, 'axes.labelsize': 12}):
        fig = plt.figure(figsize=(14, 9))
        outer = fig.add_gridspec(2, 2, width_ratios=[1, 0.025], height_ratios=[1, 0.50],
                                left=0.09, right=0.91, bottom=0.08, top=0.95, hspace=0.22, wspace=0.045)
        top = outer[0, 0].subgridspec(1, 2, wspace=0.18)
        pca_axes = []
        for index, (coordinates, title) in enumerate([
            (embedding_pca, 'Raw embedding PCA'),
            (feature_pca, '131K month-feature PCA'),
        ]):
            ax = fig.add_subplot(top[0, index])
            ax.scatter(coordinates[:, 0], coordinates[:, 1], c=palette[month_ids], s=19)
            ax.set(xlabel='PC1', ylabel='PC2', title=title)
            ax.set_box_aspect(1)
            ax.set_aspect('equal', adjustable='datalim')
            ax.set_anchor('W' if index == 0 else 'E')
            pca_axes.append(ax)
        bottom = outer[1, 0].subgridspec(2, 1, height_ratios=[0.13, 1], hspace=0.04)
        bar_ax = fig.add_subplot(bottom[0, 0])
        heat_ax = fig.add_subplot(bottom[1, 0], sharex=bar_ax)
        cb_grid = outer[1, 1].subgridspec(2, 1, height_ratios=[0.13, 1], hspace=0.04)
        colorbar_ax = fig.add_subplot(cb_grid[1, 0])
        bar_ax.set_title('Gemini 131K month-feature activations', pad=12)
        bar_ax.imshow(month_ids[None, :], aspect='auto', interpolation='nearest',
                      cmap=ListedColormap(palette), vmin=-0.5, vmax=11.5)
        for month, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
            luminance = palette[month, :3] @ np.array([0.2126, 0.7152, 0.0722])
            bar_ax.text((left + right) / 2, 0, calendar.month_abbr[month + 1] + '.',
                        ha='center', va='center', fontsize=10,
                        color='black' if luminance > 0.55 else 'white')
        for edge in edges[1:-1]:
            bar_ax.axvline(edge, color='white', linewidth=0.8)
        bar_ax.set_axis_off()
        heatmap = heat_ax.imshow(matrix, aspect='auto', interpolation='nearest', cmap=LinearSegmentedColormap.from_list('white_orange', ['#ffffff', '#ff8000']),
                                 vmin=0, vmax=0.2)
        heat_ax.set(xlabel='Day of year', ylabel='Feature', yticks=np.arange(len(features)),
                    yticklabels=[f'f{features[i]}' for i in order])
        heat_ax.set_xticks([])
        fig.colorbar(heatmap, cax=colorbar_ax, label='Raw post-TopK activation')
        fig.canvas.draw()
        positions = np.column_stack((np.arange(len(dates)), np.zeros(len(dates))))
        np.testing.assert_allclose(bar_ax.transData.transform(positions)[:, 0],
                                   heat_ax.transData.transform(positions)[:, 0], rtol=0, atol=1e-10)
        panel_sizes = np.array([[ax.get_window_extent().width, ax.get_window_extent().height]
                                for ax in pca_axes])
        np.testing.assert_allclose(panel_sizes[0], panel_sizes[1], rtol=0, atol=1e-8)
        np.testing.assert_allclose(
            [pca_axes[0].get_position().x0, pca_axes[1].get_position().x1],
            [heat_ax.get_position().x0, heat_ax.get_position().x1],
            rtol=0, atol=1e-10,
        )
        print('Verified equal PCA panel sizes and matching PCA/heatmap row edges.')
        for ax in pca_axes:
            assert not ax.lines and ax.get_legend() is None
            assert ax.get_xlabel() == 'PC1' and ax.get_ylabel() == 'PC2'
            np.testing.assert_allclose(ax.collections[0].get_facecolors(), palette[month_ids])
        output = RESULTS / 'gemini_dates_summary'
        fig.savefig(output.with_suffix('.png'), dpi=200)
        fig.savefig(output.with_suffix('.pdf'))
        plt.close(fig)
    metadata = dict(heatmap_palette=['#ffffff', '#ff8000'], heatmap_limits=[0.0, 0.2], dates=len(dates), features_in_row_order=features[order].tolist(),
                    row_order_method='activation-weighted circular mean of day position',
                    month_lengths=lengths.tolist(), pca_coordinates='existing saved coordinates, unchanged',
                    outputs=[str(output.with_suffix(ext)) for ext in ('.png', '.pdf')])
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
