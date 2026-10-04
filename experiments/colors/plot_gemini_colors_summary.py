#!/usr/bin/env python3
"""Fit 3D color PCAs on the same filtered colors shown in the feature heatmap."""
from project_paths import resource_path as _paper_path, resource_location as _paper_location
import csv
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import numpy as np
from sklearn.decomposition import PCA

ROOT = _paper_path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.colors.plot_selected_color_feature_swatches import chromatic_indices
RESULTS = ROOT / 'experiments/colors/results'


def main():
    feature_dir = RESULTS / 'color_names_hex_gemini_d131072_selected_ranks_thr0p05_pca'
    meta = json.loads((feature_dir / 'run_meta.json').read_text())
    assert meta['standardized_before_pca'] is False, 'Rebuild color feature PCA using mean-centering only'
    features = meta['neuron_ids']
    assert len(features) == 8
    with (feature_dir / 'selected_ranks_thr0p05_hex_activations_with_pcs.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    embedding_dir = ROOT / 'experiments/colors/data/gemini_embeddings'
    with (embedding_dir / 'rows.csv').open() as handle:
        embedded_rows = list(csv.DictReader(handle))
    np.testing.assert_array_equal([r['id'] for r in embedded_rows], [r['id'] for r in rows])
    embeddings = np.load(_paper_location(embedding_dir / 'embeddings.npy'))
    activations = np.load(_paper_location(feature_dir / 'selected_ranks_thr0p05_unthresholded_activation_matrix.npy'))
    first_by_hex = {}
    for index, row in enumerate(rows):
        first_by_hex.setdefault(row['hex'].lower(), index)
    heat_dir = RESULTS / 'color_names_hex_gemini_d131072_selected_ranks_thr0p05_swatches'
    with (heat_dir / 'selected_features_hue_activations.csv').open() as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames[4:]
        hue_rows = sorted(reader, key=lambda row: int(row['hue_order']))
    assert [int(c.split('_')[-1]) for c in columns] == features
    for row in hue_rows:
        row.update({channel: str(int(row['hex'][start:start+2], 16))
                    for channel, start in [('r', 1), ('g', 3), ('b', 5)]})
    kept = chromatic_indices(hue_rows, list(range(len(hue_rows))))
    hexes = [hue_rows[i]['hex'] for i in kept]
    matrix = np.array([[float(hue_rows[i][column]) for column in columns] for i in kept]).T
    selected_rows = np.array([first_by_hex[h.lower()] for h in hexes])
    assert len(selected_rows) == len(set(hexes)) == 374
    np.testing.assert_allclose(activations[selected_rows], matrix.T)
    point_colors = hexes
    embedding_fit = PCA(n_components=3, svd_solver='full')
    feature_fit = PCA(n_components=3, svd_solver='full')
    embedding_pca = embedding_fit.fit_transform(embeddings[selected_rows].astype(np.float64))
    feature_pca = feature_fit.fit_transform(matrix.T)
    np.savez_compressed(_paper_location(RESULTS / 'gemini_colors_summary_pca.npz'),
                        hexes=np.array(hexes), source_rows=selected_rows, features=np.array(features),
                        embedding_coordinates=embedding_pca, feature_coordinates=feature_pca,
                        embedding_components=embedding_fit.components_, feature_components=feature_fit.components_,
                        embedding_mean=embedding_fit.mean_, feature_mean=feature_fit.mean_,
                        embedding_explained_variance_ratio=embedding_fit.explained_variance_ratio_,
                        feature_explained_variance_ratio=feature_fit.explained_variance_ratio_)
    resultant = matrix @ np.exp(2j * np.pi * np.arange(len(kept)) / len(kept))
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
            (feature_pca, '131K color-feature PCA'),
        ]):
            ax = fig.add_subplot(top[0, index], projection='3d')
            display_coordinates = coordinates
            ax.scatter(*display_coordinates.T, c=point_colors, s=18, edgecolors='#202020',
                       linewidths=0.2, depthshade=False)
            ax.set(xlabel='PC1', ylabel='PC2', zlabel='PC3', title=title)
            ax.set_box_aspect(np.maximum(np.ptp(display_coordinates, axis=0), 1e-12))
            ax.view_init(elev=24, azim=40)
            ax.set_anchor('W' if index == 0 else 'E')
            ax.tick_params(labelsize=8, pad=1)
            pca_axes.append(ax)
        bottom = outer[1, 0].subgridspec(2, 1, height_ratios=[0.13, 1], hspace=0.04)
        bar_ax = fig.add_subplot(bottom[0, 0])
        heat_ax = fig.add_subplot(bottom[1, 0], sharex=bar_ax)
        cb_grid = outer[1, 1].subgridspec(2, 1, height_ratios=[0.13, 1], hspace=0.04)
        colorbar_ax = fig.add_subplot(cb_grid[1, 0])
        bar_ax.set_title('Gemini 131K color-feature activations', pad=12)
        bar_ax.imshow(np.arange(len(kept))[None, :], aspect='auto', interpolation='nearest', cmap=ListedColormap(hexes))
        bar_ax.set_axis_off()
        heatmap = heat_ax.imshow(matrix, aspect='auto', interpolation='nearest', cmap=LinearSegmentedColormap.from_list('white_orange', ['#ffffff', '#ff8000']), vmin=0, vmax=0.2)
        heat_ax.set(xlabel='Distinct colors sorted by hue', ylabel='Feature', yticks=np.arange(len(features)),
                    yticklabels=[f'f{features[i]}' for i in order])
        heat_ax.set_xticks([])
        fig.colorbar(heatmap, cax=colorbar_ax, label='Raw post-TopK activation')
        fig.canvas.draw()
        positions = np.column_stack((np.arange(len(kept)), np.zeros(len(kept))))
        np.testing.assert_allclose(bar_ax.transData.transform(positions)[:, 0], heat_ax.transData.transform(positions)[:, 0], rtol=0, atol=1e-10)
        sizes = np.array([[ax.get_window_extent().width, ax.get_window_extent().height] for ax in pca_axes])
        np.testing.assert_allclose(sizes[0], sizes[1], rtol=0, atol=1e-8)
        np.testing.assert_allclose([pca_axes[0].get_position().x0, pca_axes[1].get_position().x1],
                                   [heat_ax.get_position().x0, heat_ax.get_position().x1], rtol=0, atol=1e-10)
        for ax in pca_axes:
            assert not ax.lines and ax.get_legend() is None
        output = RESULTS / 'gemini_colors_summary'
        fig.savefig(output.with_suffix('.png'), dpi=200)
        fig.savefig(output.with_suffix('.pdf'))
        plt.close(fig)
    summary = dict(heatmap_palette=['#ffffff', '#ff8000'], heatmap_limits=[0.0, 0.2], pca_rows=len(kept), heatmap_colors=len(kept), features_in_row_order=[features[i] for i in order],
                   pca_coordinates='Refit on the same 374 distinct filtered colors as the heatmap',
                   pca_data=str(RESULTS / 'gemini_colors_summary_pca.npz'),
                   embedding_explained_variance_ratio=embedding_fit.explained_variance_ratio_.tolist(),
                   feature_explained_variance_ratio=feature_fit.explained_variance_ratio_.tolist(),
                   displayed_axes=['PC1', 'PC2', 'PC3'],
                   embedding_pca_preprocessing='Mean-centered; no standardization',
                   feature_pca_preprocessing='Mean-centered raw post-TopK activations; no standardization or normalization',
                   heatmap_filter='HSV saturation >= 0.60, value >= 0.50, HLS lightness <= 0.70',
                   row_order='Activation-weighted circular mean of displayed hue position',
                   outputs=[str(output.with_suffix(ext)) for ext in ('.png', '.pdf')])
    output.with_suffix('.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
