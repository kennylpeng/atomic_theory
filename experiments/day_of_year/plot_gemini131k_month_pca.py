#!/usr/bin/env python3
"""Plot 2D PCA of the 12 cached Gemini 131K month-feature activations."""
from project_paths import resource_path as _paper_path, resource_location as _paper_location
import calendar
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def main():
    folder = _paper_path(__file__).resolve().parent / 'results/gemini_smaller_dictionaries/m131072'
    with np.load(_paper_location(folder / 'matched_feature_activations.npz')) as data:
        dates = data['dates']
        features = data['matched_features']
        activations = data['activations']
    assert activations.shape == (366, 12)
    # Center raw activations without standardizing individual features.
    pca = PCA(n_components=2, svd_solver='full')
    coordinates = pca.fit_transform(activations)
    months = [calendar.month_name[i] for i in range(1, 13)]
    month_ids = np.array([months.index(str(date).split()[0]) for date in dates])
    palette = plt.get_cmap('twilight_shifted')(np.arange(12) / 12)
    fig, ax = plt.subplots(figsize=(9, 8))
    for i in range(12):
        points = coordinates[month_ids == i]
        ax.scatter(points[:, 0], points[:, 1], color=palette[i], s=22, label=calendar.month_abbr[i+1]+'.', zorder=2)
    variance = pca.explained_variance_ratio_
    ax.set(xlabel=f'PC1 ({variance[0]:.2%} variance)', ylabel=f'PC2 ({variance[1]:.2%} variance)', title='2D PCA of Gemini 131K month features\n366 dates · 12 features · raw post-TopK activations')
    ax.set_aspect('equal', adjustable='box')
    ax.legend(title='Month', loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    output = folder / 'gemini_month_features_pca2.png'
    fig.savefig(output, dpi=180)
    plt.close(fig)
    np.savez_compressed(_paper_location(folder / 'gemini_month_features_pca2.npz'), coordinates=coordinates, dates=dates, features=features, components=pca.components_, mean=pca.mean_, explained_variance_ratio=variance)
    metadata = dict(features=features.tolist(), date_count=len(dates), preprocessing='Mean-centered raw post-TopK activations; no feature standardization', explained_variance_ratio=variance.tolist(), total_explained_variance=float(variance.sum()), output=str(output))
    (folder / 'gemini_month_features_pca2.json').write_text(json.dumps(metadata, indent=2)+'\n')
    previous = np.load(_paper_location(folder / 'gemini_matched_features_pca2.npy'))
    # PCA axis signs are arbitrary; confirm the same date geometry as the existing PCA.
    np.testing.assert_allclose(coordinates @ coordinates.T, previous @ previous.T, atol=1e-6, rtol=1e-4)
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
