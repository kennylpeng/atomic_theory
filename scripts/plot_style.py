"""Shared plotting style for dictionary and recovery analyses."""

from __future__ import annotations

from typing import Iterable

SAE_TRUE_COLOR = "#006D77"
KMEANS_COLOR = "#8B5A2B"
PCA_COLOR = "#228B22"
FAMILY_MARKERS = {"sae": "o", "kmeans": "s", "pca": "^"}
FAMILY_LINEWIDTHS = {"sae": 3.0, "kmeans": 2.5, "pca": 2.5}
FAMILY_MARKERSIZES = {"sae": 8.0, "kmeans": 8.0, "pca": 8.0}


def apply_plot_style() -> None:
    """Apply repository-wide typography and axis defaults."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 15,
        "axes.labelsize": 17,
        "axes.titlesize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "figure.titlesize": 18,
    })


def sae_gradient(top_ks: Iterable[int]) -> dict[int, tuple[float, float, float, float]]:
    """Return increasingly dark blue colors ordered by SAE sparsity."""
    import matplotlib.pyplot as plt

    values = sorted(set(int(top_k) for top_k in top_ks))
    if not values:
        return {}
    cmap = plt.get_cmap("Blues")
    if len(values) == 1:
        return {values[0]: cmap(0.65)}
    return {
        top_k: cmap(0.4 + 0.55 * index / (len(values) - 1))
        for index, top_k in enumerate(values)
    }


def family_style(family: str) -> dict[str, object]:
    """Return the standard line style for one dictionary family."""
    colors = {
        "sae": SAE_TRUE_COLOR,
        "kmeans": KMEANS_COLOR,
        "pca": PCA_COLOR,
    }
    if family not in colors:
        raise ValueError(f"Unknown dictionary family: {family}")
    return {
        "color": colors[family],
        "marker": FAMILY_MARKERS[family],
        "linewidth": FAMILY_LINEWIDTHS[family],
        "markersize": FAMILY_MARKERSIZES[family],
    }
