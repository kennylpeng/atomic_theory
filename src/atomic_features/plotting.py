"""Plotting only: numerical verification lives in reproduce.py."""

import numpy as np
from .hierarchy import WIDTHS, LEVEL_SPECS


def write_plots(output, persistence, splits, hierarchy):
    """Render the three principal empirical analyses from verified numeric rows."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"font.size": 12, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for ax, key in zip(axes, ["persistent_count", "proportion"]):
        for model, color, marker in [
            ("gemini", "#0072B2", "o"),
            ("nemotron", "#D55E00", "s"),
        ]:
            for metric, style in [
                ("decoder_cosine", "-"),
                ("activation_pearson", "--"),
            ]:
                rows = [
                    r
                    for r in persistence
                    if r["model"] == model and r["metric"] == metric
                ]
                ax.plot(
                    [r["width"] for r in rows],
                    [r[key] for r in rows],
                    color=color,
                    marker=marker,
                    linestyle=style,
                    label=f'{model.title()}: {metric.replace("_"," ")}',
                )
        ax.set_xscale("log", base=2)
        ax.set_xticks(
            WIDTHS[:-1], [str(w) if w < 1024 else f"{w//1024}K" for w in WIDTHS[:-1]]
        )
        ax.set(xlabel="SAE width", ylabel=key.replace("_", " ").capitalize())
        ax.grid(alpha=0.2)
        if key == "proportion":
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        else:
            ax.legend(fontsize=9)
    for ext in ("pdf", "png"):
        fig.savefig(output / f"persistence.{ext}", dpi=180)
    plt.close(fig)
    names = ["main", "wikipedia", "no_wikipedia", "random1", "random2"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
    for i, model in enumerate(("gemini", "nemotron")):
        for j, rep in enumerate(("sae", "kmeans", "pca")):
            a = np.eye(5)
            for row in splits:
                if row["model"] == model and row["representation"] == rep:
                    a[
                        names.index(row["source_split"]),
                        names.index(row["comparison_split"]),
                    ] = float(row["proportion"])
            ax = axes[i, j]
            im = ax.imshow(a, vmin=0, vmax=1, cmap="Blues")
            ax.set_xticks(range(5), names, rotation=45, ha="right")
            ax.set_yticks(range(5), names)
            ax.set_title(f"{model.title()} {rep.upper()}")
            for y in range(5):
                for x in range(5):
                    ax.text(
                        x,
                        y,
                        f"{a[y,x]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="white" if a[y, x] > 0.55 else "black",
                    )
    fig.colorbar(
        im, ax=axes.ravel().tolist(), label="Matched source proportion", shrink=0.7
    )
    fig.savefig(output / "split_matches.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    a = np.array(
        [
            [
                next(
                    r["mean_test_f1"]
                    for r in hierarchy
                    if r["level"] == s.level and r["width"] == w
                )
                for w in WIDTHS
            ]
            for s in LEVEL_SPECS
        ]
    )
    im = ax.imshow(a, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(9), [str(w) if w < 1024 else f"{w//1024}K" for w in WIDTHS])
    ax.set_yticks(range(10), [f"{s.domain}: {s.label}" for s in LEVEL_SPECS])
    ax.set_xlabel("SAE width")
    fig.colorbar(im, ax=ax, label="Pooled held-out F1")
    fig.savefig(output / "hierarchy.pdf")
    plt.close(fig)
