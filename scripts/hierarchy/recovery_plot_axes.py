"""Linked raw-count axes for fixed-cohort recovery proportion plots."""
from matplotlib.ticker import MaxNLocator, StrMethodFormatter


def add_count_axis(ax, denominator: int, *, fontsize: int = 18, tick_fontsize: int = 14):
    if denominator <= 0 or int(denominator) != denominator:
        raise ValueError("The count-axis denominator must be a positive integer")
    denominator = int(denominator)
    axis = ax.secondary_yaxis(
        "right",
        functions=(lambda proportion: proportion * denominator, lambda count: count / denominator),
    )
    candidates = MaxNLocator(nbins=5, integer=True).tick_values(0, denominator)
    ticks = [int(round(value)) for value in candidates if 0 <= value <= denominator]
    if not ticks or ticks[0] != 0:
        ticks.insert(0, 0)
    if ticks[-1] != denominator:
        step = ticks[-1] - ticks[-2] if len(ticks) > 1 else denominator
        if len(ticks) > 1 and denominator - ticks[-1] < 0.35 * step:
            ticks[-1] = denominator
        else:
            ticks.append(denominator)
    axis.set_yticks(ticks)
    axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axis.set_ylabel("Number recovered", fontsize=fontsize, labelpad=10)
    axis.tick_params(axis="y", labelsize=tick_fontsize)
    return axis
