"""Train-only feature/threshold selection for nonnegative sparse activations."""

from dataclasses import dataclass
import numpy as np
from scipy.sparse import csc_matrix


@dataclass(frozen=True)
class Probe:
    feature: int
    threshold: float


def _inputs(activations, labels):
    x = csc_matrix(activations, dtype=np.float32)
    x.sum_duplicates()
    x.eliminate_zeros()
    y = np.asarray(labels)
    if y.shape != (x.shape[0],) or not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be one binary value per row")
    if not np.isfinite(x.data).all() or np.any(x.data < 0):
        raise ValueError("Expected finite, nonnegative activations")
    return x, y.astype(bool)


def fit_probe(activations, labels, min_tp=1):
    """Pass training rows only. Positive thresholds include ties (>=).

    Within a feature, tied F1 selects the largest threshold. Across features,
    rank by F1, true positives, then fewer false positives, then smaller ID.
    """
    x, y = _inputs(activations, labels)
    if min_tp < 1:
        raise ValueError("min_tp must be positive")
    total = int(y.sum())
    best, winner = None, None
    for f in range(x.shape[1]):
        a, b = x.indptr[f : f + 2]
        values = x.data[a:b]
        if not len(values):
            continue
        order = np.argsort(values, kind="mergesort")[::-1]
        values = values[order]
        tp = np.cumsum(y[x.indices[a:b]][order])
        ends = np.flatnonzero(np.r_[values[1:] != values[:-1], True])
        counts = ends + 1
        true = tp[ends]
        f1 = np.where(true >= min_tp, 2 * true / (total + counts), 0)
        i = int(np.argmax(f1))
        if true[i] < min_tp:
            continue
        rank = (f1[i], int(true[i]), -int(counts[i] - true[i]))
        if best is None or rank > best:
            best, winner = rank, Probe(f, float(values[ends[i]]))
    if winner is None:
        raise ValueError("No feature meets the training-positive requirement")
    return winner


def evaluate_probe(probe, activations, labels):
    x, y = _inputs(activations, labels)
    if (
        not 0 <= probe.feature < x.shape[1]
        or not np.isfinite(probe.threshold)
        or probe.threshold <= 0
    ):
        raise ValueError("Invalid probe")
    pred = x[:, probe.feature].toarray().ravel() >= probe.threshold
    tp, fp, fn = int(np.sum(pred & y)), int(np.sum(pred & ~y)), int(np.sum(~pred & y))
    return dict(
        tp=tp,
        fp=fp,
        fn=fn,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        precision=tp / (tp + fp) if tp + fp else 0.0,
        recall=tp / (tp + fn) if tp + fn else 0.0,
    )
