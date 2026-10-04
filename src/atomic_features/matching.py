"""Independent pairwise maximum matching and intersection persistence."""

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


def assignment(graph):
    """Return one destination per source row, or -1 when unmatched."""
    return maximum_bipartite_matching(graph, perm_type="column")


def validate_witness(graphs, selected, destinations):
    """Check that one source subset has an injective matching in every graph."""
    selected = np.asarray(selected)
    if (
        selected.ndim != 1
        or selected.dtype.kind not in "iu"
        or len(np.unique(selected)) != len(selected)
    ):
        raise ValueError("Source indices must be unique integers")
    if len(destinations) != len(graphs):
        raise ValueError("One destination vector per graph is required")
    for graph, dst in zip(graphs, destinations):
        dst = np.asarray(dst)
        if (
            dst.ndim != 1
            or dst.dtype.kind not in "iu"
            or len(dst) != len(selected)
            or len(np.unique(dst)) != len(dst)
        ):
            raise ValueError(
                "Destination indices must be unique integers aligned to sources"
            )
        if (
            np.any(selected < 0)
            or np.any(selected >= graph.shape[0])
            or np.any(dst < 0)
            or np.any(dst >= graph.shape[1])
        ):
            raise ValueError("Witness index out of bounds")
        if len(selected) and not np.all(np.asarray(graph[selected, dst])):
            raise ValueError("Witness contains a missing edge")


def certify_pairwise_maximum(graph, match):
    """Certify a maximum matching using an equal-size vertex cover."""
    from collections import deque
    source = np.flatnonzero(match >= 0)
    validate_witness([graph], source, [match[source]])
    partner = np.full(graph.shape[1], -1, dtype=int)
    partner[match[source]] = source
    left = match < 0
    right = np.zeros(graph.shape[1], dtype=bool)
    queue = deque(np.flatnonzero(left))
    while queue:
        u = queue.popleft()
        for v in graph.indices[graph.indptr[u]:graph.indptr[u + 1]]:
            if v == match[u] or right[v]:
                continue
            right[v] = True
            other = partner[v]
            if other < 0:
                raise ValueError("Pairwise matching has an augmenting path")
            if not left[other]:
                left[other] = True
                queue.append(other)
    if int((~left).sum() + right.sum()) != len(source) or graph[left][:, ~right].nnz:
        raise ValueError("Invalid minimum vertex cover certificate")


def independent_assignments(graphs):
    """Match each complete source graph, then intersect matched source IDs.

    Ascending original feature IDs and sorted CSR storage fix tie-breaking for
    a given SciPy version. No common-support prefiltering or joint optimization.
    Returns source IDs and full assignments, with -1 for unmatched sources.
    """
    if not graphs:
        raise ValueError("At least one larger dictionary is required.")
    n = graphs[0].shape[0]
    assignments = []
    for graph in graphs:
        if graph.shape[0] != n:
            raise ValueError("All graphs must index the same source dictionary.")
        g = csr_matrix(graph, dtype=bool, copy=True)
        g.eliminate_zeros()
        g.sum_duplicates()
        g.sort_indices()
        match = assignment(g)
        certify_pairwise_maximum(g, match)
        assignments.append(match)
    selected = np.flatnonzero(np.logical_and.reduce([a >= 0 for a in assignments]))
    return selected, assignments


def persistent_matching(graphs, time_limit=600):
    """Intersection of independently selected maximum matchings.

    time_limit is ignored by this pairwise algorithm. Persistence
    is tie-dependent and need not be monotone under changes to threshold edges.
    """
    import scipy
    selected, matches = independent_assignments(graphs)
    destinations = [a[selected] for a in matches]
    validate_witness(graphs, selected, destinations)
    stats = dict(method="independent_pairwise_intersection",
                 certificate="pairwise_vertex_covers_and_source_intersection",
                 scipy_version=scipy.__version__,
                 tie_breaking="ascending original IDs, sorted CSR, perm_type=column",
                 pairwise_counts=[int(np.count_nonzero(a >= 0)) for a in matches],
                 candidates=int(np.logical_and.reduce([np.asarray(csr_matrix(g).astype(bool).sum(axis=1)).ravel() > 0 for g in graphs]).sum()))
    return selected, destinations, stats


def cosine_graph(source, target, threshold=0.7, absolute=False, batch_size=1024):
    """Build an exhaustive inclusive cosine-threshold graph.

    Similarity blocks are bounded by batch_size on both axes. Normalized
    dictionaries are held in RAM. Use absolute=True for sign-invariant PCA.
    """
    source, target = np.asarray(source, dtype=np.float32), np.asarray(
        target, dtype=np.float32
    )
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("Dictionaries must have matching embedding dimensions")
    if not -1 <= threshold <= 1 or batch_size < 1:
        raise ValueError("Invalid threshold or batch size")

    def normalized(x):
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        if not np.isfinite(x).all() or np.any(norms == 0):
            raise ValueError("Cosine requires finite nonzero vectors")
        return x / norms

    a, b = normalized(source), normalized(target)
    rows, cols = [], []
    for i in range(0, len(a), batch_size):
        for j in range(0, len(b), batch_size):
            scores = a[i : i + batch_size] @ b[j : j + batch_size].T
            if absolute:
                scores = np.abs(scores)
            r, c = np.nonzero(scores >= np.float32(threshold))
            rows.extend(r + i)
            cols.extend(c + j)
    return csr_matrix(
        (np.ones(len(rows), dtype=bool), (rows, cols)), shape=(len(a), len(b))
    )


def candidate_graph(
    forward_ids, forward_scores, reverse_ids, reverse_scores, threshold=0.7
):
    """Union of directed candidates; signed Pearson >= threshold (inclusive).

    Optimal cardinality on this graph is a lower bound on exhaustive matching.
    This bound does not apply to candidate-restricted mutual nearest neighbors.
    """
    n, m = len(forward_ids), len(reverse_ids)
    if not -1 <= threshold <= 1:
        raise ValueError("Invalid threshold")
    rows, cols = [], []
    for ids, scores, limit, reverse in [
        (forward_ids, forward_scores, m, False),
        (reverse_ids, reverse_scores, n, True),
    ]:
        ids, scores = np.asarray(ids), np.asarray(scores)
        if ids.ndim != 2 or scores.shape != ids.shape or ids.dtype.kind not in "iu":
            raise ValueError(
                "Candidate arrays must be aligned matrices with integer IDs"
            )
        keep = (
            (ids >= 0)
            & (ids < limit)
            & np.isfinite(scores)
            & (scores >= np.float32(threshold))
        )
        r, k = np.nonzero(keep)
        c = ids[r, k]
        rows.extend(c if reverse else r)
        cols.extend(r if reverse else c)
    graph = csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(n, m))
    graph.sum_duplicates()
    return graph
