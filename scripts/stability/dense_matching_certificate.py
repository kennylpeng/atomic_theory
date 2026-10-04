"""Batched minimum-vertex-cover certificate for dense bipartite graphs."""
import numpy as np


def certify_maximum_batched(graph, source, destination, batch=64):
    assert len(np.unique(source)) == len(np.unique(destination)) == len(source)
    if len(source):
        assert np.all(np.asarray(graph[source, destination]))
    right_partner = np.full(graph.shape[1], -1, dtype=np.int64)
    right_partner[destination] = source
    reached_left = np.ones(graph.shape[0], dtype=bool)
    reached_left[source] = False
    reached_right = np.zeros(graph.shape[1], dtype=bool)
    frontier = np.flatnonzero(reached_left)
    while len(frontier):
        neighbors = np.zeros(graph.shape[1], dtype=bool)
        for start in range(0, len(frontier), batch):
            neighbors[graph[frontier[start:start+batch]].indices] = True
        new_right = neighbors & ~reached_right
        reached_right |= new_right
        partners = right_partner[new_right]
        assert np.all(partners >= 0), 'An augmenting path exists'
        frontier = partners[~reached_left[partners]]
        reached_left[frontier] = True
        # Matched L->R edges may be included above: their R endpoint was
        # already reached when L entered the frontier, so they add no vertices.
    assert int((~reached_left).sum() + reached_right.sum()) == len(source)
    uncovered_right = np.flatnonzero(~reached_right)
    if len(uncovered_right):
        left = np.flatnonzero(reached_left)
        for start in range(0, len(left), batch):
            assert graph[left[start:start+batch]][:, uncovered_right].nnz == 0


def test_certificate():
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching
    from validate_split_cardinality_outputs import certify_maximum
    rng = np.random.default_rng(52)
    for _ in range(100):
        shape = tuple(rng.integers(1, 30, size=2))
        graph = csr_matrix(rng.random(shape) < rng.random())
        assignment = maximum_bipartite_matching(graph, perm_type='column')
        src = np.flatnonzero(assignment >= 0)
        dst = assignment[src]
        certify_maximum(graph, src, dst)
        certify_maximum_batched(graph, src, dst, batch=3)
        if len(src):
            try:
                certify_maximum_batched(graph, src[:-1], dst[:-1], batch=3)
            except AssertionError:
                pass
            else:
                raise AssertionError('Accepted a nonmaximum matching')
    print('Batched certificate agrees with original on 100 graphs and rejects nonmaximum matchings.')


if __name__ == '__main__':
    test_certificate()


def maximum_matching_certified(graph, initial_neighbors=32):
    """Find a matching cheaply, accepting it only with a FULL-graph certificate.

Sample neighbors at both endpoints, preserving all low-degree neighborhoods.
If the full graph admits an augmenting path, expand the candidate graph. The
last fallback uses every edge. This changes runtime, never the exact objective.
"""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching
    rng = np.random.default_rng(20260925)
    reverse = graph.tocsc()
    max_degree = max(int(np.diff(graph.indptr).max(initial=0)), int(np.diff(reverse.indptr).max(initial=0)))
    budget = initial_neighbors
    while True:
        if budget >= max_degree:
            candidates = graph
        else:
            row_parts, col_parts = [], []
            for row in range(graph.shape[0]):
                neighbors = graph.indices[graph.indptr[row]:graph.indptr[row+1]]
                if len(neighbors) > budget:
                    neighbors = rng.choice(neighbors, budget, replace=False)
                row_parts.append(np.full(len(neighbors), row, dtype=np.int32))
                col_parts.append(neighbors)
            for col in range(graph.shape[1]):
                neighbors = reverse.indices[reverse.indptr[col]:reverse.indptr[col+1]]
                if len(neighbors) > budget:
                    neighbors = rng.choice(neighbors, budget, replace=False)
                row_parts.append(neighbors)
                col_parts.append(np.full(len(neighbors), col, dtype=np.int32))
            rows = np.concatenate(row_parts)
            cols = np.concatenate(col_parts)
            candidates = csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=graph.shape)
        assignment = maximum_bipartite_matching(candidates, perm_type='column')
        src = np.flatnonzero(assignment >= 0)
        dst = assignment[src]
        print(f'Candidate budget {budget}: {candidates.nnz} edges, {len(src)} pairs; checking full graph', flush=True)
        try:
            certify_maximum_batched(graph, src, dst)
        except AssertionError:
            if candidates is graph:
                raise
            budget *= 4
            continue
        return src, dst


def test_candidate_solver():
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching
    rng = np.random.default_rng(73)
    for _ in range(30):
        graph = csr_matrix(rng.random((40, 33)) < rng.uniform(.03, .95))
        expected = int((maximum_bipartite_matching(graph, perm_type='column') >= 0).sum())
        src, dst = maximum_matching_certified(graph, initial_neighbors=1)
        assert len(src) == expected
        from validate_split_cardinality_outputs import certify_maximum
        certify_maximum(graph, src, dst)
    print('Candidate solver exactly matches full-graph solver on 30 graphs.')
