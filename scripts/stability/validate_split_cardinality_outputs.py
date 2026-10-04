#!/usr/bin/env python
"""Validate cardinality matchings using a minimum vertex cover certificate."""
from project_paths import resource_path as _paper_path, resource_location as _paper_location
from collections import deque
import csv
import json
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz


def certify_maximum(graph, source, destination):
    """An equal-size vertex cover certifies a matching's global optimality."""
    assert len(np.unique(source)) == len(np.unique(destination)) == len(source)
    if len(source):
        assert np.all(np.asarray(graph[source, destination]))
    left_partner = np.full(graph.shape[0], -1, dtype=int)
    right_partner = np.full(graph.shape[1], -1, dtype=int)
    left_partner[source] = destination
    right_partner[destination] = source
    reached_left = left_partner < 0
    reached_right = np.zeros(graph.shape[1], dtype=bool)
    queue = deque(np.flatnonzero(reached_left))
    # Alternate unmatched L->R edges and matched R->L edges.
    while queue:
        left = queue.popleft()
        for right in graph.indices[graph.indptr[left]:graph.indptr[left + 1]]:
            if right == left_partner[left] or reached_right[right]:
                continue
            reached_right[right] = True
            partner = right_partner[right]
            assert partner >= 0, 'An augmenting path exists; matching is not maximum.'
            if not reached_left[partner]:
                reached_left[partner] = True
                queue.append(partner)
    # Cover = (unreached left vertices) union (reached right vertices).
    assert int((~reached_left).sum() + reached_right.sum()) == len(source)
    assert graph[reached_left][:, ~reached_right].nnz == 0


def main():
    root = _paper_path('full_experiments/results/split_matches/plain_and_maximum_cardinality')
    previous = root.parent / 'plain_and_maximum_weight'
    paths = list(root.glob('pairs/*/counts.json'))
    assert len(paths) == 60
    comparison = []
    for path in paths:
        result = json.loads(path.read_text())
        old = json.loads((previous / 'pairs' / path.parent.name / 'counts.json').read_text())
        for t in (.7, .8):
            r = result[str(t)]
            assignment = np.load(_paper_location(path.parent / f'maximum_cardinality_{t:g}_assignment.npz'))
            source, destination, scores = (assignment[k] for k in ('source', 'destination', 'similarity'))
            graph = load_npz(_paper_location(path.parent / f'threshold_{t:g}_graph.npz'))
            certify_maximum(graph, source, destination)
            assert np.all(scores > t)
            assert len(source) == r['maximum_cardinality'] == r['assignment_size']
            assert r['edges'] == graph.nnz
            assert r['plain_left'] == np.count_nonzero(np.diff(graph.indptr))
            assert r['plain_right'] == len(np.unique(graph.indices))
            assert len(source) <= min(r['plain_left'], r['plain_right'])
            assert len(source) >= old[str(t)]['maximum_weight']
            comparison.append(dict(pair=path.parent.name, threshold=t,
                maximum_cardinality=len(source), maximum_weight_count=old[str(t)]['maximum_weight'],
                gain=len(source)-old[str(t)]['maximum_weight']))
        for key in ('maximum_cardinality', 'plain_left', 'plain_right'):
            assert result['0.8'][key] <= result['0.7'][key]
    rows = list(csv.DictReader((root / 'counts.csv').open()))
    assert len(rows) == 480
    indexed = {(r['model'],r['representation'],r['threshold'],r['metric'],r['source_split'],r['comparison_split']):r for r in rows}
    assert len(indexed) == 480
    for key, row in indexed.items():
        n, width = int(row['matches']), int(row['source_features'])
        assert 0 <= n <= width
        assert np.isclose(float(row['proportion']), n / width)
        if row['metric'] == 'maximum_cardinality':
            assert row['matches'] == indexed[(*key[:4], key[5], key[4])]['matches']
    for extension in ('png', 'pdf'):
        assert len(list(root.glob(f'threshold_*/*/combined_*.{extension}'))) == 8
    with (root / 'comparison_to_maximum_weight.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    print('Certified optimality of all 120 threshold matchings; validated 480 rows and 16 plot files.')
    print('Pairs improved:', sum(r['gain'] > 0 for r in comparison), '; largest gain:', max(r['gain'] for r in comparison))


if __name__ == '__main__':
    main()
