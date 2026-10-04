"""Regenerate Figures 2/3 from exhaustive absolute-threshold graphs.

Positive edges of an absolute-cosine >= .7 graph are exactly its signed-cosine
>= .7 graph. Recompute only edge signs, preserving the original float32
threshold decisions. PCA in Figure 3 retains the full absolute graph.
"""
from project_paths import resource_path as _paper_path, resource_location as _paper_location
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = _paper_path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(_paper_path(__file__).resolve().parent))
import numpy as np
import torch
from scipy.sparse import csr_matrix, load_npz, save_npz
from scipy.sparse.csgraph import maximum_bipartite_matching
from project_paths import get_path
from dictionary_exclusive_matches import SIZES, dictionary_path
from utils import load_dictionary
from persistent_matching import persistent_matching, validate_witness
from validate_split_cardinality_outputs import certify_maximum
from split_16384_match_heatmaps import SPLITS, model_paths, load_vectors, plot_combined_heatmaps
from stability_plotting import plot_stability_panels

RESULTS = ROOT / 'full_experiments/results'
PERSISTENT_SOURCE = RESULTS / 'persistent_stability_absolute_0.7'
PERSISTENT_OUT = RESULTS / 'persistent_stability_signed_0.7'
SPLIT_SOURCE = RESULTS / 'split_matches/absolute_maximum_cardinality_0.7'
SPLIT_OUT = RESULTS / 'split_matches/signed_maximum_cardinality_pca_absolute_0.7'


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n')


def fingerprint(path):
    st = path.stat()
    return dict(path=str(path), size=st.st_size, mtime_ns=st.st_mtime_ns)


def graph_fingerprint(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@torch.inference_mode()
def metric_graph(graph, source, target, *, absolute=False, batch=2048):
    """Filter a complete absolute-threshold graph, never a candidate shortlist."""
    assert graph.shape == (len(source), len(target))
    if absolute:
        return graph.copy()
    rows = np.repeat(np.arange(graph.shape[0]), np.diff(graph.indptr))
    keep = np.empty(graph.nnz, dtype=bool)
    for start in range(0, graph.nnz, batch):
        stop = min(start + batch, graph.nnz)
        scores = (source[rows[start:stop]] * target[graph.indices[start:stop]]).sum(dim=1).numpy()
        # Guard against stale inputs, allowing float32 reduction differences.
        assert np.isfinite(scores).all() and np.all(np.abs(scores) >= .7 - 1e-5)
        keep[start:stop] = scores > 0
    return csr_matrix((np.ones(int(keep.sum()), dtype=bool),
                       (rows[keep], graph.indices[keep])), shape=graph.shape)


def persistent():
    all_rows = []
    for model in ('gemini', 'nemotron'):
        source_dir, output = PERSISTENT_SOURCE / model, PERSISTENT_OUT / model
        output.mkdir(parents=True, exist_ok=True)
        old_manifest = json.loads((source_dir / 'manifest.json').read_text())
        paths = [dictionary_path(_paper_path(get_path('models_dir')), model, 'sae', m) for m, _ in SIZES]
        assert [fingerprint(p) for p in paths] == old_manifest['source_files']
        vectors = [load_dictionary(p, 'sae') for p in paths]
        rows, graphs_meta = [], []
        old_counts = {r['width']: r['persistent_count'] for r in json.loads((source_dir / 'summary.json').read_text())}
        for i, (width, _) in enumerate(SIZES[:-1]):
            graphs, targets = [], [m for m, _ in SIZES[i+1:]]
            for j, target_width in enumerate(targets, start=i+1):
                path = source_dir / f'graph_{width}_to_{target_width}.npz'
                original = load_npz(_paper_location(path))
                graph = metric_graph(original, vectors[i], vectors[j])
                save_npz(_paper_location(output / path.name), graph)
                graphs.append(graph)
                graphs_meta.append(dict(**graph_fingerprint(path), original_edges=original.nnz, signed_edges=graph.nnz))
            selected, destinations, stats = persistent_matching(graphs)
            validate_witness(graphs, selected, destinations)
            # Independent intersections need not be monotone when edges are added.
            np.savez_compressed(_paper_location(output / f'witness_{width}.npz'), source=selected,
                                **{f'target_{m}': d for m, d in zip(targets, destinations)})
            row = dict(model=model, width=width, persistent_count=len(selected),
                       proportion=len(selected)/width, target_widths=targets, threshold=.7, **stats)
            write_json(output / f'result_{width}.json', row)
            rows.append(row)
            print(model, width, len(selected), 'absolute:', old_counts[width], flush=True)
        write_json(output / 'summary.json', rows)
        write_json(output / 'manifest.json', dict(model=model, similarity='signed cosine >= float32(0.7)',
                   source_files=old_manifest['source_files'], source_graphs=graphs_meta,
                   method='Keep positive edges of exhaustive absolute-threshold graphs; independent pairwise maximum matching and source intersection.'))
        all_rows.extend(rows)
        del vectors
    fields = ['model', 'width', 'persistent_count', 'proportion']
    with (PERSISTENT_OUT / 'counts.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(all_rows)
    plot_stability_panels(all_rows, PERSISTENT_OUT / 'persistent_stability', count_key='persistent_count', separate=True)


def splits():
    rows, matrices, counts, inputs = [], {}, {}, []
    for model in ('gemini', 'nemotron'):
        for rep in ('sae', 'kmeans', 'pca'):
            paths = model_paths(_paper_path(get_path('models_dir')), model, rep)
            vectors = [load_vectors(p, rep) for _, p in paths]
            inputs.extend(fingerprint(p) for _, p in paths)
            matrix = np.zeros((5, 5), dtype=int)
            for i in range(5):
                for j in range(i+1, 5):
                    name = f'{model}_{rep}_{SPLITS[i]}_to_{SPLITS[j]}'
                    source_dir, output = SPLIT_SOURCE / 'pairs' / name, SPLIT_OUT / 'pairs' / name
                    output.mkdir(parents=True, exist_ok=True)
                    path = source_dir / 'graph.npz'
                    assert all(p.stat().st_mtime_ns <= path.stat().st_mtime_ns for _, p in (paths[i], paths[j]))
                    original = load_npz(_paper_location(path))
                    graph = metric_graph(original, vectors[i], vectors[j], absolute=rep == 'pca')
                    assignment = maximum_bipartite_matching(graph, perm_type='column')
                    src = np.flatnonzero(assignment >= 0); dst = assignment[src]
                    certify_maximum(graph, src, dst)
                    old = json.loads((source_dir / 'counts.json').read_text())
                    assert len(src) <= old['matches']
                    if rep == 'pca':
                        assert len(src) == old['matches'] and (graph != original).nnz == 0
                    save_npz(_paper_location(output / 'graph.npz'), graph)
                    np.savez_compressed(_paper_location(output / 'assignment.npz'), source=src, destination=dst)
                    record = dict(matches=len(src), left_features=len(vectors[i]), right_features=len(vectors[j]),
                                  certified=True, similarity='absolute cosine' if rep == 'pca' else 'signed cosine',
                                  threshold=.7, source_graph=graph_fingerprint(path),
                                  original_edges=original.nnz, retained_edges=graph.nnz)
                    write_json(output / 'counts.json', record)
                    matrix[i,j] = matrix[j,i] = len(src)
                    for a,b in ((i,j),(j,i)):
                        rows.append(dict(model=model, representation=rep, source_split=SPLITS[a],
                                         comparison_split=SPLITS[b], matches=len(src), source_features=len(vectors[a]),
                                         proportion=len(src)/len(vectors[a])))
                    print(name, len(src), 'absolute:', old['matches'], flush=True)
            matrices[model,rep], counts[model,rep] = matrix, len(vectors[0])
    with (SPLIT_OUT / 'counts.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    old_meta = json.loads((SPLIT_SOURCE / 'metadata.json').read_text())
    for ext in ('png', 'pdf'):
        plot_combined_heatmaps(matrices, counts, list(SPLITS), SPLIT_OUT / f'combined_proportions.{ext}', color_max=old_meta['color_max'])
    write_json(SPLIT_OUT / 'metadata.json', dict(metric='maximum-cardinality one-to-one matching',
               similarity=dict(sae='signed cosine', kmeans='signed cosine', pca='absolute cosine'),
               threshold=.7, comparison='>= float32(0.7)', source_files=inputs,
               color_map='viridis', color_min=0, color_max=old_meta['color_max'],
               clipped_cells=sum(int(np.count_nonzero(matrices[k]/counts[k] > old_meta['color_max'])) for k in matrices)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure', choices=['persistent', 'splits', 'both'], default='both')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.figure in ('persistent', 'both'): persistent()
    if args.figure in ('splits', 'both'): splits()


if __name__ == '__main__':
    main()
