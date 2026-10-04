#!/usr/bin/env python
"""Cross-distribution cosine matching without a secondary threshold.

plain: number of source vectors with any cosine > threshold.
maximum_weight: count above-threshold pairs after maximizing total cosine similarity.
maximum_cardinality: maximize the number of disjoint above-threshold pairs.
"""
from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import argparse
import csv
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix, save_npz
from scipy.sparse.csgraph import maximum_bipartite_matching

from split_16384_match_heatmaps import SPLITS, REPRESENTATIONS, model_paths, load_vectors
from project_paths import get_path
from scripts.plot_style import apply_plot_style


@torch.inference_mode()
def similarity_matrix(u, v, device, batch):
    v = v.to(device)
    similarity = np.empty((len(u), len(v)), dtype=np.float32)
    for start in range(0, len(u), batch):
        similarity[start:start + batch] = (u[start:start + batch].to(device) @ v.T).clamp_(-1, 1).cpu().numpy()
    return similarity


def matching_counts(similarity, thresholds):
    # Optimize over ALL edges, including below-threshold and negative edges.
    source, destination = linear_sum_assignment(similarity, maximize=True)
    scores = similarity[source, destination]
    result = {}
    for t in thresholds:
        mask = similarity > t
        result[str(t)] = {
            'plain_left': int(mask.any(axis=1).sum()),
            'plain_right': int(mask.any(axis=0).sum()),
            'maximum_weight': int((scores > t).sum()),
            'edges': int(mask.sum()),
            'assignment_size': len(source),
            'assignment_total_similarity': float(scores.sum(dtype=np.float64)),
        }
    return result, source, destination, scores


def cardinality_counts(similarity, thresholds):
    """Solve an independent maximum-cardinality bipartite matching per cutoff."""
    result, assignments = {}, {}
    for t in thresholds:
        graph = csr_matrix(similarity > t)
        assignment = maximum_bipartite_matching(graph, perm_type='column')
        source = np.flatnonzero(assignment >= 0)
        destination = assignment[source]
        scores = similarity[source, destination]
        result[str(t)] = {
            'plain_left': int(np.count_nonzero(np.diff(graph.indptr))),
            'plain_right': int(len(np.unique(graph.indices))),
            'maximum_cardinality': len(source),
            'edges': graph.nnz,
            'assignment_size': len(source),
            'assignment_total_similarity': float(scores.sum(dtype=np.float64)),
        }
        assignments[t] = (source, destination, scores, graph)
    return result, assignments


def plot(matrices, counts, out, threshold, metric, proportion):
    apply_plot_style()
    fig, axes = plt.subplots(2, 3, figsize=(17, 11), sharex=True, sharey=True)
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('#f2f2f2')
    values = {key: matrix / np.array(counts[key])[:, None] if proportion else matrix
              for key, matrix in matrices.items()}
    vmax = 1 if proportion else max(1, max(x.max() for x in values.values()))
    for row, model in enumerate(('gemini', 'nemotron')):
        for col, rep in enumerate(REPRESENTATIONS):
            ax = axes[row, col]
            array = values[model, rep]
            im = ax.imshow(np.ma.array(array, mask=np.eye(5, dtype=bool)), cmap=cmap, vmin=0, vmax=vmax)
            ax.set_title(f'{model.title()} — ' + {'sae': 'SAE', 'kmeans': 'KMeans', 'pca': 'PCA'}[rep], fontsize=18 if proportion else 17)
            ax.set_xticks(range(5), SPLITS, rotation=35, ha='right')
            ax.set_yticks(range(5), SPLITS)
            ax.tick_params(axis="both", labelsize=18 if proportion else 14)
            for i in range(5):
                for j in range(5):
                    if i == j:
                        continue
                    val = array[i, j]
                    r, g, b, _ = im.cmap(im.norm(val))
                    color = 'black' if .2126*r + .7152*g + .0722*b > .55 else 'white'
                    ax.text(j, i, f'{val:.2f}' if proportion else f'{val:,}', ha='center', va='center', color=color, fontsize=18 if proportion else 10)
    title = {
        'plain': 'Features with any above-threshold partner',
        'maximum_weight': 'Above-threshold pairs in maximum-total-similarity matching',
        'maximum_cardinality': 'Maximum number of one-to-one pairs above threshold',
    }[metric]
    fig.suptitle(f'{title} | cosine similarity > {threshold:g} | no secondary threshold')
    fig.subplots_adjust(left=.12, right=.88, bottom=.14, top=.90, wspace=.08, hspace=.15)
    cax = fig.add_axes([.91, .22, .018, .56])
    colorbar = fig.colorbar(im, cax=cax)
    colorbar.set_label('matching feature proportion' if proportion else 'Number of features / matched pairs', fontsize=18 if proportion else 17)
    colorbar.ax.tick_params(labelsize=18 if proportion else 14)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(out.with_suffix('.' + ext), dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models-dir', type=Path, default=_paper_path(get_path('models_dir')))
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--objective', choices=('maximum_weight', 'maximum_cardinality'), default='maximum_weight')
    parser.add_argument('--thresholds', nargs='+', type=float, default=[.7, .8])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--require-cache', action='store_true', help='Only assemble completed pair results; fail if any are missing.')
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--task-index', type=int, choices=range(60), help='Compute one of the 60 pairs; a later run assembles cached results.')
    args = parser.parse_args()
    if not all(-1 <= t <= 1 for t in args.thresholds):
        parser.error('Thresholds must be within [-1, 1].')
    if args.out_dir is None:
        args.out_dir = _paper_path('full_experiments/results/split_matches') / ('plain_and_' + args.objective)
    metrics = ('plain', args.objective)
    torch.set_num_threads(4)
    # Full float32 products avoid reduced-precision ambiguity at the cutoff.
    torch.backends.cuda.matmul.allow_tf32 = False
    args.out_dir.mkdir(parents=True, exist_ok=True)
    matrices = {(t, m): {} for t in args.thresholds for m in metrics}
    counts = {}
    rows = []
    for rep in REPRESENTATIONS:
        for model in ('gemini', 'nemotron'):
            group_index = list(REPRESENTATIONS).index(rep) * 2 + ('gemini', 'nemotron').index(model)
            if args.task_index is not None and args.task_index // 10 != group_index:
                continue
            paths = model_paths(args.models_dir, model, rep)
            vectors = [load_vectors(path, rep) for _, path in paths]
            counts[model, rep] = [len(v) for v in vectors]
            for group in matrices.values():
                group[model, rep] = np.zeros((5, 5), dtype=np.int64)
            pair_index = -1
            for i in range(5):
                for j in range(i + 1, 5):
                    pair_index += 1
                    if args.task_index is not None and args.task_index % 10 != pair_index:
                        continue
                    start = time.time()
                    cache = args.out_dir / 'pairs' / f'{model}_{rep}_{SPLITS[i]}_to_{SPLITS[j]}'
                    cache.mkdir(parents=True, exist_ok=True)
                    result_path = cache / 'counts.json'
                    if result_path.exists():
                        result = json.loads(result_path.read_text())
                    else:
                        if args.require_cache:
                            raise FileNotFoundError(result_path)
                        similarity = similarity_matrix(vectors[i], vectors[j], args.device, args.batch)
                        if args.objective == 'maximum_weight':
                            result, src, dst, scores = matching_counts(similarity, args.thresholds)
                            np.savez_compressed(_paper_location(cache / 'maximum_weight_assignment.npz'), source=src, destination=dst, similarity=scores)
                        else:
                            result, assignments = cardinality_counts(similarity, args.thresholds)
                            for t, (src, dst, scores, graph) in assignments.items():
                                np.savez_compressed(_paper_location(cache / f'maximum_cardinality_{t:g}_assignment.npz'), source=src, destination=dst, similarity=scores)
                                save_npz(_paper_location(cache / f'threshold_{t:g}_graph.npz'), graph)
                        del similarity
                        result_path.write_text(json.dumps(result, indent=2) + '\n')
                    for t in args.thresholds:
                        r = result[str(t)]
                        for metric in metrics:
                            left, right = (r['plain_left'], r['plain_right']) if metric == 'plain' else (r[metric], r[metric])
                            matrix = matrices[t, metric][model, rep]
                            matrix[i, j], matrix[j, i] = left, right
                            for a, b, n in ((i, j, left), (j, i, right)):
                                rows.append(dict(model=model, representation=rep, threshold=t, metric=metric, source_split=SPLITS[a], comparison_split=SPLITS[b], matches=n, source_features=len(vectors[a]), proportion=n/len(vectors[a])))
                    print(f'{model} {rep} {SPLITS[i]} / {SPLITS[j]}: {result} ({time.time()-start:.1f}s)', flush=True)
    if args.task_index is not None:
        return
    with (args.out_dir / 'counts.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for (t, metric), group in matrices.items():
        directory = args.out_dir / f'threshold_{t:g}' / metric
        directory.mkdir(parents=True, exist_ok=True)
        for (model, rep), matrix in group.items():
            with (directory / f'{model}_{rep}_matrix.csv').open('w') as f:
                writer = csv.writer(f)
                writer.writerow(['split', *SPLITS])
                for i, label in enumerate(SPLITS):
                    writer.writerow([label, *['' if i == j else int(x) for j, x in enumerate(matrix[i])]])
        for proportion in (False, True):
            plot(group, counts, directory / ('combined_proportions' if proportion else 'combined_counts'), t, metric, proportion)
    print('Done', flush=True)


if __name__ == '__main__':
    main()
