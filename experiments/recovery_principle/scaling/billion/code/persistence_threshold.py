"""Run exhaustive persistence analysis at a specified threshold.

Each threshold writes to a separate result directory.
"""
from project_paths import resource_path as _paper_path, resource_location as _paper_location
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz
import persistence as base

_BASE_CODE_HASH = base.code_hash
BASELINE = base.OUT


def configure(threshold):
    if not 0.7 < threshold < 1:
        raise ValueError('Additional thresholds must lie strictly between 0.7 and 1; original results are preserved.')
    base.THRESHOLD = threshold
    base.OUT = base.ROOT / f'results/persistence_1024M_t{threshold:g}'
    def version():
        provenance = f'{threshold:.17g}:{_BASE_CODE_HASH()}'.encode()
        return hashlib.sha256(provenance + _paper_path(__file__).read_bytes()).hexdigest()
    base.code_hash = version


def compare_baseline():
    """Verify identical models/rows and nested graphs/counts against t=0.7."""
    manifest = json.loads((base.OUT/'manifest.json').read_text())
    previous_manifest = json.loads((BASELINE/'manifest.json').read_text())
    assert manifest['jobs'] == previous_manifest['jobs']
    comparisons = []
    graph_count = 0
    for job in manifest['jobs']:
        directory = base.OUT/job['name']
        previous = BASELINE/job['name']
        current = json.loads((directory/'summary.json').read_text())
        old = json.loads((previous/'summary.json').read_text())
        assert current['threshold'] == base.THRESHOLD > old['threshold'] == 0.7
        for key in ['checkpoints','model_state_sha256','evaluation_rows_sha256',
                    'evaluation_examples','evaluation_seed','evaluation_batch_size','constant_features']:
            assert current[key] == old[key], (job['name'], key)
        for diagnostic in current['diagnostics']:
            prefix = 'cosine' if diagnostic['metric']=='decoder_cosine' else 'pearson'
            name = f"{prefix}_{diagnostic['source']}_to_{diagnostic['target']}.npz"
            graph = load_npz(_paper_location(directory/name))
            old_graph = load_npz(_paper_location(previous/name))
            assert graph.shape == old_graph.shape
            assert graph.multiply(old_graph).nnz == graph.nnz, (job['name'], name)
            graph_count += 1
        old_rows = {(r['metric'],r['width']):r for r in old['rows']}
        for row in current['rows']:
            old_count = old_rows[row['metric'],row['width']]['persistent_count']
            # Adding edges need not increase the independently selected intersection.
            comparisons.append(dict(group=job['name'],metric=row['metric'],width=row['width'],
                                    baseline_count=old_count,count=row['persistent_count'],
                                    change=row['persistent_count']-old_count))
    assert graph_count == 2340 and len(comparisons) == 780
    base.atomic_json(base.OUT/'threshold_comparison.json',dict(
        complete=True,baseline_threshold=0.7,threshold=base.THRESHOLD,
        same_checkpoints_and_evaluation_rows=True,nested_graphs=graph_count,
        count_comparisons=len(comparisons),intersection_monotonicity_required=False,results=comparisons))
    print('Validated all 2,340 nested graphs and 780 independently computed intersection counts.',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run','report'])
    parser.add_argument('--threshold',type=float,required=True)
    parser.add_argument('--index',type=int,default=0)
    parser.add_argument('--device',default='cuda')
    args = parser.parse_args()
    configure(args.threshold)
    if args.stage == 'prepare':
        base.prepare()
    elif args.stage == 'run':
        base.run(args.index,args.device)
    else:
        # Import after configuring the analysis constants and provenance hash.
        import plot_persistence, verify_persistence_optima
        plot_persistence.OUT = base.OUT
        plot_persistence.THRESHOLD = base.THRESHOLD
        verify_persistence_optima.OUT = base.OUT
        report = plot_persistence.report
        verify = verify_persistence_optima.verify
        compare_baseline()
        report()
        verify()


if __name__ == '__main__':
    main()
