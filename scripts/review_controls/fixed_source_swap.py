"""Swap only the source to k=128, retaining every original larger-width target."""
from project_paths import resource_path as _paper_path, resource_location as _paper_location
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.csgraph import maximum_bipartite_matching

ROOT = _paper_path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.review_controls.directions import (
    Experiment, FIXED, BASELINE, OUT, SCHEDULE, certify_maximum, dump, mainpath, sha, trainpath,
)


def run(family):
    experiment = Experiment(family)
    experiment.folder = OUT / 'fixed_source_swap' / family
    experiment.folder.mkdir(parents=True, exist_ok=True)
    original = json.loads((BASELINE / family / 'summary.json').read_text())
    audits = []
    for width, seed in FIXED[family].items():
        source = trainpath(family, width, 128, seed)
        baseline = mainpath(family, width)
        experiment.load(source)
        experiment.load(baseline)
        configs = [experiment.metadata[str(path)]['config'] for path in (baseline, source)]
        assert configs[0]['top_k'] == SCHEDULE[width]
        assert configs[1]['top_k'] == 128
        keys = ('seed', 'act_size', 'dict_size', 'num_tokens', 'aux_penalty', 'top_k_aux',
                'batch_size', 'lr', 'beta1', 'beta2', 'input_unit_norm', 'n_batches_to_dead')
        for key in keys:
            assert configs[0].get(key) == configs[1].get(key), (family, width, key)
        audits.append(dict(width=width, original_k=SCHEDULE[width], replacement_k=128,
                           matched_settings={key: configs[0].get(key) for key in keys}))
        graph = experiment.graph(baseline, source, f'cross_sparsity_same_width_{width}')
        assignment = maximum_bipartite_matching(graph, perm_type='column')
        selected = np.flatnonzero(assignment >= 0)
        destination = assignment[selected]
        certify_maximum(graph, selected, destination)
        np.savez_compressed(_paper_location(experiment.folder / f'cross_sparsity_same_width_{width}_assignment.npz'),
                            source=selected, destination=destination)
        experiment.rows.append(dict(family=family, experiment='cross_sparsity_same_width', width=width,
                                    count=len(selected), proportion=len(selected)/width, edges=graph.nnz,
                                    certificate='matching_equal_vertex_cover'))
        targets = [target for target in SCHEDULE if target > width]
        graphs = [experiment.graph(source, mainpath(family, target), f'fixed_source_{width}_{target}')
                  for target in targets]
        experiment.persistence('fixed128_source_full_schedule', width, targets, graphs)
        old = next(row for row in original if row['width'] == width)
        experiment.rows.append(dict(family=family, experiment='original_source_full_schedule', width=width,
                                    target_widths=targets, count=old['persistent_count'], proportion=old['proportion']))
    dump(experiment.folder / 'results.json', experiment.rows)
    dump(experiment.folder / 'configuration_audit.json', audits)
    dump(experiment.folder / 'COMPLETE.json', dict(
        family=family, complete=True, slurm_job=os.environ.get('SLURM_JOB_ID'),
        torch_version=torch.__version__, gpu=torch.cuda.get_device_name(),
        protocol='Only source replaced by fixed k=128; all larger original nine-sweep targets retained.',
        scripts={str(path.relative_to(ROOT)): sha(path) for path in
                 [_paper_path(__file__), ROOT/'scripts/review_controls/directions.py', ROOT/'scripts/stability/persistent_matching.py']},
    ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', choices=['gemini', 'nemotron'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()
    run(args.family)
