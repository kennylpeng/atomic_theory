#!/usr/bin/env python3
"""Compute the fixed-k=128 Cities and GBIF probes for Figure 2(c)."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from project_paths import resource_path

from run_prefix_recovery_k128 import ROOT, FIXED_SEEDS, checkpoint_path, probe, sha, validate_search
from hierarchy_probe_config import HIERARCHY_DATA_DIR, GBIF_METADATA_DIR

OUT = ROOT / 'full_experiments/results/hierarchy_recovery_k128'
CACHE = resource_path('/resources/scratch_dir/hierarchy_recovery_k128')
DATASETS = ['gbif', 'geonames_country_questions', 'geonames_continent_questions']
TASKS = [(dataset, family, width) for dataset in DATASETS for family in FIXED_SEEDS for width in FIXED_SEEDS[family]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=int, choices=range(len(TASKS)))
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    validate_search()
    if args.validate:
        for dataset in DATASETS:
            model = 'gemini_m512_k32'
            probe('gemini', 512, 32, HIERARCHY_DATA_DIR / dataset / f'sparse_{model}.npz',
                  OUT / 'validation' / dataset / model,
                  compare=HIERARCHY_DATA_DIR / 'probe_results' / dataset / model / 'summary.tsv', dataset=dataset)
        (OUT / 'validation/COMPLETE.json').write_text(json.dumps(dict(datasets=DATASETS, exact_reference_agreement=True), indent=2)+'\n')
        return
    if args.task is None:
        parser.error('--task or --validate is required')
    dataset, family, width = TASKS[args.task]
    seed = FIXED_SEEDS[family][width]
    checkpoint = checkpoint_path(family, width, 128, seed)
    prefix_manifest = ROOT / 'full_experiments/results/prefix_recovery_k128/wordfreq' / f'{family}_m{width}_k128/checkpoint.json'
    verified = json.loads(prefix_manifest.read_text())
    assert resource_path(verified['path']) == checkpoint and verified['sha256'] == sha(checkpoint)
    config = verified['config']
    assert config['top_k'] == 128 and config['dict_size'] == width and not config.get('input_unit_norm', False)
    sparse = CACHE / dataset / f'sparse_{family}_m{width}_k128.npz'
    subprocess.run([sys.executable, str(ROOT / 'hierarchy_data_prep/02_compute_sparse_activations.py'),
        '--model-path', str(checkpoint), '--embeddings', str(HIERARCHY_DATA_DIR / dataset / f'embeddings_{family}.npy'),
        '--rows', str(HIERARCHY_DATA_DIR / dataset / 'rows.csv'), '--output', str(sparse),
        '--top-k', '128', '--batch-size', '2048', '--device', 'cuda', '--dtype', 'float32'], check=True)
    folder = OUT / dataset / f'{family}_m{width}_k128'
    probe(family, width, 128, sparse, folder, dataset=dataset)
    manifest = dict(path=str(checkpoint), sha256=verified['sha256'], config=config,
                    sparse=str(sparse), sparse_sha256=sha(sparse), slurm_job=os.environ.get('SLURM_JOB_ID'))
    if dataset == 'gbif':
        manifest['taxonomy_sources'] = {str(p): sha(p) for p in [
            GBIF_METADATA_DIR / 'category_threshold_f1_d131072/categories_min100_species.tsv',
            GBIF_METADATA_DIR / 'category_threshold_f1_d131072/species_common_name_pairs.csv']}
    (folder / 'checkpoint.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'COMPLETE: {folder}', flush=True)


if __name__ == '__main__':
    main()
