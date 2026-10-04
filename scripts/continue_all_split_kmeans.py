"""Continue the ten full-dimensional 16K dictionaries from iteration 20 to 30."""
from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import argparse
import csv
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from scripts import train_kmeans_full_corpus as common
from scripts import train_kmeans_full_corpus_ram as ram
from scripts import match_all_split_kmeans_activations as matching

OUT = common.ROOT / 'full_experiments/results/kmeans_raw_all_splits_iter30'
WIDTH = 16384
ADDITIONAL = 10
TOTAL = 30
SPLITS = matching.SPLITS
FAMILIES = matching.FAMILIES


def code_hashes():
    return dict(runner_sha256=common.digest(__file__), loader_sha256=common.digest(ram.__file__),
                common_sha256=common.digest(common.__file__),
                analysis_code_sha256=common.digest(matching.__file__),
                assignment_code_sha256=common.digest(matching.base.__file__),
                plot_code_sha256=common.digest(_paper_path(__file__).with_name('plot_continued_kmeans_heatmaps.py')))


def immutable_write(path, value):
    if path.exists():
        assert common.read(path) == value, f'Existing record differs: {path}'
    else:
        common.write(path, value)


def prepare(output):
    inventory = []
    for family in FAMILIES:
        for split in SPLITS:
            old_root = common.OUT if split == 'main' else matching.TRAINING / split
            old_plan_path = old_root / family / 'plan.json'
            old_plan = common.read(old_plan_path)
            source = old_root / 'models' / family / f'k{WIDTH}'
            meta = common.read(source / 'metadata.json')
            done = common.read(source / 'completed.json')
            assert done['complete'] and done['plan_id'] == meta['plan_id'] == old_plan['id']
            assert meta['actual_iterations'] == len(meta['objective']) == 20
            assert meta['sample_rows'] == old_plan['rows'] and meta['sample_path'] is None
            assert meta['k'] == WIDTH and meta['training_dim'] == old_plan['dimension']
            assert meta['training_split'] == split and meta['input_space'] == 'raw'
            assert meta['from_scratch'] and meta['all_corpus_rows'] and not meta['faiss_subsampling']
            assert meta['nredo'] == 1 and meta['seed'] == old_plan['seed']
            assert meta['max_points_per_centroid'] * WIDTH >= old_plan['rows']
            assert common.digest(source / 'metadata.json') == done['metadata_sha256']
            assert common.digest(source / 'centroids.npy') == done['centroids_sha256']
            for shard in old_plan['shards']:
                common.stat_matches(shard['path'], shard['size'], shard['mtime_ns'])
            plan = dict(family=family, training_split=split, dimension=old_plan['dimension'],
                        rows=old_plan['rows'], shards=old_plan['shards'], width=WIDTH,
                        datasets=sorted({s['relative_shard'].split('/')[0] for s in old_plan['shards']}),
                        iterations=TOTAL, additional_iterations=ADDITIONAL, source_iterations=20,
                        seed=meta['seed'], input_space='raw', preprocessing='float32 conversion only',
                        source_dir=str(source), source_plan=str(old_plan_path),
                        source_plan_sha256=common.digest(old_plan_path),
                        source_metadata_sha256=done['metadata_sha256'],
                        initial_centroids_sha256=done['centroids_sha256'],
                        max_points_per_centroid=meta['max_points_per_centroid'],
                        num_gpus=meta['num_gpus'], faiss_version=meta['faiss_version'],
                        data_loading='Original shards directly into RAM; no disk corpus copy',
                        initialization='Saved iteration-20 centroids; no new random initialization',
                        **code_hashes())
            plan['id'] = common.object_digest(plan)
            immutable_write(output / split / family / 'plan.json', plan)
            matrix_gib = plan['rows'] * plan['dimension'] * 4 / 2**30
            memory_gib = math.ceil((matrix_gib + 64) / 16) * 16
            if split == 'main':
                memory_gib = 1150 if family == 'gemini' else 1450
            inventory.append(dict(family=family, split=split, rows=plan['rows'],
                                  matrix_gib=matrix_gib, memory_gib=memory_gib,
                                  gpus=plan['num_gpus'], cpus=plan['num_gpus'], plan_id=plan['id']))
    manifest = dict(width=WIDTH, iterations=TOTAL, additional_iterations=ADDITIONAL,
                    source_iterations=20, models=inventory, no_disk_corpus_copy=True,
                    initialization='Continue saved iteration-20 centers', **code_hashes())
    manifest['id'] = common.object_digest(manifest)
    immutable_write(output / 'manifest.json', manifest)
    print(f'Prepared {len(inventory)} continuations: iteration 20 -> 30, width {WIDTH}.', flush=True)


def load_plan(output, family, split):
    plan = common.read(output / split / family / 'plan.json')
    assert plan['family'] == family and plan['training_split'] == split
    assert plan['iterations'] == TOTAL and plan['additional_iterations'] == ADDITIONAL
    assert plan['width'] == WIDTH and plan['source_iterations'] == 20
    assert plan['id'] == common.object_digest({k: v for k, v in plan.items() if k != 'id'})
    for key, value in code_hashes().items():
        assert plan[key] == value, f'Code changed: {key}'
    assert common.digest(plan['source_plan']) == plan['source_plan_sha256']
    source = _paper_path(plan['source_dir'])
    assert common.digest(source / 'metadata.json') == plan['source_metadata_sha256']
    assert common.digest(source / 'centroids.npy') == plan['initial_centroids_sha256']
    return plan


def continue_array(x, initial, source, additional=ADDITIONAL, use_gpu=True):
    """Use native FAISS warm starts and count actual iterations across early stops."""
    import faiss
    assert x.dtype == initial.dtype == np.float32 and x.flags.c_contiguous
    assert x.shape == (source['sample_rows'], source['training_dim'])
    assert initial.shape == (source['k'], source['training_dim']) and np.isfinite(initial).all()
    assert source['max_points_per_centroid'] * source['k'] >= len(x)
    assert source['nredo'] == 1 and additional > 0
    fit = faiss.Kmeans(d=source['training_dim'], k=source['k'], niter=additional,
                       nredo=1, seed=source['seed'], spherical=False, verbose=True,
                       gpu=use_gpu, max_points_per_centroid=source['max_points_per_centroid'])
    objectives, blocks = [], []
    centers = initial.copy()
    while len(objectives) < additional:
        remaining = additional - len(objectives)
        fit.cp.niter = remaining
        fit.train(x, init_centroids=centers)
        current = [float(value) for value in fit.obj]
        assert 0 < len(current) <= remaining and np.isfinite(current).all()
        blocks.append(dict(start_iteration=source['actual_iterations'] + len(objectives),
                           requested_iterations=remaining, actual_iterations=len(current),
                           iteration_stats=fit.iteration_stats))
        objectives.extend(current)
        centers = np.array(fit.centroids, dtype=np.float32, copy=True, order='C')
        assert np.isfinite(centers).all()
    assert len(objectives) == additional
    return centers, objectives, blocks


def finalize(plan, model):
    meta = common.read(model / 'metadata.json')
    assert meta['plan_id'] == plan['id'] and meta['sample_path'] is None
    assert meta['sample_rows'] == plan['rows'] and meta['training_dim'] == plan['dimension']
    assert meta['actual_iterations'] == meta['niter'] == len(meta['objective']) == TOTAL
    assert meta['additional_iterations'] == len(meta['continuation_objective']) == ADDITIONAL
    assert not meta['from_scratch'] and meta['initial_centroids_sha256'] == plan['initial_centroids_sha256']
    assert meta['training_split'] == plan['training_split'] and not meta['faiss_subsampling']
    assert meta['centroids_sha256'] == common.digest(model / 'centroids.npy')
    centers = np.load(_paper_location(model / 'centroids.npy'), mmap_mode='r')
    assert centers.shape == (plan['width'], plan['dimension']) and np.isfinite(centers).all()
    centers._mmap.close()
    common.write(model / 'completed.json', dict(complete=True, plan_id=plan['id'],
                 actual_iterations=TOTAL, additional_iterations=ADDITIONAL,
                 centroids_sha256=meta['centroids_sha256'],
                 metadata_sha256=common.digest(model / 'metadata.json')))


def train(output, family, split, require_gpu=True):
    import faiss
    plan = load_plan(output, family, split)
    model = output / split / 'models' / family / f'k{WIDTH}'
    if (model / 'completed.json').exists():
        done = common.read(model / 'completed.json')
        assert done['complete'] and done['plan_id'] == plan['id']
        assert common.digest(model / 'metadata.json') == done['metadata_sha256']
        assert common.digest(model / 'centroids.npy') == done['centroids_sha256']
        return
    if (model / 'metadata.json').exists():
        finalize(plan, model)
        return
    assert faiss.__version__ == plan['faiss_version']
    if require_gpu:
        assert faiss.get_num_gpus() == plan['num_gpus'], (faiss.get_num_gpus(), plan['num_gpus'])
    faiss.omp_set_num_threads(int(os.environ.get('OMP_NUM_THREADS', str(plan['num_gpus']))))
    source = common.read(_paper_path(plan['source_dir']) / 'metadata.json')
    initial_path = _paper_path(plan['source_dir']) / 'centroids.npy'
    initial = np.ascontiguousarray(np.load(_paper_location(initial_path)), dtype=np.float32)
    common.write(model / 'started.json', dict(plan_id=plan['id'], initialization=str(initial_path),
                 slurm_job=os.environ.get('SLURM_JOB_ID'), started_utc=datetime.now(timezone.utc).isoformat()))
    print(f'{family} {split}: continuing 20 -> 30, {plan["rows"]:,} rows, RAM only', flush=True)
    x = ram.load_corpus(plan, progress=output / split / family / 'loading.json')
    started = time.monotonic()
    centers, objectives, blocks = continue_array(x, initial, source, use_gpu=require_gpu)
    elapsed = time.monotonic() - started
    del x
    np.save(_paper_location(model / 'centroids.npy'), centers)
    index = faiss.IndexFlatL2(plan['dimension'])
    index.add(centers)
    faiss.write_index(index, str(model / 'centroids.index'))
    meta = dict(source)
    meta.update(niter=TOTAL, actual_iterations=TOTAL, source_iterations=20,
                additional_iterations=ADDITIONAL, requested_additional_iterations=ADDITIONAL,
                from_scratch=False, continued_from=plan['source_dir'], initial_centroids=str(initial_path),
                initial_centroids_sha256=plan['initial_centroids_sha256'],
                source_metadata_sha256=plan['source_metadata_sha256'],
                sample_path=None, all_corpus_rows=True, all_split_rows=True, faiss_subsampling=False,
                training_split=split, datasets=plan['datasets'], plan_id=plan['id'],
                use_gpu=require_gpu, num_gpus=faiss.get_num_gpus(), faiss_version=faiss.__version__,
                objective=source['objective'] + objectives, continuation_objective=objectives,
                continuation_blocks=blocks, faiss_stopped_early=False,
                elapsed_seconds=elapsed, source_elapsed_seconds=source['elapsed_seconds'],
                data_loading=plan['data_loading'], source_code_sha256=common.digest(__file__),
                runner_sha256=common.digest(__file__), loader_sha256=common.digest(ram.__file__),
                center_displacement_l2_median=float(np.median(np.linalg.norm(centers-initial, axis=1))),
                centroids_sha256=common.digest(model / 'centroids.npy'))
    common.write(model / 'metadata.json', meta)
    finalize(plan, model)
    print(f'COMPLETE {family} {split}: 30 actual iterations, 10 from saved iteration-20 centers', flush=True)


def prepare_analysis(output, family):
    """Keep exactly the previous row coverage, metric, and assignment implementation."""
    origin = matching.OUT / family / 'manifest.json'
    manifest = common.read(origin)
    assert manifest['analysis_code_sha256'] == common.digest(matching.__file__)
    assert manifest['source_code_sha256'] == common.digest(matching.base.__file__)
    assert manifest['width'] == WIDTH and manifest['splits'] == list(SPLITS)
    models = []
    for split in SPLITS:
        plan = load_plan(output, family, split)
        model = output / split / 'models' / family / f'k{WIDTH}'
        finalize(plan, model)
        meta = common.read(model / 'metadata.json')
        done = common.read(model / 'completed.json')
        evaluation = manifest['evaluations'][split]
        assert evaluation['rows'] == plan['rows']
        assert evaluation['shards'] == [s['relative_shard'] for s in plan['shards']]
        if split == 'main':
            assert manifest['shards'] == plan['shards']
        plan_path = output / split / family / 'plan.json'
        evaluation.update(reference=str(plan_path), reference_sha256=common.digest(plan_path))
        models.append(dict(split=split, path=str(model / 'centroids.npy'),
                           sha256=done['centroids_sha256'], training=meta,
                           metadata_sha256=done['metadata_sha256']))
    for shard in manifest['shards']:
        common.stat_matches(shard['path'], shard['size'], shard['mtime_ns'])
    manifest.pop('id')
    manifest.update(models=models, iterations=TOTAL, additional_iterations=ADDITIONAL,
                    continuation_runner_sha256=common.digest(__file__),
                    original_analysis_manifest=str(origin), original_analysis_manifest_sha256=common.digest(origin))
    manifest['id'] = common.object_digest(manifest)
    immutable_write(output / 'activation_similarity_k16384' / family / 'manifest.json', manifest)
    print(f'Prepared iteration-30 activation analysis for {family}.', flush=True)


def finish(output):
    analysis = output / 'activation_similarity_k16384'
    matching.finish(SimpleNamespace(output=analysis))
    # Label the heatmaps with the completed training iteration.
    from scripts.plot_continued_kmeans_heatmaps import plot
    summary = common.read(analysis / 'summary.json')
    summary['plot'] = plot(analysis, summary['comparisons'], summary['threshold'], TOTAL)
    summary.update(iterations=TOTAL, additional_iterations=ADDITIONAL, continued_from_iterations=20)
    common.write(analysis / 'summary.json', summary)
    records = []
    for family in FAMILIES:
        for split in SPLITS:
            plan = load_plan(output, family, split)
            model = output / split / 'models' / family / f'k{WIDTH}'
            finalize(plan, model)
            meta = common.read(model / 'metadata.json')
            records.append(dict(family=family, training_split=split, width=WIDTH,
                                rows=plan['rows'], iterations=TOTAL, additional_iterations=ADDITIONAL,
                                from_scratch=False, elapsed_seconds=meta['elapsed_seconds'],
                                model_dir=str(model), centroids_sha256=meta['centroids_sha256']))
    with (output / 'training_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    completed = datetime.now(timezone.utc).isoformat()
    common.write(output / 'summary.json', dict(complete=True, iterations=TOTAL,
                 additional_iterations=ADDITIONAL, models=records, completed_utc=completed,
                 activation_analysis=str(analysis), comparisons=40))
    jobs_path = output / 'jobs.json'
    if jobs_path.exists():
        jobs = common.read(jobs_path)
        jobs.update(complete=True, completed_utc=completed)
        common.write(jobs_path, jobs)
    print('COMPLETE: all ten continuations and all 40 iteration-30 activation comparisons.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'train', 'prepare_analysis', 'compute', 'reduce', 'finish'))
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--family', choices=FAMILIES)
    parser.add_argument('--split', choices=SPLITS)
    parser.add_argument('--task', type=int, default=0)
    parser.add_argument('--batch-rows', type=int, default=2048)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.stage in ('prepare', 'finish'):
        globals()[args.stage](args.output)
    elif args.stage == 'train':
        assert args.family and args.split
        train(args.output, args.family, args.split)
    elif args.stage == 'prepare_analysis':
        assert args.family
        prepare_analysis(args.output, args.family)
    else:
        assert args.family
        analysis = args.output / 'activation_similarity_k16384'
        manifest = common.read(analysis / args.family / 'manifest.json')
        assert manifest['iterations'] == TOTAL
        assert manifest['continuation_runner_sha256'] == common.digest(__file__)
        assert manifest['id'] == common.object_digest({k: v for k, v in manifest.items() if k != 'id'})
        args.output = analysis
        args.source = args.split
        if args.stage == 'reduce':
            assert args.split
        getattr(matching, args.stage)(args)


if __name__ == '__main__':
    main()
