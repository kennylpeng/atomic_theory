"""Continue full-corpus 4K/131K KMeans from 20 to 100 with one RAM load per family."""
from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import argparse
import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import time

import numpy as np

from scripts import continue_all_split_kmeans as kernel
from scripts import train_kmeans_full_corpus as common
from scripts import train_kmeans_full_corpus_ram as ram

OUT = common.ROOT / 'full_experiments/results/kmeans_raw_full_corpus_iter100'
SOURCE = common.OUT
WIDTHS = (4096, 131072)
FAMILIES = common.FAMILIES
SOURCE_ITERATIONS = 20
TOTAL = 100
CHECKPOINTS = (80, 90, 100)
continue_array = kernel.continue_array


def code_hashes():
    return dict(runner_sha256=common.digest(__file__), loader_sha256=common.digest(ram.__file__),
                continuation_code_sha256=common.digest(kernel.__file__),
                common_sha256=common.digest(common.__file__))


def immutable_write(path, value):
    if path.exists():
        assert common.read(path) == value, f'Existing record differs: {path}'
    else:
        common.write(path, value)


def prepare(output):
    inventory = []
    for family in FAMILIES:
        source_plan_path = SOURCE / family / 'plan.json'
        old = common.read(source_plan_path)
        for shard in old['shards']:
            common.stat_matches(shard['path'], shard['size'], shard['mtime_ns'])
        for width in WIDTHS:
            source = SOURCE / 'models' / family / f'k{width}'
            meta = common.read(source / 'metadata.json')
            done = common.read(source / 'completed.json')
            assert done['complete'] and done['plan_id'] == meta['plan_id'] == old['id']
            assert done['actual_iterations'] == meta['actual_iterations'] == len(meta['objective']) == SOURCE_ITERATIONS
            assert meta['sample_rows'] == old['rows'] and meta['sample_path'] is None
            assert meta['k'] == width and meta['training_dim'] == old['dimension']
            assert meta['training_split'] == 'main' and meta['input_space'] == 'raw'
            assert meta['all_corpus_rows'] and not meta['faiss_subsampling']
            assert meta['nredo'] == 1 and meta['seed'] == old['seed']
            assert meta['max_points_per_centroid'] * width >= old['rows']
            assert common.digest(source / 'metadata.json') == done['metadata_sha256']
            assert common.digest(source / 'centroids.npy') == done['centroids_sha256']
            plan = dict(family=family, training_split='main', dimension=old['dimension'],
                        rows=old['rows'], shards=old['shards'], width=width,
                        datasets=sorted({s['relative_shard'].split('/')[0] for s in old['shards']}),
                        iterations=TOTAL, source_iterations=SOURCE_ITERATIONS,
                        additional_iterations=TOTAL-SOURCE_ITERATIONS, checkpoints=list(CHECKPOINTS),
                        seed=meta['seed'], input_space='raw', preprocessing='float32 conversion only',
                        source_dir=str(source), source_plan=str(source_plan_path),
                        source_plan_sha256=common.digest(source_plan_path),
                        source_metadata_sha256=done['metadata_sha256'],
                        initial_centroids_sha256=done['centroids_sha256'],
                        max_points_per_centroid=meta['max_points_per_centroid'],
                        num_gpus=meta['num_gpus'], faiss_version=meta['faiss_version'],
                        data_loading='Original shards directly into one shared RAM matrix per family; no disk corpus copy',
                        initialization='Saved iteration-20 centers for the same width; no random restart',
                        **code_hashes())
            plan['id'] = common.object_digest(plan)
            immutable_write(output / family / 'plans' / f'k{width}.json', plan)
            inventory.append(dict(family=family, width=width, rows=old['rows'],
                                  dimension=old['dimension'], plan_id=plan['id']))
    manifest = dict(source_iterations=SOURCE_ITERATIONS, iterations=TOTAL,
                    additional_iterations=TOTAL-SOURCE_ITERATIONS, widths=list(WIDTHS),
                    families=list(FAMILIES), checkpoints=list(CHECKPOINTS), models=inventory,
                    one_ram_load_per_family=True, no_disk_corpus_copy=True, **code_hashes())
    manifest['id'] = common.object_digest(manifest)
    immutable_write(output / 'manifest.json', manifest)
    print('Prepared four full-corpus continuations, 20 -> 100, with 80/90/100 checkpoints.', flush=True)


def load_plan(output, family, width):
    plan = common.read(output / family / 'plans' / f'k{width}.json')
    assert plan['family'] == family and plan['width'] == width and width in WIDTHS
    assert plan['iterations'] == TOTAL and plan['source_iterations'] == SOURCE_ITERATIONS
    assert plan['training_split'] == 'main' and plan['checkpoints'] == list(CHECKPOINTS)
    assert plan['id'] == common.object_digest({k: v for k, v in plan.items() if k != 'id'})
    for key, value in code_hashes().items():
        assert plan[key] == value, f'Changed code: {key}'
    assert common.digest(plan['source_plan']) == plan['source_plan_sha256']
    source = _paper_path(plan['source_dir'])
    assert common.digest(source / 'metadata.json') == plan['source_metadata_sha256']
    assert common.digest(source / 'centroids.npy') == plan['initial_centroids_sha256']
    return plan


def model_path(output, family, width, iteration):
    assert iteration in CHECKPOINTS
    base = output / 'models' / family / f'k{width}'
    return base if iteration == TOTAL else base / 'checkpoints' / f'iter{iteration:03d}'


def finalize_checkpoint(plan, model, iteration, parent, parent_meta):
    meta = common.read(model / 'metadata.json')
    assert meta['plan_id'] == plan['id'] and meta['sample_path'] is None
    assert meta['actual_iterations'] == meta['niter'] == len(meta['objective']) == iteration
    assert meta['additional_iterations'] == len(meta['continuation_objective']) == iteration-SOURCE_ITERATIONS
    assert meta['source_iterations'] == SOURCE_ITERATIONS and meta['continued_from'] == plan['source_dir']
    assert meta['initial_centroids_sha256'] == plan['initial_centroids_sha256']
    assert meta['checkpoint_parent_dir'] == str(parent)
    assert meta['checkpoint_parent_iteration'] == parent_meta['actual_iterations']
    assert meta['checkpoint_parent_metadata_sha256'] == common.digest(parent / 'metadata.json')
    assert meta['checkpoint_parent_centroids_sha256'] == common.digest(parent / 'centroids.npy')
    assert meta['objective'][:parent_meta['actual_iterations']] == parent_meta['objective']
    assert meta['segment_iterations'] == iteration-parent_meta['actual_iterations'] == len(meta['segment_objective'])
    assert meta['continuation_objective'] == meta['objective'][SOURCE_ITERATIONS:]
    assert meta['sample_rows'] == plan['rows'] and meta['training_dim'] == plan['dimension']
    assert meta['training_split'] == plan['training_split'] and meta['input_space'] == 'raw'
    assert not meta['from_scratch'] and not meta['faiss_subsampling'] and meta['all_split_rows']
    assert meta['centroids_sha256'] == common.digest(model / 'centroids.npy')
    assert meta['centroids_index_sha256'] == common.digest(model / 'centroids.index')
    cursor = SOURCE_ITERATIONS
    for block in meta['continuation_blocks']:
        assert block['start_iteration'] == cursor
        cursor += block['actual_iterations']
    assert cursor == iteration
    centers = np.load(_paper_location(model / 'centroids.npy'), mmap_mode='r')
    assert centers.shape == (plan['width'], plan['dimension']) and np.isfinite(centers).all()
    centers._mmap.close()
    done = dict(complete=True, plan_id=plan['id'], actual_iterations=iteration,
                additional_iterations=iteration-SOURCE_ITERATIONS,
                centroids_sha256=meta['centroids_sha256'],
                metadata_sha256=common.digest(model / 'metadata.json'),
                centroids_index_sha256=meta['centroids_index_sha256'])
    immutable_write(model / 'completed.json', done)
    return meta


def saved_checkpoints(output, plan):
    parent = _paper_path(plan['source_dir'])
    parent_meta = common.read(parent / 'metadata.json')
    saved, missing = [], False
    for iteration in CHECKPOINTS:
        model = model_path(output, plan['family'], plan['width'], iteration)
        if not (model / 'metadata.json').exists():
            assert not (model / 'completed.json').exists(), model
            missing = True
            continue
        assert not missing, f'Checkpoint chain has a gap: {model}'
        meta = finalize_checkpoint(plan, model, iteration, parent, parent_meta)
        saved.append((iteration, model, meta))
        parent, parent_meta = model, meta
    return saved


def train_loaded(output, plan, x, saved, require_gpu=True):
    import faiss
    family, width, split = plan['family'], plan['width'], 'main'
    origin = _paper_path(plan['source_dir'])
    source = common.read(origin / 'metadata.json')
    parent, current = (saved[-1][1], saved[-1][2]) if saved else (origin, source)
    centers = np.ascontiguousarray(np.load(_paper_location(parent / 'centroids.npy')), dtype=np.float32)
    cumulative_seconds = current['elapsed_seconds'] if saved else 0.
    blocks = list(current['continuation_blocks']) if saved else []
    pending = [iteration for iteration in CHECKPOINTS if iteration > current['actual_iterations']]
    progress_path = output / family / f'k{width}' / 'checkpoint_progress.json'
    progress = dict(plan_id=plan['id'], slurm_job=os.environ.get('SLURM_JOB_ID'),
                    last_saved_iteration=current['actual_iterations'], pending=pending,
                    started_utc=datetime.now(timezone.utc).isoformat())
    common.write(progress_path, dict(progress, stage='training'))
    print(f'{family} k{width}: resume {current["actual_iterations"]} -> {TOTAL}; '
          f'checkpoints={pending}; {plan["rows"]:,} rows; one RAM load', flush=True)
    for iteration in pending:
        model = model_path(output, family, width, iteration)
        parent_hash = common.digest(parent / 'centroids.npy')
        parent_metadata_hash = common.digest(parent / 'metadata.json')
        segment_iterations = iteration-current['actual_iterations']
        common.write(progress_path, dict(progress, stage='training', target_iteration=iteration))
        common.write(model / 'started.json', dict(plan_id=plan['id'], target_iteration=iteration,
                     initial_centroids=str(parent / 'centroids.npy'), initial_centroids_sha256=parent_hash,
                     slurm_job=os.environ.get('SLURM_JOB_ID')))
        started = time.monotonic()
        centers, objectives, new_blocks = continue_array(
            x, centers, current, additional=segment_iterations, use_gpu=require_gpu)
        elapsed = time.monotonic()-started
        cumulative_seconds += elapsed
        blocks.extend(new_blocks)
        temporary = model / 'centroids.partial.npy'
        np.save(_paper_location(temporary), centers)
        temporary.replace(model / 'centroids.npy')
        index = faiss.IndexFlatL2(plan['dimension'])
        index.add(centers)
        temporary_index = model / 'centroids.partial.index'
        faiss.write_index(index, str(temporary_index))
        temporary_index.replace(model / 'centroids.index')
        history = current['objective'] + objectives
        meta = dict(source)
        meta.update(niter=iteration, actual_iterations=iteration, source_iterations=SOURCE_ITERATIONS,
                    additional_iterations=iteration-SOURCE_ITERATIONS,
                    requested_additional_iterations=iteration-SOURCE_ITERATIONS,
                    from_scratch=False, continued_from=str(origin),
                    initial_centroids=str(origin / 'centroids.npy'),
                    initial_centroids_sha256=plan['initial_centroids_sha256'],
                    source_metadata_sha256=plan['source_metadata_sha256'],
                    checkpoint_parent_dir=str(parent), checkpoint_parent_iteration=current['actual_iterations'],
                    checkpoint_parent_centroids_sha256=parent_hash,
                    checkpoint_parent_metadata_sha256=parent_metadata_hash,
                    segment_iterations=segment_iterations, segment_objective=objectives,
                    segment_elapsed_seconds=elapsed, elapsed_seconds=cumulative_seconds,
                    source_elapsed_seconds=source['elapsed_seconds'],
                    sample_path=None, all_corpus_rows=True, all_split_rows=True, faiss_subsampling=False,
                    training_split=split, datasets=plan['datasets'], plan_id=plan['id'],
                    use_gpu=require_gpu, num_gpus=faiss.get_num_gpus(), faiss_version=faiss.__version__,
                    objective=history, continuation_objective=history[SOURCE_ITERATIONS:],
                    continuation_blocks=list(blocks), faiss_stopped_early=False,
                    data_loading=plan['data_loading'], source_code_sha256=common.digest(__file__),
                    runner_sha256=common.digest(__file__), loader_sha256=common.digest(ram.__file__),
                    centroids_sha256=common.digest(model / 'centroids.npy'),
                    centroids_index_sha256=common.digest(model / 'centroids.index'),
                    saved_utc=datetime.now(timezone.utc).isoformat())
        # The source run's displacement diagnostic describes a different interval.
        meta.pop('center_displacement_l2_median', None)
        common.write(model / 'metadata.json', meta)
        finalize_checkpoint(plan, model, iteration, parent, current)
        parent, current = model, meta
        progress.update(last_saved_iteration=iteration,
                        pending=[target for target in CHECKPOINTS if target > iteration])
        common.write(progress_path, dict(progress, stage='complete' if iteration == TOTAL else 'checkpoint_saved'))
        print(f'SAVED {family} k{width}: iteration {iteration}, {segment_iterations} additional '
              f'iterations in {elapsed:.1f}s -> {model}', flush=True)


def train(output, family, require_gpu=True):
    import faiss
    plans = [load_plan(output, family, width) for width in WIDTHS]
    reference = plans[0]
    for plan in plans[1:]:
        for key in ('shards', 'rows', 'dimension', 'num_gpus', 'faiss_version'):
            assert plan[key] == reference[key], f'Widths cannot share input: {key}'
    pending = []
    for plan in plans:
        saved = saved_checkpoints(output, plan)
        if len(saved) < len(CHECKPOINTS):
            pending.append((plan, saved))
    if not pending:
        print(f'Already complete: {family}, both widths and all checkpoints.', flush=True)
        return
    assert faiss.__version__ == reference['faiss_version']
    if require_gpu:
        assert faiss.get_num_gpus() == reference['num_gpus']
    faiss.omp_set_num_threads(int(os.environ.get('OMP_NUM_THREADS', str(reference['num_gpus']))))
    common.write(output / family / 'started.json', dict(
        widths=[plan['width'] for plan, _ in pending], one_ram_load=True,
        slurm_job=os.environ.get('SLURM_JOB_ID'), started_utc=datetime.now(timezone.utc).isoformat()))
    print(f'{family}: load {reference["rows"]:,} rows once for widths '
          f'{[plan["width"] for plan, _ in pending]}', flush=True)
    x = ram.load_corpus(reference, progress=output / family / 'loading.json')
    for plan, saved in pending:
        train_loaded(output, plan, x, saved, require_gpu=require_gpu)
    del x
    common.write(output / family / 'completed.json', dict(
        complete=True, widths=list(WIDTHS), checkpoints=list(CHECKPOINTS),
        completed_utc=datetime.now(timezone.utc).isoformat()))


def finish(output):
    records = []
    for family in FAMILIES:
        for width in WIDTHS:
            plan = load_plan(output, family, width)
            saved = saved_checkpoints(output, plan)
            assert [iteration for iteration, _, _ in saved] == list(CHECKPOINTS)
            for iteration, model, meta in saved:
                records.append(dict(family=family, training_split='main', width=width,
                    rows=plan['rows'], iterations=iteration, additional_iterations=iteration-SOURCE_ITERATIONS,
                    segment_iterations=meta['segment_iterations'], segment_elapsed_seconds=meta['segment_elapsed_seconds'],
                    cumulative_fit_seconds=meta['elapsed_seconds'], model_dir=str(model),
                    centroids_sha256=meta['centroids_sha256']))
    with (output / 'training_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    completed = datetime.now(timezone.utc).isoformat()
    common.write(output / 'summary.json', dict(complete=True, source_iterations=SOURCE_ITERATIONS,
        iterations=TOTAL, widths=list(WIDTHS), checkpoints=list(CHECKPOINTS), models=records,
        saved_checkpoints=len(records), completed_utc=completed))
    if (output / 'jobs.json').exists():
        jobs = common.read(output / 'jobs.json')
        jobs.update(complete=True, completed_utc=completed)
        common.write(output / 'jobs.json', jobs)
    print('COMPLETE: both widths and both families at 100, with twelve verified 80/90/100 checkpoints.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'train', 'finish'))
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--family', choices=FAMILIES)
    args = parser.parse_args()
    if args.stage == 'train':
        assert args.family
        train(args.output, args.family)
    else:
        globals()[args.stage](args.output)


if __name__ == '__main__':
    main()
