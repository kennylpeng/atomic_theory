"""Run the existing FAISS trainer on all original shards, loaded directly into RAM."""
from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import argparse
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import time

import numpy as np

from scripts import fit_embedding_kmeans as previous
from scripts import train_kmeans_full_corpus as common


def load_corpus(plan, batch_rows=8192, progress=None):
    """Allocate one RAM matrix; read each original shard without creating a data file."""
    x = np.empty((plan['rows'], plan['dimension']), dtype=np.float32)
    written = 0
    started = time.monotonic()
    for index, shard in enumerate(plan['shards']):
        common.stat_matches(shard['path'], shard['size'], shard['mtime_ns'])
        assert shard['offset'] == written
        source = np.load(_paper_location(shard['path']), mmap_mode='r', allow_pickle=False)
        try:
            assert source.shape == (shard['rows'], plan['dimension'])
            for start in range(0, len(source), batch_rows):
                stop = min(start + batch_rows, len(source))
                destination = x[written+start:written+stop]
                destination[:] = source[start:stop]
                assert np.isfinite(destination).all(), shard['path']
        finally:
            source._mmap.close()
        common.stat_matches(shard['path'], shard['size'], shard['mtime_ns'])
        written += shard['rows']
        if (index+1) % 10 == 0 or index+1 == len(plan['shards']):
            state = dict(rows_loaded=written, total_rows=plan['rows'], shards_loaded=index+1,
                         total_shards=len(plan['shards']), elapsed_seconds=time.monotonic()-started,
                         storage='RAM only', slurm_job=os.environ.get('SLURM_JOB_ID'))
            print(f'{plan["family"]}: loaded {written:,}/{plan["rows"]:,} rows into RAM; '
                  f'{state["elapsed_seconds"]:.1f}s', flush=True)
            if progress is not None:
                common.write(progress, state)
    assert written == plan['rows'] and x.flags.c_contiguous
    return x


def train(plan, output, require_gpu=True):
    import faiss
    assert not require_gpu or faiss.get_num_gpus() > 0
    faiss.omp_set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '16')))
    family = plan['family']
    folder = output / 'models' / family
    pending = []
    for width in reversed(plan['widths']):
        model = folder / f'k{width}'
        if (model / 'completed.json').exists():
            done = common.read(model / 'completed.json')
            assert done['complete'] and done['plan_id'] == plan['id']
            assert done['actual_iterations'] == plan['iterations']
            assert common.digest(model / 'centroids.npy') == done['centroids_sha256']
        else:
            # The KMeans trainer skips existing metadata; only accept a completed model from this run.
            assert not (model / 'metadata.json').exists(), f'Incomplete model requires inspection: {model}'
            pending.append(width)
    if not pending:
        return
    x = load_corpus(plan, progress=output / family / 'loading.json')
    for width in pending:
        cap = math.ceil(plan['rows'] / width)
        assert cap * width >= plan['rows']
        previous.train_kmeans(
            sample_path=None, sample_array=x, output_dir=folder, k_values=[width],
            training_dim=plan['dimension'], input_dim=plan['dimension'], input_space='raw',
            sample_rows=plan['rows'], seed=plan['seed'], niter=plan['iterations'], nredo=1,
            max_points_per_centroid=cap, use_gpu=require_gpu, ensure_niter=True)
        model = folder / f'k{width}'
        meta = common.read(model / 'metadata.json')
        assert meta['actual_iterations'] == len(meta['objective']) == plan['iterations']
        centroid_hash = common.digest(model / 'centroids.npy')
        meta.update(from_scratch=True, initial_centroids=None, all_corpus_rows=True,
                    faiss_subsampling=False, training_split='main', plan_id=plan['id'],
                    data_loading='Original embedding shards directly into RAM; no disk corpus copy',
                    source_code_sha256=common.digest(previous.__file__),
                    runner_sha256=common.digest(__file__), centroids_sha256=centroid_hash)
        common.write(model / 'metadata.json', meta)
        common.write(model / 'completed.json', dict(
            complete=True, model_dir=str(model.resolve()), plan_id=plan['id'],
            actual_iterations=meta['actual_iterations'], centroids_sha256=centroid_hash,
            metadata_sha256=common.digest(model / 'metadata.json')))
        print(f'COMPLETE {family} k={width}: all {plan["rows"]:,} rows, '
              f'{plan["iterations"]} iterations from scratch', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('train', 'finish'))
    parser.add_argument('--family', choices=common.FAMILIES)
    parser.add_argument('--output', type=Path, default=common.OUT)
    args = parser.parse_args()
    if args.command == 'finish':
        common.finish(args)
        jobs_path = args.output / 'jobs.json'
        jobs = common.read(jobs_path)
        jobs.update(complete=True, completed_utc=datetime.now(timezone.utc).isoformat())
        common.write(jobs_path, jobs)
    else:
        assert args.family is not None
        plan = common.load_plan(args.output, args.family)
        assert plan['runner_sha256'] == common.digest(__file__)
        assert plan['trainer_sha256'] == common.digest(previous.__file__)
        train(plan, args.output)


if __name__ == '__main__':
    main()
