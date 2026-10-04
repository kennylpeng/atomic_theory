"""Fresh, full-dimensional FAISS KMeans on every main-corpus row, 20 iterations."""
from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time

import numpy as np

ROOT = _paper_path(__file__).resolve().parents[1]
OUT = ROOT / 'full_experiments/results/kmeans_raw_full_corpus_iter20'
SOURCE = ROOT / 'full_experiments/results/kmeans_raw_cross_width_activation_matches'
SCRATCH = _paper_path('/resources/scratch_dir/kmeans_raw_full_corpus_iter20')
FAMILIES = ('gemini', 'nemotron')
WIDTHS = (4096, 16384, 131072)
SEED = 20260624


def read(path):
    return json.loads(_paper_path(path).read_text())


def write(path, value):
    path = _paper_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with _paper_path(path).open('rb') as f:
        while block := f.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def stat_matches(path, size, mtime_ns):
    st = _paper_path(path).stat()
    assert st.st_size == size and st.st_mtime_ns == mtime_ns, str(path)


def prepare(args):
    for family in FAMILIES:
        path = args.output / family / 'plan.json'
        if path.exists():
            assert read(path)['source_code_sha256'] == digest(__file__)
            print('Existing plan:', path, flush=True)
            continue
        previous_path = SOURCE / family / 'manifest.json'
        previous = read(previous_path)
        shards, offset = [], 0
        for shard in previous['shards']:
            old = read(SOURCE / family / 'assignments' / shard['relative_shard'] / 'complete.json')
            assert old['manifest_id'] == previous['id'] and old['shard'] == shard
            stat_matches(old['embedding_path'], old['embedding_size'], old['embedding_mtime_ns'])
            shards.append(dict(relative_shard=shard['relative_shard'], rows=shard['rows'], offset=offset,
                               path=old['embedding_path'], size=old['embedding_size'],
                               mtime_ns=old['embedding_mtime_ns']))
            offset += shard['rows']
        assert offset == previous['rows'] and len({s['path'] for s in shards}) == len(shards)
        plan = dict(family=family, dimension=previous['dimension'], rows=offset, shards=shards,
                    widths=list(WIDTHS), iterations=20, checkpoint_iterations=5, seed=SEED,
                    sample_path=str(args.scratch / family / 'full_corpus.f32'),
                    input_space='raw', preprocessing='float32 conversion only',
                    initialization='FAISS RANDOM from full corpus, one fresh initialization per width',
                    training='Euclidean KMeans; every row each iteration; no subsampling',
                    source_manifest_path=str(previous_path), source_manifest_sha256=digest(previous_path),
                    source_code_sha256=digest(__file__), created_utc=datetime.now(timezone.utc).isoformat())
        plan['id'] = object_digest(plan)
        write(path, plan)
        print(f'{family}: {offset:,} rows, {plan["dimension"]} dimensions, '
              f'{offset*plan["dimension"]*4/2**30:.1f} GiB staged float32', flush=True)


def load_plan(output, family):
    plan = read(output / family / 'plan.json')
    assert plan['source_code_sha256'] == digest(__file__)
    return plan


def stage(plan, output, batch_rows=8192):
    """Resumable, byte-verified conversion of every original shard to float32."""
    family = plan['family']
    folder = output / family
    path = _paper_path(plan['sample_path'])
    path.parent.mkdir(parents=True, exist_ok=True)
    size = plan['rows'] * plan['dimension'] * 4
    identity = path.parent / 'identity.json'
    if identity.exists():
        assert read(identity)['plan_id'] == plan['id']
    else:
        assert not path.exists(), 'Refusing to reuse untracked staged data'
        write(identity, dict(plan_id=plan['id'], bytes=size))
    if not path.exists():
        with path.open('xb') as f:
            f.truncate(size)
    assert path.stat().st_size == size
    destination = np.memmap(_paper_location(path), dtype=np.float32, mode='r+', shape=(plan['rows'], plan['dimension']))
    completed, started = [], time.monotonic()
    for index, shard in enumerate(plan['shards']):
        stat_matches(shard['path'], shard['size'], shard['mtime_ns'])
        checkpoint = folder / 'staged_shards' / f'{index:04d}.json'
        if checkpoint.exists():
            record = read(checkpoint)
            assert record['plan_id'] == plan['id'] and record['shard'] == shard
            completed.append(record)
            continue
        x = np.load(_paper_location(shard['path']), mmap_mode='r', allow_pickle=False)
        assert x.shape == (shard['rows'], plan['dimension'])
        h = hashlib.sha256()
        offset = shard['offset']
        for start in range(0, len(x), batch_rows):
            stop = min(start + batch_rows, len(x))
            block = np.ascontiguousarray(x[start:stop], dtype=np.float32)
            assert np.isfinite(block).all()
            h.update(memoryview(block).cast('B'))
            destination[offset+start:offset+stop] = block
        destination.flush()
        observed = hashlib.sha256(memoryview(destination[offset:offset+len(x)]).cast('B')).hexdigest()
        assert observed == h.hexdigest(), 'Staged bytes disagree with converted input'
        stat_matches(shard['path'], shard['size'], shard['mtime_ns'])
        record = dict(plan_id=plan['id'], shard=shard, float32_sha256=observed,
                      all_rows_finite=True, staged_bytes_verified=True)
        write(checkpoint, record)
        completed.append(record)
        print(f'{family} staged {index+1}/{len(plan["shards"])} shards; '
              f'{offset+len(x):,}/{plan["rows"]:,} rows; {time.monotonic()-started:.1f}s', flush=True)
    assert len(completed) == len(plan['shards'])
    assert sum(r['shard']['rows'] for r in completed) == plan['rows']
    destination.flush()
    del destination
    st = path.stat()
    write(folder / 'staged.json', dict(complete=True, plan_id=plan['id'], rows=plan['rows'],
          path=str(path), size=st.st_size, mtime_ns=st.st_mtime_ns,
          shard_records_sha256=object_digest(completed), source_code_sha256=digest(__file__),
          all_rows_finite=True, all_staged_bytes_verified=True))
    print(f'{family}: full corpus staged and verified', flush=True)


def train_width(plan, output, width, require_gpu=True, stop_after=None):
    import faiss
    family, n, dim = plan['family'], plan['rows'], plan['dimension']
    assert width in plan['widths'] and width < n
    staged_path = output / family / 'staged.json'
    staged = read(staged_path)
    assert staged['complete'] and staged['plan_id'] == plan['id'] and staged['rows'] == n
    assert staged['all_rows_finite'] and staged['all_staged_bytes_verified']
    stat_matches(staged['path'], staged['size'], staged['mtime_ns'])
    model = output / 'models' / family / f'k{width}'
    model.mkdir(parents=True, exist_ok=True)
    if (model / 'completed.json').exists():
        done = read(model / 'completed.json')
        assert done['plan_id'] == plan['id'] and done['actual_iterations'] == plan['iterations']
        assert digest(model / 'centroids.npy') == done['centroids_sha256']
        print('Verified completed model:', model, flush=True)
        return
    gpu_count = faiss.get_num_gpus() if require_gpu else 0
    assert not require_gpu or gpu_count > 0
    faiss.omp_set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '2')))
    cap = math.ceil(n / width)
    assert cap * width >= n
    sample = np.memmap(_paper_location(staged['path']), dtype=np.float32, mode='r', shape=(n, dim))
    fit = faiss.Kmeans(d=dim, k=width, niter=plan['checkpoint_iterations'], nredo=1,
                      seed=plan['seed'], spherical=False, verbose=True, gpu=gpu_count,
                      max_points_per_centroid=cap,
                      init_method=faiss.ClusteringInitMethod_RANDOM,
                      check_input_data_for_NaNs=False)
    assert fit.cp.nredo == 1 and fit.cp.max_points_per_centroid * width >= n
    state_path = model / 'last_checkpoint.json'
    state = dict(plan_id=plan['id'], width=width, iterations=0, objective=[], iteration_stats=[],
                 training_seconds=0., initialization='fresh_faiss_random', seed=plan['seed'], blocks=[])
    centers = None
    if state_path.exists():
        state = read(state_path)
        assert state['plan_id'] == plan['id'] and state['width'] == width
        assert state['initialization'] == 'fresh_faiss_random'
        assert digest(state['centroids_path']) == state['centroids_sha256']
        centers = np.load(_paper_location(state['centroids_path']), allow_pickle=False)
        assert centers.shape == (width, dim) and np.isfinite(centers).all()
    target = plan['iterations'] if stop_after is None else min(stop_after, plan['iterations'])
    while state['iterations'] < target:
        count = min(plan['checkpoint_iterations'], target-state['iterations'])
        fit.cp.niter = count
        print(f'{family} k={width}: iterations {state["iterations"]} -> '
              f'{state["iterations"]+count}; all {n:,} rows; {gpu_count} GPUs', flush=True)
        started = time.monotonic()
        fit.train(np.asarray(sample), init_centroids=centers)
        elapsed = time.monotonic()-started
        actual = len(fit.obj)
        assert 0 < actual <= count
        centers = np.asarray(fit.centroids, dtype=np.float32)
        assert centers.shape == (width, dim) and np.isfinite(centers).all()
        assert np.isfinite(fit.obj).all()
        state['iterations'] += actual
        state['objective'].extend(float(v) for v in fit.obj)
        state['iteration_stats'].extend(fit.iteration_stats)
        state['training_seconds'] += elapsed
        state['blocks'].append(dict(iterations=actual, elapsed_seconds=elapsed, gpu_count=gpu_count,
                                    host=socket.gethostname(), slurm_job=os.environ.get('SLURM_JOB_ID')))
        path = model / 'checkpoints' / f'iteration_{state["iterations"]:02d}.npy'
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.stem+f'.{os.getpid()}.tmp.npy')
        np.save(_paper_location(temporary), centers)
        temporary.replace(path)
        state.update(centroids_path=str(path.resolve()), centroids_sha256=digest(path))
        stat_matches(staged['path'], staged['size'], staged['mtime_ns'])
        write(state_path, state)
        print(f'SAVED {family} k={width}: {state["iterations"]}/{plan["iterations"]} '
              f'iterations; objective={state["objective"][-1]:.8g}', flush=True)
    if state['iterations'] != plan['iterations']:
        return
    assert len(state['objective']) == plan['iterations']
    assert state['objective'][-1] <= state['objective'][0]
    final_centroids = model / 'centroids.npy'
    if final_centroids.exists():
        assert digest(final_centroids) == state['centroids_sha256']
    else:
        os.link(state['centroids_path'], final_centroids)
    index = faiss.IndexFlatL2(dim)
    index.add(centers)
    faiss.write_index(index, str(model / 'centroids.index'))
    metadata = dict(k=width, input_space='raw', input_dim=dim, training_dim=dim, pca_dim=None,
                    sample_rows=n, sample_path=staged['path'], seed=plan['seed'],
                    niter=plan['iterations'], actual_iterations=state['iterations'], nredo=1,
                    from_scratch=True, initialization='FAISS RANDOM over full corpus',
                    initial_centroids=None, spherical=False, max_points_per_centroid=cap,
                    faiss_subsampling=False, training_split='main', all_corpus_rows=True,
                    num_gpus=gpu_count, use_gpu=require_gpu, faiss_version=faiss.__version__,
                    objective=state['objective'], iteration_stats=state['iteration_stats'],
                    elapsed_seconds=state['training_seconds'], checkpoint_blocks=state['blocks'],
                    plan_id=plan['id'], source_code_sha256=digest(__file__),
                    staged_metadata_sha256=digest(staged_path), centroids_sha256=state['centroids_sha256'])
    write(model / 'metadata.json', metadata)
    write(model / 'completed.json', dict(complete=True, model_dir=str(model.resolve()),
          plan_id=plan['id'], actual_iterations=state['iterations'],
          centroids_sha256=state['centroids_sha256'], metadata_sha256=digest(model / 'metadata.json')))
    print(f'COMPLETE {family} k={width}: fresh fit, {n:,} rows, {state["iterations"]} iterations', flush=True)


def finish(args):
    rows = []
    for family in FAMILIES:
        plan = load_plan(args.output, family)
        for width in WIDTHS:
            model = args.output / 'models' / family / f'k{width}'
            done, meta = read(model / 'completed.json'), read(model / 'metadata.json')
            assert done['complete'] and done['plan_id'] == meta['plan_id'] == plan['id']
            assert meta['from_scratch'] and meta['initial_centroids'] is None
            assert meta['sample_rows'] == plan['rows'] and meta['actual_iterations'] == 20
            assert meta['all_corpus_rows'] and not meta['faiss_subsampling']
            assert meta['max_points_per_centroid'] * width >= plan['rows']
            assert len(meta['objective']) == 20 and np.isfinite(meta['objective']).all()
            assert digest(model / 'metadata.json') == done['metadata_sha256']
            assert digest(model / 'centroids.npy') == done['centroids_sha256']
            centers = np.load(_paper_location(model / 'centroids.npy'), mmap_mode='r', allow_pickle=False)
            assert centers.shape == (width, plan['dimension']) and centers.dtype == np.float32
            assert np.isfinite(centers).all()
            rows.append(dict(family=family, width=width, training_rows=plan['rows'],
                             dimension=plan['dimension'], iterations=20,
                             elapsed_seconds=meta['elapsed_seconds'],
                             initial_objective=meta['objective'][0], final_objective=meta['objective'][-1],
                             final_objective_per_row=meta['objective'][-1]/plan['rows'],
                             model_dir=str(model.resolve())))
    with (args.output / 'training_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    write(args.output / 'summary.json', dict(complete=True, models=rows, from_scratch=True,
          all_rows=True, iterations=20, completed_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(rows, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'stage', 'train', 'finish'))
    parser.add_argument('--family', choices=FAMILIES)
    parser.add_argument('--width', type=int, choices=WIDTHS)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--scratch', type=Path, default=SCRATCH)
    args = parser.parse_args()
    if args.stage in ('prepare', 'finish'):
        globals()[args.stage](args)
    else:
        assert args.family is not None
        plan = load_plan(args.output, args.family)
        if args.stage == 'stage':
            stage(plan, args.output)
        else:
            for width in ((args.width,) if args.width else (131072, 16384, 4096)):
                train_width(plan, args.output, width)


if __name__ == '__main__':
    main()
