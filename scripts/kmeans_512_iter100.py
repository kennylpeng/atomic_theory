"""Full-corpus 512-cluster KMeans: 100 iterations, audited prevalence, figure rendering."""
from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location
import argparse
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scripts import train_kmeans_full_corpus as common
from scripts import train_kmeans_full_corpus_ram as ram
from scripts import kmeans_prevalence_iter100 as prevalence

OUT = common.ROOT / 'full_experiments/results/kmeans_512_full_corpus_iter100'
WIDTH = 512
ITERATIONS = 100


def hashes():
    return {name: common.digest(module.__file__) for name, module in
            [('runner_sha256', sys.modules[__name__]), ('loader_sha256', ram),
             ('common_sha256', common), ('prevalence_sha256', prevalence)]}


def prepare(output):
    for family in common.FAMILIES:
        source = common.OUT / family / 'plan.json'
        old = common.read(source)
        # The shared loader checks each shard's size and timestamp before and after reading.
        plan = dict(family=family, rows=old['rows'], dimension=old['dimension'],
                    shards=old['shards'], seed=old['seed'], width=WIDTH,
                    iterations=ITERATIONS, checkpoint_iterations=10,
                    source_plan=str(source), source_plan_sha256=common.digest(source),
                    training_split='main', input_space='raw',
                    preprocessing='float32 conversion only',
                    initialization='Fresh FAISS random initialization; one run; no subsampling',
                    **hashes())
        plan['id'] = common.object_digest(plan)
        path = output / family / 'plan.json'
        if path.exists():
            assert common.read(path) == plan, path
        else:
            common.write(path, plan)
        print(f'Prepared {family}: {plan["rows"]:,} rows, {plan["dimension"]} dimensions', flush=True)


def load_plan(output, family, *, verify_training_code=True):
    plan = common.read(output / family / 'plan.json')
    assert plan['id'] == common.object_digest({k:v for k,v in plan.items() if k != 'id'})
    assert plan['width'] == WIDTH and plan['iterations'] == ITERATIONS
    assert common.digest(plan['source_plan']) == plan['source_plan_sha256']
    if verify_training_code:
        for key, value in hashes().items():
            assert plan[key] == value, key
    return plan


def fit_loaded(plan, x, model, use_gpu=True):
    """Run native FAISS in resumable ten-iteration blocks without subsampling."""
    import faiss
    model.mkdir(parents=True, exist_ok=True)
    state_path = model / 'last_checkpoint.json'
    centers = None
    state = dict(plan_id=plan['id'], iterations=0, objective=[], blocks=[], elapsed_seconds=0.)
    if state_path.exists():
        state = common.read(state_path)
        assert state['plan_id'] == plan['id']
        assert common.digest(state['centroids_path']) == state['centroids_sha256']
        centers = np.load(_paper_location(state['centroids_path']), allow_pickle=False)
        assert centers.shape == (plan['width'], plan['dimension'])
    cap = math.ceil(plan['rows']/plan['width'])
    gpu_count = faiss.get_num_gpus() if use_gpu else 0
    assert not use_gpu or gpu_count > 0
    while state['iterations'] < plan['iterations']:
        requested = min(plan['checkpoint_iterations'], plan['iterations']-state['iterations'])
        fit = faiss.Kmeans(plan['dimension'], plan['width'], niter=requested, nredo=1,
                           seed=plan['seed'], spherical=False, verbose=True,
                           gpu=gpu_count, max_points_per_centroid=cap,
                           check_input_data_for_NaNs=False)
        assert fit.cp.max_points_per_centroid*plan['width'] >= len(x)
        started = time.monotonic()
        fit.train(x, init_centroids=centers)
        elapsed = time.monotonic()-started
        actual = len(fit.obj)
        assert 0 < actual <= requested
        centers = np.asarray(fit.centroids, dtype=np.float32)
        assert np.isfinite(centers).all() and np.isfinite(fit.obj).all()
        state['iterations'] += actual
        state['objective'].extend(float(v) for v in fit.obj)
        state['elapsed_seconds'] += elapsed
        state['blocks'].append(dict(iterations=actual, seconds=elapsed, gpus=gpu_count,
                                    slurm_job=os.environ.get('SLURM_JOB_ID')))
        path = model / f'iteration_{state["iterations"]:03d}.npy'
        np.save(_paper_location(path), centers)
        state.update(centroids_path=str(path.resolve()), centroids_sha256=common.digest(path))
        common.write(state_path, state)
        print(f'SAVED {plan["family"]}: {state["iterations"]}/100 iterations', flush=True)
    assert state['iterations'] == len(state['objective']) == plan['iterations']
    np.save(_paper_location(model / 'centroids.npy'), centers)
    index = faiss.IndexFlatL2(plan['dimension'])
    index.add(centers)
    faiss.write_index(index, str(model / 'centroids.index'))
    meta = dict(plan_id=plan['id'], family=plan['family'], k=plan['width'],
                niter=plan['iterations'], actual_iterations=state['iterations'],
                objective=state['objective'], blocks=state['blocks'], elapsed_seconds=state['elapsed_seconds'],
                sample_rows=plan['rows'], sample_path=None, input_space='raw',
                training_dim=plan['dimension'], input_dim=plan['dimension'], training_split='main',
                from_scratch=True, all_corpus_rows=True, faiss_subsampling=False,
                max_points_per_centroid=cap, nredo=1, seed=plan['seed'], spherical=False,
                faiss_version=faiss.__version__, num_gpus=gpu_count,
                centroids_sha256=common.digest(model / 'centroids.npy'), **hashes())
    common.write(model / 'metadata.json', meta)
    common.write(model / 'completed.json', dict(complete=True, plan_id=plan['id'],
        actual_iterations=state['iterations'], centroids_sha256=meta['centroids_sha256'],
        metadata_sha256=common.digest(model / 'metadata.json')))
    return centers


def count_loaded(plan, x, centers, output, use_gpu=True, batch_rows=32768):
    import faiss
    from scripts.plot_sae_kmeans_prevalence import count_density, GRID
    index = faiss.IndexFlatL2(plan['dimension'])
    index.add(centers)
    if use_gpu:
        index = faiss.index_cpu_to_all_gpus(index)
    counts = np.zeros(plan['width'], dtype=np.int64)
    audit_rows = np.unique(np.linspace(0, len(x)-1, 128, dtype=np.int64))
    observed = np.empty(len(audit_rows), dtype=np.int64)
    for start in range(0, len(x), batch_rows):
        stop = min(start+batch_rows, len(x))
        _, labels = index.search(x[start:stop], 1)
        labels = labels[:, 0]
        assert np.all((labels >= 0) & (labels < plan['width']))
        counts += np.bincount(labels, minlength=plan['width'])
        keep = np.flatnonzero((audit_rows >= start) & (audit_rows < stop))
        observed[keep] = labels[audit_rows[keep]-start]
    prevalence.valid_counts(counts, plan['width'], len(x))
    audit = prevalence.audit_labels(x[audit_rows].astype(np.float64),
                                    {plan['width']: observed}, {plan['width']: centers})
    rates = counts/len(x)
    path = output / f'{plan["family"]}_k{plan["width"]}_prevalence.npz'
    prevalence.atomic_npz(path, counts=counts, prevalence=rates, evaluation_rows=len(x),
                          log10_grid=GRID, count_density=count_density(rates),
                          audit_rows=audit_rows, audit_labels=observed, plan_id=plan['id'])
    return path, audit


def train(output, family, use_gpu=True):
    import faiss
    plan = load_plan(output, family)
    done_path = output / family / 'completed.json'
    if done_path.exists():
        validate(output, family)
        return
    faiss.omp_set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '16')))
    x = ram.load_corpus(plan, progress=output / family / 'loading.json')
    model = output / 'models' / family / f'k{WIDTH}'
    centers = fit_loaded(plan, x, model, use_gpu)
    path, audit = count_loaded(plan, x, centers, output, use_gpu)
    del x
    common.write(done_path, dict(complete=True, plan_id=plan['id'], family=family,
        width=WIDTH, iterations=ITERATIONS, evaluation_rows=plan['rows'],
        prevalence_path=str(path), prevalence_sha256=common.digest(path), audit=audit,
        model_metadata_sha256=common.digest(model / 'metadata.json'),
        completed_utc=datetime.now(timezone.utc).isoformat()))
    validate(output, family)
    print(f'COMPLETE {family}: 512 clusters, 100 iterations; full-corpus prevalence audited', flush=True)


def validate(output, family):
    # Rendering verifies saved model and count hashes. Training/resume also
    # verifies the training program's source hash.
    plan = load_plan(output, family, verify_training_code=False)
    done = common.read(output / family / 'completed.json')
    model = output / 'models' / family / f'k{WIDTH}'
    meta = common.read(model / 'metadata.json')
    assert done['complete'] and done['plan_id'] == meta['plan_id'] == plan['id']
    assert meta['actual_iterations'] == len(meta['objective']) == ITERATIONS
    assert meta['k'] == WIDTH and meta['sample_rows'] == done['evaluation_rows'] == plan['rows']
    assert meta['all_corpus_rows'] and not meta['faiss_subsampling']
    assert common.digest(model / 'centroids.npy') == meta['centroids_sha256']
    assert common.digest(model / 'metadata.json') == done['model_metadata_sha256']
    assert common.digest(done['prevalence_path']) == done['prevalence_sha256']
    assert done['audit'] and all(r['checked'] > 0 and r['max_float64_score_gap'] <= r['tolerance'] for r in done['audit'])
    with np.load(_paper_location(done['prevalence_path']), allow_pickle=False) as z:
        assert str(z['plan_id']) == plan['id'] and int(z['evaluation_rows']) == plan['rows']
        prevalence.valid_counts(z['counts'], WIDTH, plan['rows'])
        np.testing.assert_array_equal(z['prevalence'], z['counts']/plan['rows'])
    return done


def finish(output):
    records = [validate(output, family) for family in common.FAMILIES]
    common.write(output / 'summary.json', dict(complete=True, width=WIDTH, iterations=ITERATIONS, models=records))
    subprocess.run([sys.executable, '-m', 'scripts.plot_sae_kmeans_prevalence'], cwd=common.ROOT, check=True)
    fixed = common.ROOT / 'full_experiments/plots/sae_kmeans_prevalence_fixed_k128_iter100/provenance.json'
    if fixed.exists():
        subprocess.run([sys.executable, '-m', 'scripts.plot_fixed_k128_sae_kmeans_prevalence'], cwd=common.ROOT, check=True)
    common.write(output / 'COMPLETE.json', dict(complete=True,
        completed_utc=datetime.now(timezone.utc).isoformat(), models=records))
    jobs_path = output / 'jobs.json'
    if jobs_path.exists():
        jobs = common.read(jobs_path)
        jobs.update(complete=True, completed_utc=datetime.now(timezone.utc).isoformat())
        common.write(jobs_path, jobs)
    print('COMPLETE: both 512-cluster curves rendered.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'train', 'finish'])
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--family', choices=common.FAMILIES)
    args = parser.parse_args()
    if args.stage == 'train':
        assert args.family
        train(args.output, args.family)
    else:
        globals()[args.stage](args.output)


if __name__ == '__main__':
    main()
