"""Scientific helper and isolated workflow regressions."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from tools import runner
from scripts.hierarchy.aggregate_probes import aggregate
from scripts.hierarchy.circle_packing import pack_circles, validate_packing
from atomic_features import matching
from scripts.stability import persistent_matching as compatibility
ROOT = Path(__file__).resolve().parents[1]


def test_one_matching_implementation():
    assert compatibility.persistent_matching is matching.persistent_matching
    assert compatibility.independent_assignments is matching.independent_assignments
    assert not hasattr(compatibility, 'joint_persistent_matching')


def test_circle_packing_order_and_geometry():
    entries = [('a', .81), ('ab', .16), ('ac', .09), ('ad', .04)]
    packed = pack_circles(entries)
    validate_packing(packed)
    for a, b in zip(packed, pack_circles(entries[::-1])):
        assert a[:3] == b[:3]
        np.testing.assert_array_equal(a[3], b[3])
        assert a[2] == pytest.approx(a[1] ** .5)


def write_probe(root, dataset, model, text):
    folder = root / 'probe_results' / dataset / model
    folder.mkdir(parents=True)
    (folder / 'summary.tsv').write_text(text)


def test_aggregate_retains_rows_and_schema_union(tmp_path):
    write_probe(tmp_path, 'wordfreq', 'gemini', 'category\tF1\nword\t0.75\n')
    write_probe(tmp_path, 'gbif', 'nemotron', 'category\tF1\tlevel\nbird\t0.9\tclass\n')
    root = tmp_path / 'probe_results'
    aggregate(SimpleNamespace(out_base=root))
    with (root / 'all_summaries.tsv').open() as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    assert rows == [dict(category='bird', F1='0.9', level='class'), dict(category='word', F1='0.75', level='')]
    assert not (root / 'plots').exists()


def test_registry_program_and_asset_coverage():
    registry = runner.read(ROOT / 'registry.json')
    ids = [w['id'] for w in registry['workflows']]
    assert len(ids) == len(set(ids))
    assert registry['artifacts']
    assert 'table_overrides' not in registry
    assert not {'paper', 'transform-layout'} & set(ids)
    for workflow in registry['workflows']:
        assert workflow['family']
        for command in runner.commands_for(workflow):
            assert command[0] == '${python}'
            path = 'src/' + command[2].replace('.', '/') + '.py' if command[1] == '-m' else command[1]
            assert (ROOT / path).is_file(), path
    for asset in registry['artifacts']:
        if asset.get('producer'): assert (ROOT / asset['producer']).is_file()
    for analysis in registry['analyses']:
        for source in analysis['code']: assert (ROOT / source).is_file(), source


def test_synthetic_sweep_complete_and_ordered():
    registry = runner.read(ROOT / 'registry.json')
    workflow = next(w for w in registry['workflows'] if w['id'] == 'synthetic-train')
    commands = list(runner.commands_for(workflow))
    assert len(commands) == 468
    assert [int(c[c.index('--index') + 1]) for c in commands] == list(range(468))
    assert all(c[-2:] == ['--device', 'cuda'] for c in commands)


def test_prepare_and_run_with_spaces_and_frozen_sources(tmp_path):
    inputs = tmp_path / 'original inputs'
    write_probe(inputs, 'wordfreq', 'gemini', 'category\tF1\na\t0.8\n')
    (inputs / 'untrusted.py').write_text('raise RuntimeError("must not execute")')
    work = tmp_path / 'work with spaces'
    config_file = tmp_path / 'resources.json'
    config_file.write_text(json.dumps(dict(work=str(work), paths={'hierarchy_data_dir': str(work / 'data/hierarchy')}, inputs={'data/hierarchy': str(inputs)})))
    config = runner.configuration(config_file)
    registry = runner.read(ROOT / 'registry.json')
    runner.prepare(config, registry)
    assert (work / 'scripts/hierarchy/aggregate_probes.py').read_bytes() == (ROOT / 'scripts/hierarchy/aggregate_probes.py').read_bytes()
    assert not (work / 'data/hierarchy/untrusted.py').exists()
    workflow = next(w for w in registry['workflows'] if w['id'] == 'aggregate-probes')
    assert runner.run(workflow, config)['success']
    assert (work / 'data/hierarchy/probe_results/all_summaries.tsv').read_text() == 'category\tF1\na\t0.8\n'
    assert not (inputs / 'probe_results/all_summaries.tsv').exists()
    (work / 'scripts/hierarchy/aggregate_probes.py').write_text('raise RuntimeError("changed")')
    with pytest.raises(ValueError, match='Prepared source changed'): runner.run(workflow, config)


def test_paths_and_config_boundaries(tmp_path):
    with pytest.raises(ValueError): runner.inside(tmp_path, '../escape')
    with pytest.raises(ValueError): runner.inside(tmp_path, '/absolute')
    config = dict(work=str(tmp_path / 'run'), paths={}, inputs={}, path_aliases={'/old': 'missing'})
    with pytest.raises(ValueError, match='Configure alias paths'): runner.mapping(config, {'resource_roots': {}})


def test_paper_persistence_computes_membership_from_graphs(tmp_path, monkeypatch):
    from scipy.sparse import csr_matrix, save_npz
    from scripts.stability import paper_pairwise_persistence as paper
    monkeypatch.setattr(paper, 'WIDTHS', (2, 3))
    monkeypatch.setattr(paper, 'MODELS', ('gemini',))
    monkeypatch.setattr(paper, 'METRICS', ('cosine',))
    directory = paper.source_directory(tmp_path, 'gemini', 'cosine')
    directory.mkdir(parents=True)
    (directory / 'summary.json').write_text(json.dumps([
        dict(width=2, target_widths=[3], threshold=.7, persistent_count=0)
    ]))
    (directory / 'manifest.json').write_text('{}')
    save_npz(directory / 'graph_2_to_3.npz', csr_matrix([[True, False, False], [True, False, False]]))
    rows = paper.compute(tmp_path, tmp_path / 'output')
    assert rows == [dict(model='gemini', metric='cosine', width=2, persistent_count=1, proportion=.5)]
    with np.load(tmp_path / 'output/cosine/gemini/assignments_2.npz') as assignments:
        np.testing.assert_array_equal(assignments['persistent_source'], [0])
    assert not (tmp_path / 'output/comparison_to_joint.png').exists()


def test_generated_asset_outputs_are_registered():
    registry = runner.read(ROOT / 'registry.json')
    outputs = {p for w in registry['workflows'] for p in w.get('outputs', [])}
    for asset in registry['artifacts']:
        if asset.get('producer'):
            assert any(p in outputs for p in asset['sources']), asset['macro']


def test_source_distribution_excludes_manuscript():
    assert not (ROOT / 'latex').exists()
    assert not (ROOT / 'provenance').exists()
    assert not (ROOT / 'tools/build_release.py').exists()
    assert not (ROOT / 'tools/inventory.py').exists()
    assert not any(p.suffix in {'.tex', '.bib', '.sty', '.cls'} for p in runner.source_files(ROOT))
    for name in ('build_paper.py', 'paper_assets.py', 'pdf_table_assets.py', 'manuscript.py'):
        assert not (ROOT / 'tools' / name).exists()


def test_kmeans_finish_without_manuscript(tmp_path, monkeypatch):
    from scripts import kmeans_512_iter100 as kmeans
    monkeypatch.setattr(kmeans.common, 'ROOT', tmp_path)
    monkeypatch.setattr(kmeans, 'validate', lambda output, family: {'family': family, 'complete': True})
    commands = []
    monkeypatch.setattr(kmeans.subprocess, 'run', lambda argv, **kwargs: commands.append(argv))
    output = tmp_path / 'results'
    output.mkdir()
    (output / 'jobs.json').write_text('{}')
    kmeans.finish(output)
    assert json.loads((output / 'COMPLETE.json').read_text())['complete']
    assert json.loads((output / 'jobs.json').read_text())['complete']
    assert len(json.loads((output / 'summary.json').read_text())['models']) == 2
    assert [command[2] for command in commands] == ['scripts.plot_sae_kmeans_prevalence']
    assert not (tmp_path / 'latex').exists()


def test_prepare_discovers_edited_sources_and_script_uses_prepared_copy(tmp_path, monkeypatch):
    import sys
    source = tmp_path / 'source'
    (source / 'scripts').mkdir(parents=True)
    script = source / 'scripts/task.py'
    script.write_text('raise RuntimeError("old code")\n')
    registry = {'resource_roots': {}, 'workflows': []}
    (source / 'registry.json').write_text(json.dumps(registry))
    # Ordinary edits and new files require no hash-list maintenance.
    script.write_text('from pathlib import Path\nPath("result.txt").write_text("edited")\n')
    for name in ('experiments/results/stale.json', 'scripts/__pycache__/junk.py',
                 '.venv/lib/site.py', 'build/lib/package.py', 'reproduction_state.json'):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
    (source / 'scripts/local-link.py').symlink_to(script)
    monkeypatch.setattr(runner, 'ROOT', source)
    monkeypatch.setattr(runner, 'REGISTRY', source / 'registry.json')
    config_file = tmp_path / 'resources.json'
    work = tmp_path / 'work'
    config_file.write_text(json.dumps(dict(work=str(work), paths={}, inputs={})))
    config = runner.configuration(config_file)
    runner.prepare(config, registry)
    state = runner.read(work / 'reproduction_state.json')
    assert set(state['source_manifest']) == {'registry.json', 'scripts/task.py'}
    assert state['source_manifest']['scripts/task.py'] == runner.sha(work / 'scripts/task.py')
    assert not (work / 'provenance').exists()
    # The prepared copy is self-contained even if the original script is removed.
    script.unlink()
    monkeypatch.setattr(sys, 'argv', ['paper.py', 'script', '--config', str(config_file), '--', 'scripts/task.py'])
    runner.main()
    assert (work / 'result.txt').read_text() == 'edited'
    # A later-added script was not part of preparation and cannot be run via script.
    (work / 'scripts/extra.py').write_text('raise RuntimeError("must not run")')
    monkeypatch.setattr(sys, 'argv', ['paper.py', 'script', '--config', str(config_file), '--', 'scripts/extra.py'])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2


def test_control_source_checks_use_actual_bytes_or_supplied_snapshot(tmp_path):
    from scripts.review_controls.report_fixed_source_swap import verify_source_snapshots
    script = tmp_path / 'scripts/control.py'
    script.parent.mkdir()
    script.write_text('original source')
    expected = {'scripts/control.py': runner.sha(script)}
    verify_source_snapshots(expected, tmp_path)
    script.write_text('edited source')
    with pytest.raises(ValueError, match='Missing matching provenance source'):
        verify_source_snapshots(expected, tmp_path)
    snapshot = tmp_path / 'data/source_snapshots/scripts/control.py.source'
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text('original source')
    verify_source_snapshots(expected, tmp_path)
    script.unlink()  # A supplied snapshot suffices when the source path is absent.
    verify_source_snapshots(expected, tmp_path)
    snapshot.write_text('tampered source')
    with pytest.raises(ValueError, match='Missing matching provenance source'):
        verify_source_snapshots(expected, tmp_path)


@pytest.mark.parametrize('equals_form', [False, True])
@pytest.mark.parametrize('fail', [False, True])
def test_runner_keeps_credentials_out_of_plans_and_reports(tmp_path, monkeypatch, capsys, equals_form, fail):
    import hashlib
    import sys

    source = tmp_path / 'source'
    (source / 'scripts').mkdir(parents=True)
    (source / 'scripts/provider.py').write_text(
        'import argparse, hashlib\n'
        'from pathlib import Path\n'
        'p = argparse.ArgumentParser(allow_abbrev=False)\n'
        'p.add_argument("--api-key", required=True)\n'
        'p.add_argument("--fail", action="store_true")\n'
        'args = p.parse_args()\n'
        'Path("received.sha256").write_text(hashlib.sha256(args.api_key.encode()).hexdigest())\n'
        'print("Provider invoked")\n'
        'raise SystemExit(7 if args.fail else 0)\n'
    )
    registry = {'resource_roots': {}, 'workflows': []}
    (source / 'registry.json').write_text(json.dumps(registry))
    monkeypatch.setattr(runner, 'ROOT', source)
    monkeypatch.setattr(runner, 'REGISTRY', source / 'registry.json')
    work = tmp_path / 'work'
    config_file = tmp_path / 'resources.json'
    config_file.write_text(json.dumps(dict(work=str(work), paths={}, inputs={})))
    config = runner.configuration(config_file)
    runner.prepare(config, registry)
    # A synthetic credential with punctuation also checks that argv is forwarded intact.
    secret = 'test-provider-credential=with spaces&punctuation'
    args = [f'--api-key={secret}'] if equals_form else ['--api-key', secret]
    if fail:
        args.append('--fail')
    command = ['${python}', 'scripts/provider.py', *args]
    workflow = dict(id='provider', stage='custom', commands=[command])
    planned = runner.plan(workflow, config)
    assert secret not in json.dumps(planned)
    assert '[REDACTED]' in json.dumps(planned)
    monkeypatch.setattr(sys, 'argv', [
        'paper.py', 'script', '--config', str(config_file), '--', 'scripts/provider.py', *args,
    ])
    if fail:
        with pytest.raises(SystemExit) as error:
            runner.main()
        assert error.value.code == 1
    else:
        runner.main()
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err
    assert (work / 'received.sha256').read_text() == hashlib.sha256(secret.encode()).hexdigest()
    report_path, = (work / 'run_logs').glob('*.json')
    report = runner.read(report_path)
    assert report['success'] is not fail
    assert report['commands'] == planned['commands']
    assert report['steps'][0]['command'] == planned['commands'][0]
    assert report['steps'][0]['returncode'] == (7 if fail else 0)
    for path in (work / 'run_logs').iterdir():
        assert secret not in path.read_text()


def test_control_checkpoint_names_and_resource_remapping(tmp_path, monkeypatch):
    from scripts.control_models import CHECKPOINTS, FIXED, ALTERNATE, checkpoint_path
    from scripts.review_controls.directions import trainpath
    from project_paths import resource_path

    model_root = tmp_path / 'models'
    remap = tmp_path / 'remap.json'
    remap.write_text(json.dumps({'/resources/control_models_dir': str(model_root)}))
    monkeypatch.setenv('ATOMIC_PATH_REMAP', str(remap))
    paths = set()
    for record in CHECKPOINTS:
        family, width, top_k, seed = (record[k] for k in ('family', 'width', 'k', 'seed'))
        if record['control'] == 'fixed-k':
            assert FIXED[family][width] == seed
        else:
            assert ALTERNATE[width] == seed
        path = checkpoint_path(family, width, top_k, seed)
        assert path.parent == model_root
        assert path.name == f'{family}_m{width}_k{top_k}_seed{seed}.pt'
        assert trainpath(family, width, top_k, seed) == path
        assert resource_path(path) == path
        paths.add(path)
    assert len(paths) == 9
