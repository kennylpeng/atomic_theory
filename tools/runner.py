"""Run registered paper stages using immutable source and explicit resources."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / 'registry.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


# Stage source/configuration files, excluding generated data and local environments.
SOURCE_DIRECTORIES = {
    'src', 'tools', 'scripts', 'experiments', 'hierarchy_data_prep',
    'data_prep', 'trainer', 'setup_configs', 'release', 'tests', 'assets',
}
SOURCE_SUFFIXES = {'.py', '.json', '.yaml', '.yml', '.txt', '.md', '.png'}
ROOT_FILES = {
    'LICENSE', 'README.md', 'REPRODUCTION.md', 'DATA_TERMS.md',
    'paper.py', 'project_paths.py', 'registry.json', 'resources.example.json',
    'constraints.txt', 'pyproject.toml', 'MANIFEST.in',
}


def source_files(root):
    """Discover source files at preparation time; no maintained hash list is needed."""
    excluded = {'build', 'dist', 'results', 'plots', 'runs', 'data', 'generated', '__pycache__'}
    for folder, directories, files in os.walk(root, followlinks=False):
        folder = Path(folder)
        directories[:] = sorted(d for d in directories
            if not d.startswith('.') and d not in excluded and not d.endswith('.egg-info')
            and not (folder / d).is_symlink()
            and (folder != root or d in SOURCE_DIRECTORIES))
        for name in sorted(files):
            path = folder / name
            if path.is_symlink() or name.startswith('.'):
                continue
            if (folder == root and name in ROOT_FILES) or (folder != root and path.suffix in SOURCE_SUFFIXES):
                yield path


def inside(root, name):
    if Path(name).is_absolute():
        raise ValueError(f'Destination must be relative: {name}')
    target = root / name
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Destination escapes work directory: {name}')
    return target


def configuration(path):
    path = Path(path).resolve()
    config = read(path)
    def absolute(value):
        p = Path(os.path.expandvars(value)).expanduser()
        return str((path.parent / p).resolve()) if not p.is_absolute() else str(p.resolve())
    config['work'] = absolute(config['work'])
    config['paths'] = {k: absolute(v) for k, v in config.get('paths', {}).items() if v}
    config['inputs'] = {k: absolute(v) for k, v in config.get('inputs', {}).items() if v}
    aliases = config.get('path_aliases', {})
    if not isinstance(aliases, dict) or any(not Path(k).is_absolute() or not isinstance(v, str) for k, v in aliases.items()):
        raise ValueError('path_aliases must map recorded absolute roots to configured path keys')
    return config


def mapping(config, registry):
    paths = dict(config['paths'], workspace=config['work'])
    paths.setdefault('scratch_dir', str(Path(config['work']) / 'scratch'))
    paths.setdefault('local_cache_dir', str(Path(config['work']) / 'scratch/cache'))
    aliases = config.get('path_aliases', {})
    missing = set(aliases.values()) - set(paths)
    if missing:
        raise ValueError('Configure alias paths: ' + ', '.join(sorted(missing)))
    roots = dict(registry['resource_roots'], **aliases)
    remap = {old: paths[key] for old, key in roots.items() if key in paths}
    # These aliases apply to filenames read from saved metadata.
    return remap, paths


def prepare(config, registry):
    work = Path(config['work'])
    if work.exists():
        raise FileExistsError(f'Use a new work directory: {work}')
    if work.is_relative_to(ROOT):
        raise ValueError('Choose a work directory outside the source tree')
    for name, source in config['inputs'].items():
        dest = inside(work, name)
        if dest.relative_to(work).parts[0] not in {'data', 'full_experiments', 'experiments', 'exports', 'releases'}:
            raise ValueError(f'Not an asset namespace: {name}')
        if not Path(source).exists():
            raise FileNotFoundError(source)
    remap, paths = mapping(config, registry)
    work.mkdir(parents=True)
    try:
        inventory = {}
        for source in source_files(ROOT):
            name = source.relative_to(ROOT).as_posix()
            target = inside(work, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            inventory[name] = sha(target)
        def copy_asset(source, dest):
            dest = Path(dest)
            if dest.suffix in {'.py', '.pyc', '.sh', '.so', '.pth'}:
                return str(dest)
            if dest.exists():
                raise FileExistsError(f'Input collision: {dest}')
            return shutil.copy2(source, dest)
        for name, source in config['inputs'].items():
            target = inside(work, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            if Path(source).is_dir():
                shutil.copytree(source, target, dirs_exist_ok=True, copy_function=copy_asset)
            elif target.suffix == '.py':
                snapshot = target.with_name(target.name + '.source')
                if snapshot.exists():
                    raise FileExistsError(snapshot)
                shutil.copy2(source, snapshot)
            else:
                copy_asset(source, target)
        import yaml
        (work / 'config').mkdir(exist_ok=True)
        (work / 'config/paths.yaml').write_text(yaml.safe_dump(paths))
        (work / 'config/remap.json').write_text(json.dumps(remap, indent=2) + '\n')
        (work / 'reproduction_state.json').write_text(json.dumps({
            'configuration': config, 'source_manifest': inventory,
            'registry_sha256': inventory['registry.json'],
            'created_utc': datetime.now(timezone.utc).isoformat(),
        }, indent=2) + '\n')
    except Exception:
        (work / 'PREPARE_FAILED').touch()
        raise
    return str(work)


def expand(value, config):
    values = dict(config['paths'], work=config['work'], python=sys.executable)
    def replacement(match):
        key = match.group(1)
        if key not in values:
            raise ValueError(f'Set paths.{key} in resources configuration')
        return values[key]
    return re.sub(r'\$\{([a-zA-Z0-9_]+)\}', replacement, value)


def commands_for(workflow):
    """Expand an ordered parameter sweep without duplicating command definitions."""
    axes = workflow.get('matrix', {})
    names = list(axes)
    values = [range(*v['range']) if isinstance(v, dict) else v for v in axes.values()]
    for combination in itertools.product(*values):
        substitutions = dict(zip(names, combination))
        for command in workflow['commands']:
            expanded = []
            for arg in command:
                for key, value in substitutions.items():
                    arg = arg.replace('${' + key + '}', str(value))
                expanded.append(arg)
            yield expanded


CREDENTIAL_OPTIONS = {
    '--api-key', '--api_key', '--access-token', '--access_token',
    '--auth-token', '--auth_token', '--token', '--password',
    '--client-secret', '--client_secret', '--secret', '--authorization',
}


def redact_command(command):
    """Return a printable argv without credential values; never alter execution argv."""
    redacted = []
    hide_next = False
    for arg in command:
        if hide_next:
            redacted.append('[REDACTED]')
            hide_next = False
            continue
        option, separator, _ = arg.partition('=')
        if option.lower() in CREDENTIAL_OPTIONS:
            redacted.append(option + '=[REDACTED]' if separator else option)
            hide_next = not separator
        else:
            redacted.append(arg)
    return redacted


def execution_commands(workflow, config):
    return [[expand(arg, config) for arg in command] for command in commands_for(workflow)]


def plan(workflow, config):
    commands = [redact_command(command) for command in execution_commands(workflow, config)]
    required = [expand(p, config) for p in workflow.get('requires', [])]
    return dict(id=workflow['id'], family=workflow.get('family'), stage=workflow['stage'],
                commands=commands, missing=[p for p in required if not Path(p).exists()],
                outputs=workflow.get('outputs', []), notes=workflow.get('notes', ''))


def run(workflow, config):
    work = Path(config['work'])
    state_path = work / 'reproduction_state.json'
    if not state_path.is_file() or (work / 'PREPARE_FAILED').exists():
        raise ValueError('Run prepare successfully before run')
    state = read(state_path)
    if state['configuration'] != config:
        raise ValueError('Configuration changed; prepare a new work directory')
    if sha(REGISTRY) != state['registry_sha256']:
        raise ValueError('Registry changed; prepare a new work directory')
    # Verify staged code to keep the provenance attached to the code actually executed.
    for name, expected in state['source_manifest'].items():
        path = work / name
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f'Prepared source changed: {name}')
    execution = plan(workflow, config)
    if execution['missing']:
        raise FileNotFoundError('Missing inputs:\n' + '\n'.join(execution['missing']))
    logdir = work / 'run_logs'
    logdir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(work / 'src'), str(work), str(work / 'tools')]),
               MPLBACKEND='Agg', ATOMIC_THEORY_PATHS=str(work / 'config/paths.yaml'),
               ATOMIC_PATH_REMAP=str(work / 'config/remap.json'))
    report = dict(execution, started_utc=stamp, steps=[], success=False)
    try:
        for i, command in enumerate(execution_commands(workflow, config)):
            logfile = logdir / f'{workflow["id"]}-{stamp}-{i}.log'
            with logfile.open('w') as stream:
                result = subprocess.run(command, cwd=work, env=env, stdout=stream, stderr=subprocess.STDOUT)
            report['steps'].append(dict(command=redact_command(command), returncode=result.returncode, log=str(logfile)))
            if result.returncode:
                raise RuntimeError(f'Workflow failed ({result.returncode}); see {logfile}')
        outputs = [inside(work, p) for p in workflow.get('outputs', [])]
        for path in outputs:
            if not path.is_file():
                raise FileNotFoundError(f'Expected output not generated: {path}')
        report['output_sha256'] = {str(p.relative_to(work)): sha(p) for p in outputs}
        report['success'] = True
    finally:
        (logdir / f'{workflow["id"]}-{stamp}.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['list', 'prepare', 'plan', 'run', 'script', 'guide'])
    parser.add_argument('workflow', nargs='?')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--family', help='Filter list by workflow family')
    argv = sys.argv[1:]
    sep = argv.index('--') if '--' in argv else len(argv)
    args = parser.parse_args(argv[:sep])
    extra = argv[sep + 1:]
    registry = read(REGISTRY)
    try:
        if args.action == 'list':
            for w in sorted(registry['workflows'], key=lambda w: (w['family'], w['id'])):
                if not args.family or args.family == w['family']:
                    print(f'{w["family"]:15s} {w["id"]:28s} {w["stage"]:12s} {w["description"]}')
            return
        if args.action == 'guide':
            for a in registry['analyses']:
                print(f'{", ".join(a["paper"])}: {a["title"]}')
                print('  Stage:', a.get('workflow', 'authored/upstream; see REPRODUCTION.md'))
                for asset in registry['artifacts']:
                    if asset['macro'] in a['assets']:
                        print(' ', asset['macro'], '->', asset.get('producer', 'authored/support input'))
            return
        if not args.config:
            parser.error('--config is required')
        config = configuration(args.config)
        if args.action == 'prepare':
            if extra:parser.error('Extra arguments require script')
            result = prepare(config, registry)
        else:
            if args.action == 'script':
                if not extra:parser.error('Use script --config CONFIG -- relative/program.py [arguments]')
                program = inside(Path(config['work']), extra[0])
                inventory = read(Path(config['work']) / 'reproduction_state.json')['source_manifest']
                if extra[0] not in inventory or program.suffix != '.py':
                    parser.error('Choose a Python program copied during prepare')
                workflow = dict(id='script-' + program.stem, stage='custom', commands=[['${python}', *extra]])
            else:
                if extra:parser.error('Extra arguments require script')
                workflow = next((w for w in registry['workflows'] if w['id'] == args.workflow), None)
                if workflow is None:parser.error('Choose a stage from list')
            result = (plan if args.action == 'plan' else run)(workflow, config)
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, str(exc) + '\n')
