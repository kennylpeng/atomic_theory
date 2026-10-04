"""Validate and report fixed-sparsity source replacements against all original targets."""
from project_paths import resource_path as _paper_path, resource_location as _paper_location
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz

ROOT = _paper_path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.review_controls.directions import FIXED, BASELINE, OUT, SCHEDULE, mainpath, sha, trainpath
from scripts.stability.persistent_matching import persistent_matching, validate_witness
from scripts.stability.validate_split_cardinality_outputs import certify_maximum
from project_paths import resource_path

DEST = OUT / 'fixed_source_swap'


def get(rows, experiment, width):
    found = [r for r in rows if r['experiment'] == experiment and r['width'] == width]
    assert len(found) == 1
    return found[0]


def count_pct(row, latex=False):
    return f"{row['count']:,} ({100 * row['proportion']:.1f}" + (r'\%)' if latex else '%)')


def verify_source_snapshots(scripts, root):
    """Verify recorded script hashes against source files or supplied snapshots."""
    for source, digest in scripts.items():
        current = root / source
        snapshot_source = root / 'data/source_snapshots' / source
        snapshot = snapshot_source.with_name(snapshot_source.name + '.source')
        if not any(path.is_file() and sha(path) == digest for path in (current, snapshot)):
            raise ValueError(f'Missing matching provenance source: {source}; supply data/source_snapshots/{source}.source through inputs')


def main():
    summary, all_results, checks = [], {}, []
    for family in ('gemini', 'nemotron'):
        folder = DEST / family
        complete = json.loads((folder / 'COMPLETE.json').read_text())
        assert complete['complete']
        verify_source_snapshots(complete['scripts'], ROOT)
        rows = json.loads((folder / 'results.json').read_text())
        all_results[family] = rows
        checkpoints = {str(resource_path(path)): record for path, record in
                       json.loads((folder / 'checkpoints.json').read_text()).items()}
        for path, metadata in checkpoints.items():
            stat = _paper_path(path).stat()
            # Copies and archive extraction can change timestamps. Verify the
            # checkpoint contents against the recorded scientific identity.
            assert stat.st_size == metadata['size_bytes']
            assert sha(path) == metadata['sha256'], path
        for width, seed in FIXED[family].items():
            source = trainpath(family, width, 128, seed)
            source_meta = checkpoints[str(source)]
            source_config = source_meta['config']
            baseline_config = checkpoints[str(mainpath(family, width))]['config']
            assert source_config['top_k'] == 128 and baseline_config['top_k'] == SCHEDULE[width]
            assert source_config['seed'] == baseline_config['seed']
            same = get(rows, 'cross_sparsity_same_width', width)
            graph_path = folder/'graphs'/f'cross_sparsity_same_width_{width}.npz'
            graph = load_npz(_paper_location(graph_path))
            metadata = json.loads(graph_path.with_suffix('.json').read_text())
            assert metadata['source_sha256'] == checkpoints[str(mainpath(family, width))]['sha256']
            assert metadata['target_sha256'] == source_meta['sha256']
            with np.load(_paper_location(folder / f'cross_sparsity_same_width_{width}_assignment.npz')) as z:
                certify_maximum(graph, z['source'], z['destination'])
                assert len(z['source']) == same['count']
            new = get(rows, 'fixed128_source_full_schedule', width)
            old = get(rows, 'original_source_full_schedule', width)
            targets = [v for v in SCHEDULE if v > width]
            assert targets == new['target_widths'] == old['target_widths']
            if family == 'gemini':
                seed_rows = json.loads((OUT / family / 'results.json').read_text())
                assert targets == get(seed_rows, 'alternate_seed_source_full_schedule', width)['target_widths']
            graphs, hashes = [], {}
            for target in targets:
                path = folder / 'graphs' / f'fixed_source_{width}_{target}.npz'
                graph = load_npz(_paper_location(path))
                assert graph.shape == (width, target)
                metadata = json.loads(path.with_suffix('.json').read_text())
                assert metadata['source_sha256'] == source_meta['sha256']
                assert metadata['target_sha256'] == checkpoints[str(mainpath(family, target))]['sha256']
                assert metadata['threshold'] == 0.7
                graphs.append(graph)
                hashes[str(path.relative_to(ROOT))] = sha(path)
            with np.load(_paper_location(folder / f'fixed128_source_full_schedule_witness_{width}.npz')) as z:
                validate_witness(graphs, z['source'], [z[f'target_{v}'] for v in targets])
                assert len(z['source']) == new['count']
            selected, _, _ = persistent_matching(graphs)
            assert len(selected) == new['count']
            baseline_graphs = [load_npz(_paper_location(BASELINE / family / f'graph_{width}_to_{v}.npz')) for v in targets]
            selected, _, _ = persistent_matching(baseline_graphs)
            assert len(selected) == old['count']
            for row in (same, old, new):
                assert row['proportion'] == row['count']/width
            if width == 32768:
                previous = json.loads((OUT / family / 'results.json').read_text())
                assert new['count'] == get(previous, 'fixed128_five_widths', width)['count']
            summary.append(dict(family=family, width=width, original_k=SCHEDULE[width], source_k=128,
                                target_widths=','.join(map(str, targets)), cross_sparsity_matches=same['count'],
                                original_persistence=old['count'], fixed_source_persistence=new['count'],
                                difference_pp=100*(new['proportion']-old['proportion'])))
            checks.append(dict(family=family, width=width, target_widths=targets,
                               maximum_same_width_matching_certified=True,
                               source_and_target_fingerprints_verified=True,
                               original_and_replacement_persistence_recomputed=True, graph_sha256=hashes))
    with (DEST/'summary.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    lines = [r'\begin{table}[htbp]', r'    \centering', r'    \small',
             r'    \caption{Fixed-sparsity source replacement, using the same full-sweep target sets as the seed control. Only the source changes from its original $k$ to $k=128$; all original larger-width targets are retained. Same-width counts match the original and replacement source dictionaries. All comparisons use signed decoder cosine similarity $\geq0.7$; percentages divide by source width.}',
             r'    \label{tab:fixed-sparsity-persistence}', r'    \begin{tabular}{llccc}',
             r'        \toprule',
             r'        Model & \shortstack{Original\\width/$k$} & \shortstack{Same-width\\cross-sparsity matches} & \shortstack{Original source\\persistence} & \shortstack{$k=128$ source\\persistence} \\',
             r'        \midrule']
    markdown=['# Fixed-sparsity source-swap control', '',
              'Only the source SAE is replaced by its k=128 checkpoint; every original larger-width target remains unchanged. Thus widths 512, 4096 and 32768 have exactly 8, 5 and 2 targets, respectively, as in the independent-seed source-swap experiment. Checkpoint seeds and checked training settings are matched; sample order is not asserted identical.', '',
              '| Model | Original width/k | Same-width cross-sparsity matches | Original source persistence | k=128 source persistence |',
              '|---|---|---|---|---|']
    for fi, family in enumerate(('gemini', 'nemotron')):
        if fi: lines.append(r'        \midrule')
        for wi, width in enumerate(FIXED[family]):
            records=[get(all_results[family], name, width) for name in
                     ('cross_sparsity_same_width','original_source_full_schedule','fixed128_source_full_schedule')]
            columns=[family.capitalize() if wi==0 else '', f'{width:,}/{SCHEDULE[width]}']
            lines.append('        '+' & '.join(columns+[count_pct(r, True) for r in records])+r' \\')
            markdown.append('| '+' | '.join([family.capitalize(), f'{width:,}/{SCHEDULE[width]}']+[count_pct(r) for r in records])+' |')
    lines += [r'        \bottomrule', r'    \end{tabular}', r'\end{table}', '']
    (DEST/'table.tex').write_text('\n'.join(lines))
    markdown += ['',
        'Persistence is the exact intersection of independently selected pairwise maximum matchings into every original larger dictionary. Same-width matching is certified by an equal-size minimum vertex cover. Saved persistence intersections and both original and replacement pairwise maxima are validated. Graph metadata binds every source and target to its checkpoint SHA-256. Signed cosine uses float32 with TF32 disabled.', '',
        'This tests source sparsity while keeping the target sequence fixed; it is distinct from holding k=128 across the entire five-width control panel. The 65K source is unchanged and need not be swapped; 131K has no larger target.', '',
        'From the source directory, run `python paper.py run fixed-source-gemini --config /path/to/resources.json`, `python paper.py run fixed-source-nemotron --config /path/to/resources.json`, then `python paper.py run fixed-source-table --config /path/to/resources.json`.', '']
    (DEST/'README.md').write_text('\n'.join(markdown))
    (DEST/'validation.json').write_text(json.dumps(dict(checks=checks,passed=True),indent=2)+'\n')
    print('\n'.join(markdown[:13]),flush=True)


if __name__=='__main__':
    main()
