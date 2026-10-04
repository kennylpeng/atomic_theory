#!/usr/bin/env python3
"""Expand the twelve paper Pearson example tables to 15 texts per model.

The saved 12-example selections are immutable inputs. Extend them by reservoir
priority, verify both activations on aligned rows, and render two tables per page.
Use --render-only to rebuild TeX/CSV from the verified 360-row selection.
"""
from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path

from scripts.build_diverse_pearson_appendix import multilingual_cells
from scripts.example_table import excerpt, render_table

ROOT = _paper_path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'full_experiments/plots/platonic_features_131k_pearson'
SOURCES = (PACKAGE / 'main_examples/selection_and_examples.json',
           PACKAGE / 'diverse_appendix_examples/selection_and_examples.json')
OUTPUT = PACKAGE / 'appendix_examples_15'
FONT_PREAMBLE = r'''% Font coverage for sampled multilingual text; requires fontspec and bidi.
\newfontfamily\PlatonicBengali[Script=Bengali]{Noto Sans Bengali}
\newfontfamily\PlatonicArabic[Script=Arabic]{DejaVu Sans}
\newfontfamily\PlatonicUnicode{DejaVu Sans}
'''
SAMPLING = (
    'Fifteen distinct shared-row texts per feature/model. Preserve the original twelve '
    'and add two from (0.05,0.1] and one above 0.1, selected in ascending reservoir '
    'priority after whitespace-normalized text deduplication and full-text CJK-script '
    'exclusion. This gives eight lower-range and seven higher-range examples. '
    'Exception: Gemini 71850 has no cached activations above 0.1 and uses fifteen '
    'lower-range examples. If a reservoir is exhausted, scan shared corpus shards in order '
    'and select distinct eligible texts by a seeded random priority within each shard '
    '(seed 20260925 + feature ID). Sort each model sample by its source activation.'
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def original_pairs():
    first, additional = [json.loads(p.read_text()) for p in SOURCES]
    pairs = copy.deepcopy(first['pairs'][:2] + additional['pairs'])
    assert len(pairs) == 12
    for index, p in enumerate(pairs):
        p['paper_table_number'] = index + 14
        p['label'] = (f'tab:platonic-pearson-semantic-combined-{index+1}' if index < 2
                      else f'tab:platonic-pearson-diverse-{index-1}')
    return pairs


def validate(pairs, originals):
    assert len(pairs) == 12
    for p, seed in zip(pairs, originals):
        for key in ('gemini_feature', 'nemotron_feature', 'pearson', 'interpretation', 'label'):
            assert p[key] == seed[key], key
        for model in ('gemini', 'nemotron'):
            examples = p[model + '_examples']
            assert len(examples) == 15
            assert len({' '.join(e['text'].split()) for e in examples}) == 15
            allocation = {'low': 15, 'high': 0} if model == 'gemini' and p['gemini_feature'] == 71850 else {'low': 8, 'high': 7}
            assert p[model + '_allocation'] == allocation
            assert sum(0.05 < e['activation'] <= 0.1 for e in examples) == allocation['low']
            assert sum(e['activation'] > 0.1 for e in examples) == allocation['high']
            by_row = {e['global_row']: e for e in examples}
            for e in seed[model + '_examples']:
                assert by_row[e['global_row']] == e, 'Original sample changed'
            for e in examples:
                assert abs(e[model + '_activation'] - e['activation']) <= 5e-5
                assert all(math.isfinite(e[m + '_activation']) and e[m + '_activation'] >= 0
                           for m in ('gemini', 'nemotron'))


def supplement_from_corpus(model, feature, group, count, seen):
    """Fill an exhausted reservoir from seeded candidates in corpus shard order."""
    import numpy as np
    from project_paths import get_path
    from scripts.cache_binned_example_text_catalog import BulkTextResolver
    from scripts.example_sampling import contains_excluded_script
    from scripts.paired_examples import (
        GEMINI_CACHE, NEMOTRON_CACHE, GEMINI_MODEL, NEMOTRON_MODEL, FixedTopKShard,
    )
    roots = {'gemini': GEMINI_CACHE, 'nemotron': NEMOTRON_CACHE}
    names = {'gemini': GEMINI_MODEL, 'nemotron': NEMOTRON_MODEL}
    other = NEMOTRON_CACHE if model == 'gemini' else GEMINI_CACHE
    shared = {s['relative_shard'] for s in json.loads((other / 'plan.json').read_text())['source_shards']}
    manifest = json.loads((roots[model] / 'manifest.json').read_text())
    resolver = BulkTextResolver(_paper_path(manifest['embeddings_dir']), _paper_path(get_path('datasets_dir')))
    rng = np.random.default_rng(20260925 + feature)
    chosen = []
    print(f'Supplementing {model} {feature}: {count} {group}-range texts from full cache', flush=True)
    for record in manifest['shards']:
        if record['relative_shard'] not in shared:
            continue
        shard = FixedTopKShard(roots[model], record['relative_shard'], names[model], 131072)
        local, slot = np.nonzero(np.asarray(shard.indices) == feature)
        values = np.asarray(shard.data[local, slot])
        valid = (values > 0.05) & ((values <= 0.1) if group == 'low' else (values > 0.1))
        local, values = local[valid], values[valid]
        if not len(local):
            continue
        priorities = rng.integers(0, 2**63, size=len(local), dtype=np.int64)
        global_rows = local + record['global_row_start']
        texts = {}
        for batch in resolver.resolve_batches(record, global_rows, 512):
            texts.update(zip(batch['global_row'].to_pylist(), batch['full_text'].to_pylist()))
        for position in np.argsort(priorities, kind='stable'):
            row = int(global_rows[position])
            text = texts[row]
            normalized = ' '.join(text.split()) or '[EMPTY]'
            if normalized in seen or contains_excluded_script(text):
                continue
            seen.add(normalized)
            activation = float(values[position])
            from scripts.cache_binned_random_examples import BIN_LABELS, BIN_UPPER_BOUNDS
            bin_index = int(np.searchsorted(BIN_UPPER_BOUNDS, np.float32(activation), side='left'))
            chosen.append(dict(bin_index=bin_index, bin_label=BIN_LABELS[bin_index], global_row=row,
                               activation=activation, sample_priority=int(priorities[position]),
                               text=text, sampling_origin='supplemental full-cache sample',
                               relative_shard=record['relative_shard'], shard_row=int(local[position]),
                               supplemental_seed=20260925 + feature))
            if len(chosen) == count:
                return chosen
    raise ValueError(f'Not enough distinct {group}-range corpus texts for {model} {feature}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    originals = original_pairs()
    provenance = [dict(path=str(p.relative_to(ROOT)), sha256=sha256(p)) for p in SOURCES]
    selection_path = OUTPUT / 'selection_and_examples.json'
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.render_only:
        selection = json.loads(selection_path.read_text())
        assert selection['original_selections'] == provenance
        pairs = selection['pairs']
    else:
        from scripts.example_sampling import select_balanced_examples
        from scripts.paired_examples import (
            attach_paired_activations, GEMINI_CACHE, NEMOTRON_CACHE,
            GEMINI_NUMERIC, NEMOTRON_NUMERIC,
        )
        pairs = copy.deepcopy(originals)
        select_balanced_examples(pairs, low_count=8, high_count=7, preserve_existing=True,
                                 supplement=supplement_from_corpus)
        print('Verifying both activations on 360 sampled rows', flush=True)
        attach_paired_activations(pairs)
        selection = dict(
            original_selections=provenance, correlation_rows=89227558,
            feature_count_per_model=131072, examples_per_feature=15, rows_per_table=30,
            sampling=SAMPLING,
            correlations='Feature pairs, interpretations, and full-corpus Pearson values are '
                         'unchanged from the original selections; only displayed samples are expanded.',
            paired_activations='Read directly from both sparse post-TopK caches at the same relative '
                               'shard and local row; reservoir source activations verified to 5e-5.',
            cache_provenance=[dict(path=str(p), sha256=sha256(p)) for p in (
                GEMINI_CACHE / 'plan.json', NEMOTRON_CACHE / 'plan.json',
                GEMINI_NUMERIC / 'manifest.json', NEMOTRON_NUMERIC / 'manifest.json')],
            color_scale=dict(minimum=0, maximum=0.15, color='orange', shared=True),
            excerpt_characters=96, pairs=pairs,
        )
    validate(pairs, originals)
    tables, flat = [FONT_PREAMBLE], []
    for index, p in enumerate(pairs):
        rows = [dict(source_model=m, source_display_rank=i, **e, excerpt=excerpt(e['text']))
                for m in ('gemini', 'nemotron')
                for i, e in enumerate(p[m + '_examples'], 1)]
        table = render_table(p, rows).replace(
            'scripts/example_table.py', 'scripts/build_pearson_appendix_tables.py'
        ).replace('main_examples/selection_and_examples.json',
                  'appendix_examples_15/selection_and_examples.json'
        ).replace(f"tab:platonic-pearson-semantic-combined-{p['paper_example_index']}", p['label']
        ).replace(r'\begin{table*}[tp]', r'\begin{table*}[p]'
        ).replace(r'\renewcommand{\arraystretch}{1.0}', r'\renewcommand{\arraystretch}{1.1}')
        tables.append(multilingual_cells(table))
        if index % 2 == 1:
            tables.append(r'\clearpage')
        flat.extend(dict(paper_table_number=p['paper_table_number'], gemini_feature=p['gemini_feature'],
                         nemotron_feature=p['nemotron_feature'], interpretation=p['interpretation'],
                         pearson=p['pearson'], **r) for r in rows)
    selection_path.write_text(json.dumps(selection, indent=2, ensure_ascii=False) + '\n')
    (OUTPUT / 'tables.tex').write_text('\n'.join(tables) + '\n')
    with (OUTPUT / 'examples_full_text.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in flat for k in row)))
        writer.writeheader()
        writer.writerows(flat)
    readme = ['# Pearson appendix tables: 15 examples per model', '', SAMPLING, '',
              selection['correlations'], '', selection['paired_activations'], '',
              'Each table has 30 rows, divided after the 15 Gemini-source examples. Two tables '
              'are grouped on each float page. Text excerpts and the shared orange activation '
              'scale match the paper tables. Fonts: Noto Sans Bengali and DejaVu Sans.', '',
              '| Table | Interpretation | Gemini | Nemotron | Pearson r |',
              '| ---: | --- | ---: | ---: | ---: |']
    readme.extend(f"| {p['paper_table_number']} | {p['interpretation']} | {p['gemini_feature']} | "
                  f"{p['nemotron_feature']} | {p['pearson']:.6f} |" for p in pairs)
    readme += ['', 'From the source directory, regenerate with '
               '`python paper.py run semantic-examples --config /path/to/resources.json`; use `python paper.py run semantic-render --config /path/to/resources.json` to '
               'render the saved verified examples. Original full texts, corpus rows, '
               'reservoir priorities, cache hashes, and unrounded activations are retained.', '']
    (OUTPUT / 'README.md').write_text('\n'.join(readme))
    print(f'Wrote {len(pairs)} tables and {len(flat)} rows to {OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
