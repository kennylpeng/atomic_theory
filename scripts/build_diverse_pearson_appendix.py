#!/usr/bin/env python3
"""Build ten diverse, higher-correlation analogs of the semantic MNN tables.

Use the configured paper launcher to resample and verify cached activations;
--render-only uses the saved, verified examples and needs only standard Python.
"""
from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

from scripts.example_table import excerpt, render_table

ROOT = _paper_path(__file__).resolve().parents[1]
MATCHES = ROOT / 'full_experiments/results/platonic_mnn_updated_sketches/sae_131072_mnn_pairs.csv'
OUTPUT = ROOT / 'full_experiments/plots/platonic_features_131k_pearson/diverse_appendix_examples'
# Deliberately selected after reviewing cached texts, ordered by Pearson r.
PAIRS = (
    (31660, 42453, 'School activities and educational offerings'),
    (23469, 114122, 'Trigonometric half-angle identities'),
    (64628, 31241, 'Telescopes and astronomical optics'),
    (61225, 124167, 'Trade tariffs and protectionism'),
    (96133, 76359, 'Comments in source code'),
    (76787, 40156, 'Mechanical bearings'),
    (59059, 17701, 'Butter'),
    (61020, 89987, 'Sign languages'),
    (123114, 97094, 'Eurovision Song Contest entries'),
    (104756, 51458, 'Astragalus and milkvetch species'),
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(pairs, matches):
    assert len(pairs) == 10
    assert len({p['gemini_feature'] for p in pairs}) == 10
    assert len({p['nemotron_feature'] for p in pairs}) == 10
    for p, (g, n, interpretation) in zip(pairs, PAIRS):
        assert (p['gemini_feature'], p['nemotron_feature']) == (g, n)
        assert p['interpretation'] == interpretation
        assert p['pearson'] == matches[g, n]
        assert 0.5 <= p['pearson'] <= 0.9
        for model in ('gemini', 'nemotron'):
            examples = p[model + '_examples']
            assert len(examples) == 12
            assert len({' '.join(e['text'].split()) for e in examples}) == 12
            assert sum(0.05 < e['activation'] <= 0.1 for e in examples) == 6
            assert sum(e['activation'] > 0.1 for e in examples) == 6
            for e in examples:
                assert abs(e[model + '_activation'] - e['activation']) <= 5e-5
                assert all(math.isfinite(e[m + '_activation']) and e[m + '_activation'] >= 0
                           for m in ('gemini', 'nemotron'))


def multilingual_cells(table):
    """Preserve sampled text while giving non-Latin cells suitable fonts."""
    def replace(match):
        text = match.group(1)
        if any('\u0980' <= c <= '\u09ff' for c in text):
            text = re.sub(r'[\u0980-\u09ff\u200c\u200d]+',
                          lambda run: r'{\PlatonicBengali ' + run.group() + '}', text)
        elif any('\u0600' <= c <= '\u06ff' for c in text):
            text = r'\RL{\PlatonicArabic ' + text + '}'
        elif any(ord(c) > 0x024f for c in text):
            text = r'{\PlatonicUnicode ' + text + '}'
        return '{' + text + '} & '
    return re.sub(r'^\{(.*?)\} & ', replace, table, flags=re.MULTILINE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    with MATCHES.open(newline='') as f:
        matches = {(int(r['gemini_feature_id']), int(r['nemotron_feature_id'])):
                   float(r['pearson']) for r in csv.DictReader(f)}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selection_path = OUTPUT / 'selection_and_examples.json'
    if args.render_only:
        selection = json.loads(selection_path.read_text())
        assert selection['matching_source_sha256'] == sha256(MATCHES)
        pairs = selection['pairs']
    else:
        from scripts.example_sampling import select_balanced_examples
        from scripts.paired_examples import (
            attach_paired_activations, GEMINI_CACHE, NEMOTRON_CACHE,
            GEMINI_NUMERIC, NEMOTRON_NUMERIC,
        )
        pairs = [dict(paper_example_index=i, gemini_feature=g, nemotron_feature=n,
                      interpretation=label, pearson=matches[g, n])
                 for i, (g, n, label) in enumerate(PAIRS, 1)]
        select_balanced_examples(pairs)
        print('Verifying both activations on 240 sampled rows', flush=True)
        attach_paired_activations(pairs)
        selection = dict(
            matching_source=str(MATCHES.relative_to(ROOT)),
            matching_source_sha256=sha256(MATCHES),
            matching_method='Mutual top-one neighbors in 512-dimensional CountSketch / '
                            '32-candidate searches, with full-corpus Pearson rescoring.',
            correlation_rows=89227558,
            feature_count_per_model=131072,
            selection='Ten manually selected pairs spanning Pearson 0.5--0.9 and diverse themes. '
                      'Illustrative, not a representative sample; interpretations are qualitative.',
            sampling='Twelve distinct shared-row texts per model: six from (0.05,0.1] and six '
                     'above 0.1, selected by ascending cached reservoir priority, deduplicated '
                     'by whitespace-normalized text, and sorted by source activation. Full texts '
                     'containing Han, kana, Hangul, or Bopomofo are excluded, as in Tables 14--15. '
                     'These are activation-bin samples, not global top-activating texts.',
            paired_activations='Read directly from both sparse post-TopK caches at the same '
                               'relative shard and local row; source activations checked against '
                               'the reservoirs with absolute tolerance 5e-5.',
            cache_provenance=[dict(path=str(p), sha256=sha256(p)) for p in (
                GEMINI_CACHE / 'plan.json', NEMOTRON_CACHE / 'plan.json',
                GEMINI_NUMERIC / 'manifest.json', NEMOTRON_NUMERIC / 'manifest.json')],
            color_scale=dict(minimum=0, maximum=0.15, color='orange', shared=True),
            excerpt_characters=96, pairs=pairs,
        )
    validate(pairs, matches)
    tables = [r'''% Font coverage for sampled multilingual text; requires fontspec and bidi.
\newfontfamily\PlatonicBengali[Script=Bengali]{Noto Sans Bengali}
\newfontfamily\PlatonicArabic[Script=Arabic]{DejaVu Sans}
\newfontfamily\PlatonicUnicode{DejaVu Sans}
''']
    flat = []
    for p in pairs:
        rows = [dict(source_model=m, source_display_rank=i, **e, excerpt=excerpt(e['text']))
                for m in ('gemini', 'nemotron')
                for i, e in enumerate(p[m + '_examples'], 1)]
        table = render_table(p, rows).replace(
            'scripts/example_table.py', 'scripts/build_diverse_pearson_appendix.py'
        ).replace('main_examples/selection_and_examples.json',
                  'diverse_appendix_examples/selection_and_examples.json'
        ).replace('tab:platonic-pearson-semantic-combined-', 'tab:platonic-pearson-diverse-')
        tables.append(multilingual_cells(table))
        flat.extend(dict(gemini_feature=p['gemini_feature'], nemotron_feature=p['nemotron_feature'],
                         interpretation=p['interpretation'], pearson=p['pearson'], **r) for r in rows)
    selection_path.write_text(json.dumps(selection, indent=2, ensure_ascii=False) + '\n')
    (OUTPUT / 'tables.tex').write_text('\n'.join(tables))
    with (OUTPUT / 'examples_full_text.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    readme = ['# Diverse higher-Pearson appendix examples', '', selection['selection'], '',
              selection['sampling'], '', selection['paired_activations'], '',
              'Correlations use all 89,227,558 aligned corpus rows, independently of the displayed '
              'samples. Both activation columns use the original tables\' shared orange scale '
              '(white at 0, saturated at 0.15), three decimals, and 96-character excerpts.', '',
              'Font dependencies: Noto Sans Bengali and DejaVu Sans, with fontspec and bidi. '
              'Bengali and Arabic cells use script-aware fonts; Arabic cells use right-to-left text.', '',
              '| Theme | Gemini | Nemotron | Pearson r |', '| --- | ---: | ---: | ---: |']
    readme.extend(f"| {p['interpretation']} | {p['gemini_feature']} | {p['nemotron_feature']} | "
                  f"{p['pearson']:.6f} |" for p in pairs)
    readme += ['', 'From the source directory, regenerate with '
               '`python paper.py script --config /path/to/resources.json -- scripts/build_diverse_pearson_appendix.py`. Re-render saved examples with '
               '`python paper.py script --config /path/to/resources.json -- scripts/build_diverse_pearson_appendix.py --render-only`. '
               'The saved JSON records the match-file hash, cache manifests, original full texts, '
               'row identifiers, reservoir priorities, and unrounded activations.', '']
    (OUTPUT / 'README.md').write_text('\n'.join(readme))
    print(f'Wrote {len(pairs)} tables and {len(flat)} rows to {OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
