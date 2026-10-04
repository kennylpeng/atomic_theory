"""Select balanced, distinct text examples using the paper sampling rules."""

from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import bisect

from pathlib import Path

from scripts.paired_examples import GEMINI_NUMERIC, GEMINI_TEXT, GEMINI_MODEL, NEMOTRON_NUMERIC, NEMOTRON_TEXT, NEMOTRON_MODEL, load_sample_arrays

from scripts.sample_io import load_texts

from scripts.shared_ranges import shared_ranges

ROOT = _paper_path(__file__).resolve().parents[1]

PACKAGE = ROOT / 'full_experiments/plots/platonic_features_131k_pearson'

METRICS = ROOT / 'full_experiments/results/cross_model_activation_matches/gemini_nemotron_m131072_pearson/relaxed_jaccard_full_distribution/per_pair.csv'

PAIRS = (
    (71850, 119331, 'Clarifying the meaning or scope of a term'),
    (25799, 102037, 'Whether algebraic transformations preserve solutions'),
    (76558, 5300, 'Interior carpentry and home repair'),
    (103169, 54566, 'Introductions framing a practical problem'),
    (31660, 42453, 'School activities and educational offerings'),
)

def contains_excluded_script(text):
    """Detect Han, kana, Hangul, and associated CJK script characters."""
    ranges = (
        (0x1100, 0x11FF), (0x2E80, 0x2FFF), (0x3005, 0x3007),
        (0x3021, 0x3029), (0x3031, 0x3035), (0x3038, 0x303B),
        (0x3040, 0x318F), (0x31A0, 0x31FF), (0x3200, 0x33FF),
        (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xA960, 0xA97F),
        (0xAC00, 0xD7FF), (0xF900, 0xFAFF), (0xFF65, 0xFFDC),
        (0x16FE0, 0x16FFF), (0x1AFF0, 0x1AFFF), (0x1B000, 0x1B16F),
        (0x1F200, 0x1F2FF), (0x20000, 0x323AF),
    )
    return any(lo <= ord(char) <= hi for char in text for lo, hi in ranges)

def select_balanced_examples(pairs, low_count=6, high_count=6, preserve_existing=False, supplement=None):
    """Select distinct shared-row texts by priority, optionally extending saved samples."""
    ranges = shared_ranges()
    for model, numeric, text_root, model_name, shared in (
        ('gemini', GEMINI_NUMERIC, GEMINI_TEXT, GEMINI_MODEL, ranges[0]),
        ('nemotron', NEMOTRON_NUMERIC, NEMOTRON_TEXT, NEMOTRON_MODEL, ranges[1]),
    ):
        manifest, arrays = load_sample_arrays(numeric, model_name)
        starts = [start for start, _ in shared]
        candidates, requested = {}, set()
        for pair in pairs:
            feature = pair[model + '_feature']
            groups = {'low': [], 'high': []}
            for b in range(1, 7):
                for slot in range(int(arrays['sample_sizes'][feature, b])):
                    row = int(arrays['sample_rows'][feature, b, slot])
                    ri = bisect.bisect_right(starts, row) - 1
                    if ri < 0 or row >= shared[ri][1]:
                        continue
                    example = dict(bin_index=b, bin_label=manifest['bin_labels'][b],
                                   global_row=row, activation=float(arrays['sample_activations'][feature,b,slot]),
                                   sample_priority=int(arrays['sample_priorities'][feature,b,slot]))
                    groups['low' if b == 1 else 'high'].append(example)
                    requested.add(row)
            candidates[feature] = groups
        print(f'Loading {len(requested)} candidate texts for {model}', flush=True)
        _, texts = load_texts(text_root, numeric/'manifest.json', requested)
        for pair in pairs:
            feature = pair[model + '_feature']
            existing = pair.get(model + '_examples', []) if preserve_existing else []
            selected = list(existing)
            seen = {' '.join(e['text'].split()) or '[EMPTY]' for e in selected}
            if len(seen) != len(selected):
                raise ValueError(f'Duplicate saved texts for {model} {feature}')
            allocation = ({'low': low_count + high_count, 'high': 0}
                          if model == 'gemini' and feature == 71850
                          else {'low': low_count, 'high': high_count})
            pair[model + '_allocation'] = allocation
            for group, pool in candidates[feature].items():
                existing_count = sum((e['bin_index'] == 1) == (group == 'low') for e in existing)
                required = allocation[group] - existing_count
                if required < 0:
                    raise ValueError(f'Saved {model} {feature} sample exceeds {group} allocation')
                if required == 0:
                    continue
                chosen = []
                for e in sorted(pool, key=lambda e: (e['sample_priority'], e['global_row'])):
                    e['text'] = texts[e['global_row']]
                    if contains_excluded_script(e['text']):
                        continue
                    norm = ' '.join(e['text'].split()) or '[EMPTY]'
                    if norm in seen:
                        continue
                    seen.add(norm)
                    chosen.append(e)
                    if len(chosen) == required:
                        break
                if len(chosen) < required and supplement is not None:
                    chosen.extend(supplement(model, feature, group, required - len(chosen), seen))
                if len(chosen) != required:
                    raise ValueError(f'{model} {feature}: only {len(chosen)} distinct shared texts without excluded scripts in {group} range; need {required}')
                selected.extend(chosen)
            pair[model + '_examples'] = sorted(selected, key=lambda e: e['activation'], reverse=True)
