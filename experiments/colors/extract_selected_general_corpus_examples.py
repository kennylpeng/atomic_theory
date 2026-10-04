"""Extract exact top distinct texts for the eight selected Gemini 131K neurons."""
from project_paths import resource_path as _paper_path, resource_location as _paper_location
import concurrent.futures
import heapq
import json
from pathlib import Path
import time
import numpy as np

ROOT = _paper_path(__file__).resolve().parents[2]
OUT = ROOT / 'experiments/colors/results/selected_features_general_corpus_examples'
CACHE = _paper_path('/resources/activation_cache_dir/gemini_all_corpus_post_topk')
FEATURES = [57539, 71717, 13055, 78457, 129556, 96080, 124857, 1397]
LIMIT = 1000

def scan_shard(record):
    path = _paper_path(record['cache_shard_dir']) / 'gemini_m131072_k128'
    ids = np.load(_paper_location(path / 'indices.npy'), mmap_mode='r')
    values = np.load(_paper_location(path / 'data.npy'), mmap_mode='r')
    lookup = np.zeros(131072, dtype=bool)
    lookup[FEATURES] = True
    positions = np.flatnonzero(lookup[ids])
    selected_ids = ids[positions]
    selected_values = values[positions]
    result = {}
    for feature in FEATURES:
        idx = np.flatnonzero((selected_ids == feature) & (selected_values > 0))
        if len(idx) > LIMIT:
            idx = idx[np.argpartition(selected_values[idx], -LIMIT)[-LIMIT:]]
        result[feature] = [(float(selected_values[i]), record['relative_shard'], int(positions[i] // 128)) for i in idx]
    return result

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((CACHE / 'manifest.json').read_text())
    assert manifest['complete']
    heaps = {feature: [] for feature in FEATURES}
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for number, result in enumerate(pool.map(scan_shard, manifest['shards']), 1):
            for feature, candidates in result.items():
                heap = heaps[feature]
                for item in candidates:
                    if len(heap) < LIMIT:
                        heapq.heappush(heap, item)
                    elif item > heap[0]:
                        heapq.heapreplace(heap, item)
            if number % 25 == 0:
                print(f'{number}/{len(manifest["shards"])} shards, {time.monotonic()-start:.0f}s', flush=True)
    report = dict(source_manifest=str(CACHE / 'manifest.json'), corpus_rows=manifest['expected_rows'], datasets=manifest['datasets'], activation_formula=manifest['activation_formula'], candidate_limit=LIMIT, candidates={str(f): [dict(activation=a, shard=s, shard_row=r) for a,s,r in sorted(h, reverse=True)] for f,h in heaps.items()})
    (OUT / 'candidates.json').write_text(json.dumps(report, indent=2))
    print('Scan complete', flush=True)

def resolve():
    import sys
    sys.path.insert(0, str(ROOT))
    from scripts.corpus_io import TextResolver
    from project_paths import get_path
    resolver = TextResolver(_paper_path(get_path('gemini_embeddings_dir')), _paper_path(get_path('datasets_dir')))
    report = json.loads((OUT / 'candidates.json').read_text())
    features = {}
    candidates_by_feature = report.pop('candidates')
    heatmap_order = [124857, 96080, 78457, 1397, 13055, 129556, 71717, 57539]
    for feature_id in heatmap_order:
        feature = str(feature_id)
        candidates = candidates_by_feature[feature]
        examples, seen = [], set()
        for candidate in candidates:
            item = resolver.resolve(candidate)
            # Read the complete source string for exact text deduplication.
            source_path = _paper_path(get_path('datasets_dir')) / item['dataset']
            if item['config'] is not None:
                source_path = source_path / ('config=' + item['config'])
            loaded = resolver.dataset_cache[str(source_path)]
            from datasets import DatasetDict
            dataset = loaded[item['split']] if isinstance(loaded, DatasetDict) else loaded
            raw_text = dataset[item['source_row']][item['text_column']]
            item['text'] = str(raw_text).strip() if raw_text is not None else ''
            if not item['text'] or item['text'] in seen:
                continue
            seen.add(item['text'])
            item['rank'] = len(examples) + 1
            examples.append(item)
            if len(examples) == 10:
                break
        assert len(examples) == 10, f'Insufficient distinct candidates for {feature}'
        assert examples[-1]['activation'] > candidates[-1]['activation'], 'Candidate boundary could affect distinct top 10'
        features[feature] = examples
        print(f"Feature {feature}: " + ' | '.join(f"{x['activation']:.4f}: {x['text'][:180]}" for x in examples[:3]), flush=True)
    report['features'] = features
    (OUT / 'top_examples.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    lines = ['# Top general-corpus examples for selected Gemini 131K neurons', '',
             f"Corpus: {report['corpus_rows']:,} rows across {len(report['datasets'])} datasets.",
             'Activations: raw post-TopK. Exact duplicate texts removed within each neuron.', '']
    for feature, examples in features.items():
        lines += [f'## Feature {feature}', '']
        for x in examples:
            lines += [f"### {x['rank']}. Activation {x['activation']:.6f} — {x['dataset']} / {x['text_column']}", '', x['text'], '',
                      f"Source: `{x['shard']}`, shard row {x['shard_row']}, source row {x['source_row']}.", '']
    (OUT / 'top_examples.md').write_text('\n'.join(lines))

if __name__ == '__main__':
    import sys
    if '--resolve' in sys.argv:
        resolve()
    else:
        main()
