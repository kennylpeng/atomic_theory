#!/usr/bin/env python3
"""Show exact top corpus texts and WikiArt images for eight Gemini color features."""
from project_paths import resource_path as _paper_path, resource_location as _paper_location
import csv
import json
from pathlib import Path
import sys
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image, ImageOps

ROOT = _paper_path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.artwork_selection import collect_top_images, load_metadata, DEFAULT_INDEX
RESULTS = ROOT / 'experiments/colors/results'
WIKIART = ROOT / 'full_experiments/results/wikiart_feature_enrichment/gemini_m131072_k128_full3072'
TOP = 5


def shorten(text, width, lines):
    return '\n'.join(textwrap.wrap(' '.join(text.split()), width=width, max_lines=lines, placeholder='…'))


def main():
    order = json.loads((RESULTS / 'gemini_colors_summary.json').read_text())['features_in_row_order']
    text_path = RESULTS / 'selected_features_general_corpus_examples/top_examples.json'
    texts = json.loads(text_path.read_text())
    indices = np.load(_paper_location(WIKIART / 'wikiart_feature_indices.npy'), mmap_mode='r')
    values = np.load(_paper_location(WIKIART / 'wikiart_activation_values.npy'), mmap_mode='r')
    assert indices.shape == values.shape == (116261, 128)
    top = collect_top_images(indices, values, order, TOP)
    metadata = load_metadata(DEFAULT_INDEX, len(indices))
    report = dict(model='gemini_m131072_k128', text_source=str(text_path),
                  text_corpus_rows=texts['corpus_rows'], wikiart_rows=len(indices),
                  wikiart_index=str(DEFAULT_INDEX), wikiart_activation_cache=str(WIKIART),
                  ranking='Raw post-TopK activation, descending, independently within each corpus',
                  text_deduplication='Exact full text, using the previously computed top-10 report',
                  image_selection='Exact top 5 positive activations across all WikiArt cache rows; no color filtering',
                  features=[])
    for feature in order:
        examples = texts['features'][str(feature)][:TOP]
        assert len(examples) == TOP and len(top[feature]) == TOP
        artworks = []
        for rank, (activation, row) in enumerate(top[feature], 1):
            source = metadata[row]
            mask = indices[row] == feature
            assert mask.sum() == 1 and float(values[row][mask][0]) == activation
            assert _paper_path(source['image_path']).is_file()
            artworks.append({**source, 'rank': rank, 'activation': activation, 'vector_index': row})
        report['features'].append(dict(feature_id=feature, texts=examples, wikiart=artworks))

    families = ['DejaVu Sans']
    for family in ('Noto Sans CJK SC', 'Droid Sans Fallback'):
        try:
            font_manager.findfont(font_manager.FontProperties(family=family), fallback_to_default=False)
        except ValueError:
            continue
        families.append(family)
        break
    with plt.rc_context({'font.family': families, 'pdf.fonttype': 42}):
        fig = plt.figure(figsize=(22, 22))
        grid = fig.add_gridspec(8, 7, width_ratios=[0.85, 5.2, 3, 3, 3, 3, 3],
                               left=0.015, right=0.99, bottom=0.015, top=0.985,
                               wspace=0.10, hspace=0.16)
        for row_index, feature in enumerate(report['features']):
            label_ax = fig.add_subplot(grid[row_index, 0])
            label_ax.set_axis_off()
            label_ax.text(0, 0.96, f"f{feature['feature_id']}", fontsize=11, weight='bold', va='top')
            text_ax = fig.add_subplot(grid[row_index, 1])
            text_ax.set_axis_off()
            for i, example in enumerate(feature['texts']):
                excerpt = textwrap.shorten(' '.join(example['text'].split()), width=87, placeholder='…')
                label = textwrap.fill(f"{excerpt} ({example['activation']:.4f})", width=52)
                text_ax.text(0, 0.96 - i * 0.20, label,
                             fontsize=11, va='top', linespacing=1.15)
            for column, artwork in enumerate(feature['wikiart'], 2):
                inner = grid[row_index, column].subgridspec(2, 1, height_ratios=[1, 0.09], hspace=0.025)
                image_ax = fig.add_subplot(inner[0, 0])
                with Image.open(artwork['image_path']) as original:
                    picture = ImageOps.exif_transpose(original).convert('RGB')
                    picture.thumbnail((900, 900), Image.Resampling.LANCZOS)
                    image_ax.imshow(picture)
                image_ax.set_axis_off()
                caption_ax = fig.add_subplot(inner[1, 0])
                caption_ax.set_axis_off()
                caption_ax.text(0.5, 1, f"({artwork['activation']:.4f})",
                                ha='center', va='top', fontsize=10)
        assert fig._suptitle is None and not fig.texts
        assert all(not ax.lines for ax in fig.axes)
        output = RESULTS / 'gemini_color_feature_examples'
        fig.savefig(output.with_suffix('.png'), dpi=180)
        fig.savefig(output.with_suffix('.pdf'))
        plt.close(fig)
    output.with_suffix('.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    flat = [dict(feature_id=f['feature_id'], **a) for f in report['features'] for a in f['wikiart']]
    with (RESULTS / 'gemini_color_feature_examples_wikiart.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(f'Created {output}.png and .pdf: 8 features, 40 exact top texts, 40 exact top WikiArt images.')


if __name__ == '__main__':
    main()
