# Gemini colors summary figure

The final figure is `results/gemini_colors_summary.png`,
with a `results/gemini_colors_summary.pdf` and
`results/gemini_colors_summary.json`.
[plot_gemini_colors_summary.py](plot_gemini_colors_summary.py) assembles two 3D PCA
panels above an eight-feature activation heatmap using cached inputs.

The data, checkpoints, cached results, and generated images named below are
external resources. Configure their locations through the
[paper runner](../../REPRODUCTION.md); paths below are logical
locations in its prepared work tree.

## Inputs and selected features

The input table is `data/color_names_hex.csv`. Each of
its 865 rows supplies a hex-string embedding input (for example, `#5d8aa8`) and
RGB values used to color the scatter points. There are 765 distinct hex values.
The summary selects the same **374 distinct filtered colors for all three
panels**, keeping the first input occurrence of each hex value.

The SAE is Gemini `gemini_m131072_k128`: 131,072 features, TopK 128. The selected
features come from ranks 6, 7, 8, 10, 11, 12, 15, and 16 in
`data/color_names_hex_gemini_d131072_top100_prevalent_neurons_thr0p05.tsv`.
The 0.05 threshold belongs to feature selection; it is not applied to the plotted
activation values.

| Selection rank | Feature |
|---|---|
| 6 | 57539 |
| 7 | 71717 |
| 8 | 13055 |
| 10 | 78457 |
| 11 | 129556 |
| 12 | 96080 |
| 15 | 124857 |
| 16 | 1397 |

## How the 865 input rows become 374 colors

The filter selects strongly saturated colors and excludes near-neutral, dark,
and very pale colors. It depends only on the input RGB values, not on neuron
activations or PCA coordinates. The filter uses the thresholds below.

First, deduplicate hex values case-insensitively, retaining the first original
row for each color. Different names with the same hex value count as one color.
Divide each 8-bit RGB channel by 255, then use Python's `colorsys.rgb_to_hsv`
and `colorsys.rgb_to_hls` to compute saturation S, value V, and lightness L.
All three quantities lie in [0, 1]. Retain a color exactly when:

```python
S >= 0.60 and V >= 0.50 and L <= 0.70
```

The counts below were checked against `data/color_names_hex.csv`. Thresholds
are shown sequentially to explain the filter; the implementation combines them with AND.

| Step | Rows/colors remaining | Removed at this step |
|---|---:|---:|
| Original color-name rows | 865 | — |
| Deduplicate hex values | 765 | 100 |
| Require HSV saturation S ≥ 0.60 | 428 | 337 |
| Also require HSV value V ≥ 0.50 | 374 | 54 |
| Also require HLS lightness L ≤ 0.70 | **374** | 0 |

S measures saturation relative to the brightest RGB channel and excludes
washed-out or gray colors. V is the maximum RGB channel and excludes dark
colors. HLS lightness is `(max(R, G, B) + min(R, G, B)) / 2`, which limits
very pale colors. At these particular cutoffs, the lightness condition removes
no additional colors: S ≥ 0.60 already implies L ≤ 0.70 for RGB in [0, 1].
It remains explicit in the filter to record the pale-color cutoff.

Thus 391 of the 765 distinct colors are excluded, leaving **374 distinct
colors**. The retained colors are ordered by increasing HSV hue, with decreasing
saturation and then increasing value as tie-breakers. This sorting changes only
order, not membership. `chromatic_indices()` in
[plot_selected_color_feature_swatches.py](plot_selected_color_feature_swatches.py)
implements the threshold rule. The summary uses the same first-occurrence source
rows for the raw embeddings and selected-feature activations, fits both PCAs on
these 374 rows, and uses them as the heatmap's 374 columns. The exact retained
hexes and original row indices are stored as `hexes` and `source_rows` in
`results/gemini_colors_summary_pca.npz`.

## PCA panels

Both PCAs are fitted on exactly the 374 distinct colors retained by the heatmap
filter above, in its hue order. Both use scikit-learn PCA with three
components and the full SVD solver, mean-centering only: no standardization,
normalization, or whitening.

The left panel uses the 374 × 3,072 subset of cached raw embeddings from
`data/gemini_embeddings/embeddings.npy`. Row IDs in the accompanying `rows.csv`
are checked against the activation export. The first original input row for
each retained hex is selected.

The right panel uses the 374 × 8 raw post-TopK activation matrix. The summary
checks that selected original activation rows match the heatmap CSV exactly
within numerical tolerance. The original activation matrix and row metadata
come from
`results/color_names_hex_gemini_d131072_selected_ranks_thr0p05_pca`,
produced by [plot_gemini_sae_color_pca.py](plot_gemini_sae_color_pca.py).
Missing sparse entries are zero; no additional 0.05 activation cutoff is used.

The fitted coordinates, components, means, explained variance ratios, source
row indices, retained hexes, and feature IDs are saved in
`results/gemini_colors_summary_pca.npz`.

Points use their actual RGB colors, with dark outlines and no depth shading
or connecting lines. Both views use elevation 24° and azimuth 40°, with 3D box
proportions set by the displayed coordinate ranges. The displayed x, y, and z
axes are PC1, PC2, and PC3, respectively, matching the saved coordinate order. Explained variance is recorded
in the summary JSON and PCA archive rather than on the axes.

## Heatmap

The heatmap reads
`results/color_names_hex_gemini_d131072_selected_ranks_thr0p05_swatches/selected_features_hue_activations.csv`.
[plot_selected_color_feature_swatches.py](plot_selected_color_feature_swatches.py)
defines its data preparation and filter:

1. Deduplicate hex values case-insensitively, keeping the first input occurrence.
2. Sort chromatic colors by HSV hue, then decreasing saturation, then value.
3. Retain colors with HSV saturation ≥ 0.60, HSV value ≥ 0.50, and HLS lightness
   ≤ 0.70, using RGB channels scaled to [0, 1]. This leaves **374 columns**.

**Both PCA panels are also fitted and plotted using precisely these 374 colors.**

Each heatmap cell is a raw post-TopK activation, without standardization,
binarization, or row normalization. The shared scale interpolates linearly from white (`#ffffff`) at zero to
orange (`#ff8000`) at activation 0.2, with larger values saturated at orange.
Both summary heatmaps use this same fixed 0–0.2 range, matching the orange
used in the a-prefix molecule LaTeX table. The top strip shows the exact hex color of
each column, with identical horizontal positions and bounds.

Rows are sorted by activation-weighted circular mean of displayed column
position. For N retained columns, let `theta[j] = 2*pi*j/N`; feature i has mean
angle `arg(sum_j activation[i,j] * exp(1j*theta[j])) mod 2*pi`. Sorting these
angles treats the two ends of the hue axis as adjacent. This averages displayed
positions, not numerical hue values. Rows with zero or effectively balanced
resultants are placed last; ties use stable sorting.

The resulting feature order is 124857, 96080, 78457, 1397, 13055, 129556, 71717,
57539.

## Reproduction

Install the paper dependencies and prepare a work directory as described in
[REPRODUCTION.md](../../REPRODUCTION.md). From the source directory, run:

```bash
python paper.py script --config /path/to/resources.json -- experiments/colors/plot_gemini_colors_summary.py
```

This refits both PCAs using cached embeddings, selected activations, and the
heatmap CSV described above; it performs no embedding API calls
or SAE inference. To rebuild the selected activation matrix and metadata from
cached sparse activations first:

```bash
python paper.py script --config /path/to/resources.json -- experiments/colors/plot_gemini_sae_color_pca.py --variant selected_ranks_thr0p05
python paper.py script --config /path/to/resources.json -- experiments/colors/plot_gemini_colors_summary.py
```

The summary reads `selected_features_hue_activations.csv`. The standalone swatch
script writes files with a `selected_ranks_thr0p05_` prefix; supply the CSV at the
input path above when regenerating swatches.

The combined figure uses a compact layout with a heatmap row half the height
of the PCA row and reduced panel spacing. It is 14 × 9 inches, exported as a 200-dpi PNG and a PDF.
The script checks equal PCA panel dimensions, matching outer row edges, and
alignment of every heatmap column with the top strip. It also checks that the
PCA panels have no legends or connecting lines. PCA axis signs may change when refitting in a different numerical environment,
without changing the underlying geometry.

## General-corpus and WikiArt example figure

`results/gemini_color_feature_examples.png`
and its `results/gemini_color_feature_examples.pdf` show the same eight
features in summary heatmap order. Each row contains the top five distinct
corpus texts and the top five WikiArt images, ranked independently by raw
post-TopK activation. Texts are shortened only for display and followed by activation scores in
parentheses. Images show only their activation scores in parentheses. The
figure has no title, column headers, rankings, dataset labels, feature underlines,
or artwork credits; full source texts and artwork credits remain in the JSON/CSV.

The text examples reuse
`results/selected_features_general_corpus_examples/top_examples.json`,
which scanned 89,827,558 general-corpus rows and removed exact duplicate texts
within each feature. They are not samples from activation bins.

The image examples are selected by a complete scan of the existing
116,261 × 128 sparse WikiArt activation arrays in
`full_experiments/results/wikiart_feature_enrichment/gemini_m131072_k128_full3072/`.
Only positive stored activations are candidates; a top-five heap is maintained
for each feature. Image rows are not restricted by the 374-color filter, artwork
style, or enrichment ranking. Ties are resolved by decreasing vector index.
The cache uses the native 3,072-dimensional Gemini embeddings and the
`gemini_m131072_k128` checkpoint, without new model inference.

The embedding index at
`runs/full_batch_3072/embeddings_index.csv` beneath the configured `wikiart_dir` maps each
activation row to its local image, title, artist, and source URL. Images retain
their aspect ratios and are resized only for presentation; no artwork content
is generated. The text and image ranking scores share an activation definition
but are ranked separately within their respective corpora.

With the resources prepared, run from the source directory:

```bash
python paper.py script --config /path/to/resources.json -- experiments/colors/plot_gemini_color_feature_examples.py
```

This needs NumPy, Matplotlib, Pillow, the saved text report, WikiArt activation
arrays, index, and local images. A local CJK fallback font is used when available
for the Japanese example. The script verifies index coverage, image existence,
and each selected image's cached score. It exports an 22 × 22 inch PNG at
180 dpi and a PDF, plus
`results/gemini_color_feature_examples.json`
and `results/gemini_color_feature_examples_wikiart.csv`.

The paper's LaTeX table is generated by
[render_color_examples_table.py](render_color_examples_table.py) from the frozen
`sources.json` selections. Use the `color-examples` workflow to reproduce it.
It contains five texts and five images per feature, with 40 artwork JPEGs and
three Unicode text snippets supplied externally. The output works with pdfLaTeX
and `graphicx`; it does not repeat the corpus scans or choose new examples.
