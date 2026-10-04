# Gemini dates summary figure

The final figure is `results/gemini_dates_summary.png`,
with a `results/gemini_dates_summary.pdf` and
`results/gemini_dates_summary.json`.
[plot_gemini_dates_summary.py](plot_gemini_dates_summary.py) places a raw embedding
PCA and a 131K month-feature PCA side by side, above the month-feature heatmap.

The data, checkpoints, cached results, and generated images named below are
external resources. Configure their locations through the
[paper runner](../../REPRODUCTION.md); paths below are logical
locations in its prepared work tree.

## Inputs and selected features

The inputs are all 366 month/day strings from `January 1` through `December 31`,
including `February 29`; no year is included in the strings.
[run.py](run.py) defines this sequence. All three panels use this same date order.
The cached raw Gemini embeddings have shape 366 × 3,072 and are stored in
`results/gemini_embeddings.npy`.

The 12 month features are selected in the seed-42 65,536-feature
Gemini reference SAE.
[plot_gemini_smaller_dictionaries.py](plot_gemini_smaller_dictionaries.py) maps
each one to its maximum decoder-cosine match in `gemini_m131072_k128` (131,072
features, TopK 128). These are decoder matches, not a new prevalence ranking.

| Reference 65K feature | Matched 131K feature |
|---|---|
| 54190 | 3530 |
| 11053 | 28411 |
| 29452 | 9754 |
| 13219 | 84886 |
| 61549 | 27997 |
| 45772 | 61387 |
| 63515 | 112643 |
| 11312 | 126249 |
| 18738 | 5344 |
| 17281 | 121554 |
| 44782 | 18953 |
| 15540 | 25960 |

The matching script computes the full encoder's ReLU scores, applies TopK 128,
and extracts the selected features, giving zero when a feature is outside
TopK. It saves the 366 × 12 matrix, feature IDs, date strings, and matching
cosines in
`results/gemini_smaller_dictionaries/m131072/matched_feature_activations.npz`.
The sibling `summary.json` records the checkpoint and matching results.

## PCA panels

The left panel shows PC1 and PC2 of the raw embeddings. The upstream script
[plot_gemini_date_embeddings_pca.py](plot_gemini_date_embeddings_pca.py) fits three
components to the cached float32 embeddings using scikit-learn PCA. It centers
the dimensions without standardizing or additionally normalizing them. The
summary takes the first two coordinate columns from
`results/gemini_date_embeddings_pca/gemini_date_embeddings_pca3.npz`.

The right panel shows PC1 and PC2 of the 366 × 12 raw post-TopK feature matrix.
[plot_gemini131k_month_pca.py](plot_gemini131k_month_pca.py) fits two components
with the full SVD solver and saves
`results/gemini_smaller_dictionaries/m131072/gemini_month_features_pca2.npz`.
**Features are mean-centered, not standardized.** No activation threshold,
feature scaling, or date filtering is applied. PC1 and PC2 explain approximately
14.51% and 13.72% of the feature variance; the raw embedding values are 11.01%
and 10.37%. These percentages are omitted from the final axes by design.

The summary does not refit either PCA. It checks that dates and selected-feature
IDs agree across the saved inputs. Each point is a day, colored by its month
using `twilight_shifted` sampled at `0/12, 1/12, ..., 11/12`. Both panels use
identical colors, plain PC1/PC2 labels, no connecting lines, and no legends.
They have equal square panel sizes and equal physical scaling of PC1 and PC2
within each panel; the two panels do not share numerical axis limits.

## Heatmap and month strip

The heatmap uses the same 366 × 12 raw post-TopK matrix as the feature PCA,
without centering, standardization, binarization, or row normalization.
Columns are dates in calendar order. The color scale interpolates linearly from white (`#ffffff`) at zero to
orange (`#ff8000`) at activation 0.2, with larger values saturated at orange.
Both summary heatmaps use this same fixed 0–0.2 range, matching the orange
used in the a-prefix molecule LaTeX table.

Rows are sorted by their activation-weighted circular mean day position:
`theta[j] = 2*pi*j/366`, followed by
`arg(sum_j activation[i,j] * exp(1j*theta[j])) mod 2*pi`. Thus December and
January are adjacent when computing a feature's mean. Sorting is stable;
zero or effectively balanced resultants are placed last. The final order is
the 131K feature order in the table above.

The labeled Jan.–Dec. strip is both the column guide and the PCA color key.
Its segments have widths proportional to month lengths:
31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 days. It uses exactly the same
month colors as the scatterplots. Segment boundaries lie on column edges,
and the strip and heatmap share horizontal coordinates and bounds.

## Reproduction

Install the paper dependencies and prepare a work directory as described in
[REPRODUCTION.md](../../REPRODUCTION.md). From the source directory, render
from the saved PCA coordinates and activations:

```bash
python paper.py script --config /path/to/resources.json -- experiments/day_of_year/plot_gemini_dates_summary.py
```

To rebuild the upstream PCA artifacts from cached inputs first (also requires
scikit-learn; the baseline script imports the experiment's PyTorch utilities):

```bash
python paper.py script --config /path/to/resources.json -- experiments/day_of_year/plot_gemini_date_embeddings_pca.py
python paper.py script --config /path/to/resources.json -- experiments/day_of_year/plot_gemini131k_month_pca.py
python paper.py script --config /path/to/resources.json -- experiments/day_of_year/plot_gemini_dates_summary.py
```

To regenerate the 131K decoder matches and activations, configure `models_dir`
and `date_reference_models_dir`, then run through the configured launcher:

```bash
python paper.py script --config /path/to/resources.json -- experiments/day_of_year/plot_gemini_smaller_dictionaries.py --widths 131072
```

This needs the seed-42 `d65536_k128_f32` reference run (distinct from
the main sweep's 65K model), the 131K checkpoint, and the ordered embeddings.
`--reference-run` can override the reference location. Rebuild both PCA inputs
after regenerating activations; the summary draws its heatmap from that matrix.

The combined figure uses a compact layout with a heatmap row half the height
of the PCA row and reduced panel spacing. It is 14 × 9 inches, exported as a 200-dpi PNG and a PDF.
The script checks matching outer edges between the PCA and heatmap rows,
equal PCA panel sizes, identical month colors, alignment of all 366 columns,
absence of PCA legends/lines, and plain PC1/PC2 labels. Reusing saved PCA
coordinates preserves the saved orientation; signs can differ after an
upstream PCA refit without changing the underlying geometry.
