# Training, evaluation, and figure reproduction

This guide describes the resources and commands used to train the models,
conduct the evaluations, and reproduce empirical results in the paper and appendix.
Table exporters can write TeX snippets without a TeX compiler.

Use `python paper.py guide` for the paper-to-code map and `python paper.py list`
for stages grouped by family. `registry.json` is the single source for commands,
inputs, outputs, scientific methods and asset definitions. Numerical conventions
are included at the end of this guide.

Supply the data, model weights, embeddings, sparse activation caches, curated
example selections, artwork, and fonts required by your workflow. Saved-result
bundles support numerical checks without training. Illustrative examples require
the saved selections, labels, and row IDs.

## Install and choose asset locations

Use Python 3.10 or later. The reference numerical environment uses NumPy 2.2.6,
SciPy **1.15.3** and PyTorch 2.7.1. SciPy's version matters for nonunique matching
choices. CUDA, BLAS and floating-point near-ties can affect training/inference;
use the original checkpoints and cached embeddings for exact paper comparisons.

```bash
python -m pip install -c constraints.txt .
python -m pip install '.[paper,training]'
cp resources.example.json /path/to/my-resources.json
python paper.py list
```

Run workflow commands from the source directory. FAISS GPU is needed for large
KMeans fitting and candidate-search jobs.
Corpus regeneration additionally uses `datasets`, `pyarrow`, `google-genai`,
`transformers`, `sentence-transformers`, and `wordfreq` through the providers extra.
Embedding generation requires provider credentials and may incur API charges;
saved-result analysis and local rendering do not require embedding API calls.
Set `GEMINI_API_KEY` in the environment for Gemini embedding calls. The runner
redacts credential argument values from printed plans and saved command reports.

Set `work` to a **new writable directory**, preferably on a large scratch volume.
Each `paths` value can point anywhere; relative values resolve against the JSON
configuration's directory, not the shell's working directory. Empty entries are
unset. For example:

```json
{
  "work": "/path/to/my-paper-run",
  "paths": {
    "models_dir": "/models/atomic-checkpoints",
    "hierarchy_data_dir": "/datasets/hierarchy",
    "gbif_metadata_dir": "/datasets/gbif-metadata",
    "results_bundle": "/data/atomic-results",
    "candidates_bundle": "/data/atomic-candidates"
  },
  "inputs": {
    "full_experiments/results/persistent_stability_signed_0.7": "/results/cosine-graphs",
    "full_experiments/results/persistent_stability_pearson_0.7": "/results/gemini-pearson",
    "full_experiments/results/nemotron_persistent_stability_pearson_0.7": "/results/nemotron-pearson"
  }
}
```

`inputs` maps a logical repository-relative **file or directory** to its actual
location. Add the input groups for the workflows you want, using the tables below
and `registry.json`. There is no requirement to put assets inside the code
repository. You can override each intermediate file independently. The logical
input destinations describe the analysis: controls use
`full_experiments/results/sparsity_seed_controls`, and prefix molecule probes use
`probe_results/prefix_molecule_selection` beneath `hierarchy_data_dir`.
Map your resource files to these destinations through `inputs` or configure a
`path_aliases` entry for an individual logical path.

```bash
python paper.py prepare --config /path/to/my-resources.json
python paper.py plan persistence --config /path/to/my-resources.json
python paper.py run persistence --config /path/to/my-resources.json
```

`prepare` copies source and explicitly selected intermediates into `work`; it
does not run an experiment. Source paths beginning with `/resources/` are logical
aliases, not directories you need to create. The resource resolver maps these aliases and
explicit file-opening calls to the configured locations. Saved JSON/NPZ
metadata is left byte-for-byte intact, including hashes; paths read from that
metadata are resolved at use. Staged source hashes and the configuration are
saved in `reproduction_state.json`. Input files are copied into the work directory;
budget disk space for these copies. Put large corpus/model roots in `paths`, rather than copying them through
`inputs`. Some **compute** workflows intentionally write new caches beneath
configured cache roots; use a writable cache destination separate from sealed
reference assets when regenerating them.

If an external manifest records paths from another machine, add a private
`path_aliases` object to your resource configuration, mapping each recorded absolute
root to a key in `paths`. For example, `"/source-machine/models": "models_dir"`
relocates recorded checkpoint paths to the configured models directory. The
manifest bytes and checksums stay intact. Keep machine-specific configurations
outside the source directory.

Python and shell files found inside asset directories are not staged as code.
The fixed-source control checks script hashes recorded in each run's
`COMPLETE.json`. When a recorded hash differs from the source in the work
directory, supply the matching Python snapshot as an explicit file in `inputs`, with a
destination such as `data/source_snapshots/scripts/review_controls/directions.py`.
The runner appends `.source` to the destination filename for hash verification.
These snapshots are checked as data and are not executed.

`plan` prints argument arrays and missing declared resources without executing.
`run` records stdout/stderr, commands, exit codes and declared output hashes in
`work/run_logs/`; it stops at the first failed step. The registry distinguishes
`train`, `compute`, `recompute`, and `render`. The input
checks are an initial preflight; individual producers validate more detailed
file layouts. Do not use Python `-O`: scientific validators use assertions.
For another configuration or source revision, prepare a new directory.

For upstream programs beyond the named workflows, `script` supplies the same
configured paths, working directory, Python environment and execution logs:

```bash
python paper.py script --config /path/to/my-resources.json -- scripts/compute_gemini_131k_cooccurrence.py --help
python paper.py script --config /path/to/my-resources.json -- scripts/generate_model_configs.py --include-controls
WANDB_MODE=disabled python paper.py script --config /path/to/my-resources.json -- trainer/train.py --config generated_model_configs/gemini_all_d512_k32_aux0p25_steps1m_seed1138983768.yaml --setup-config setup_configs/gemini.yaml --output-dir trained/gemini_m512_k32
```

The generator writes all 35 recipes without reading data. The last command trains
one model; training resolves the `gemini.yaml` embedding root through
`paths.gemini_embeddings_dir` at load time (use the Nemotron counterpart for that family).
W&B logging is disabled in this example. `--output-dir` chooses the new checkpoint
location; preserve each run's seed/settings and supply its resulting checkpoint
under the logical model names expected by later analyses. Arguments after `--`
belong to that program.

For multilingual example rendering, supply **Noto Sans CJK SC** (including Hangul) and
**Noto Sans Thai**, **Noto Sans Bengali**, plus DejaVu Sans and an Arabic-capable font such as Amiri.
Droid Sans Fallback alone does not cover the paper's Korean examples.
Fonts can live anywhere: set `FONTCONFIG_FILE=/path/to/fonts.conf` to a Fontconfig
file containing their directory before invoking the runner. Set `XDG_CACHE_HOME`
to a writable font-cache location if needed. The table renderer rejects missing
glyphs. Reference font hashes are in `release/provenance/font-resources.json`; font
binaries remain external.

## Resource inventory

Supply the resources required by your workflow. Preserve SHA256 hashes,
model/provider revisions, row ordering, preprocessing parameters, and split labels.

| Resource / configuration key | Required contents and exact identity | Used by |
| --- | --- | --- |
| `models_dir` | 26 PyTorch checkpoint **files** named `{gemini,nemotron}_m{width}_k{k}` and corresponding subset checkpoints. `W_enc`: `(d,m)`, `W_dec`: `(m,d)`, `b_dec`: `(d,)`; saved config/seed. Nine full widths 512,1024,2048,4096,8192,16384,32768,65536,131072 with k=32,32,32,32,64,64,64,128,128. Four extra 16,384/k64 dataset-mixture models per family. Do not substitute the NPY model bundle for these PyTorch checkpoint files without conversion. | All empirical SAE inference, decoder graphs, molecules |
| `control_models_dir` | Six additional k128 checkpoints: widths 512/4096/32768 for each family, plus three Gemini alternate-seed checkpoints at original k32/k32/k64. Supply checkpoint files named `{family}_m{width}_k{k}_seed{seed}.pt`. Exact seeds and reference SHA-256 hashes are in `release/provenance/control-checkpoints.json`; `scripts/control_models.py` resolves these names for both decoder and hierarchy controls. These bring the main width/distribution experiments and controls to **35 SAE checkpoints**. Regenerating the date-feature selection also requires the separate date-feature reference listed below. | Tables 5–7, Figs. 11/16/20 |
| `date_reference_models_dir` | `d65536_k128_f32` / Nemotron counterpart for date-feature discovery. The Gemini reference is a seed-42 65K checkpoint, distinct from the main full-sweep 65K model, used for date-feature decoder matches. | Date feature selection |
| `gemini_embeddings_dir`, `nemotron_embeddings_dir` | Ordered training/evaluation shards plus index/metadata files, dtype and text-column information. Gemini: 89,827,558 vectors, 929 shards, d=3072, float16, provider metadata `gemini-embedding-2-preview`, `RETRIEVAL_DOCUMENT`. Nemotron: 89,227,558 vectors, 923 shards, d=4096. Preserve the six missing 100K Gemini/Nemotron MIRACL-text shard difference and the shared-row alignment. | Training, PCA, KMeans, corpus activations/Pearson |
| `datasets_dir` | Exact prepared text rows: emotion, fever, gooaq, hotpotqa, msmarco, natural-questions, nfcorpus, sampled PAQ, scifact, squad, trivia-qa, WebInstructSub, MIRACL text/title collections. PAQ: 10M records sampled with seed 42, emitting question and answer separately. Preserve language/config/split/text-column/source-row identifiers. The 13 dataset families yield 14 prepared collections because MIRACL text/title are separate. | Corpus regeneration and text examples |
| `baselines_dir` | Ten PCA fits (two embedding families × five mixtures), with all d components, means, eigenvalues and float64 sufficient statistics. Eight active KMeans models (two families × cluster counts 512,4096,16384,131072), full-corpus **100-iteration** centroids, metadata, objective histories and assignment counts. KMeans resource paths within results can instead be supplied individually using `inputs`. | Fig. 2b, Figs. 10–11/13 |
| `hierarchy_data_dir` | `wordfreq`, `geonames_country_questions`, `geonames_continent_questions`, `gbif`; each has original ordered `rows.csv`, embeddings per model and saved `split`. Original per-model/width sparse files `sparse_{model}_m{width}_k{k}.npz` have `row_indices`, `feature_indices`, `values`, `shape`, `top_k`. `probe_results/{dataset}/{model}_m{width}_k{k}/summary.tsv`, combined `all_summaries.tsv`, and `probe_results/prefix_molecule_selection` for molecule/cosine selection. | P3 and prefix/molecule analyses |
| `gbif_metadata_dir` | Exact GBIF Taxon/Vernacular source snapshot and derived `category_threshold_f1_d131072/categories_min100_species.tsv`, `species_common_name_pairs.csv`, plus prepared species/name membership and split metadata read by target constructors. Scientific names:123,495; common-name strings:259,286; total:382,781 rows. | GBIF targets and adjacent-rank pairs |
| `activation_cache_dir` | Corpus post-TopK caches with plan/manifest/COMPLETE markers, shard identity, model identity, feature indices and values. Main Gemini and Nemotron cache trees, 16K models evaluated on every mixture, binned-example numeric caches and text catalogs. Fixed-TopK corpus arrays are **not** SciPy CSR or the hierarchy COO NPZ schema. | Pearson discovery, prevalence, containment and illustrative examples |
| `parent_child_dir` | `gemini_m131072_k128_gt_0p05_rlower_0p7`: full count/support arrays, qualifying ancestor-pair and reduced parent-child CSVs, block metadata and summary. Co-occurrence input is exact dense uint32 row blocks for the 131,072² matrix at activation >0.05. | 154,316 ancestor / 114,762 retained pairs; Food/Music |
| `research_data_dir`, `gemini_artifacts_dir` | Additional research data and Gemini text-label/export resources used by cache and example producers. Configure only if those upstream stages are used; the compact numerical checks do not need them. | Full corpus examples and labels |
| `wikiart_dir` | Original 116,261-row image index, source images/attribution and gemini-embedding-2 d3072 image embeddings. Image rows, content hashes and selected IDs must align with the activation cache. | Color text/image examples |
| `synthetic_runs_dir` / synthetic inputs | `experiments/recovery_principle/scaling/billion/configs/sweep.json`; 468 run directories containing truth dictionaries, model+optimizer+RNG checkpoints and configs; 11 milestones each, 5,148 recovery arrays and summaries. For replotting only: `results/checkpoints.csv`. For persistence verification: t0.7 and t0.8 manifests, 78 groups of six models, all graphs, assignments and per-group summaries. | Figs. 22–25 |
| `results_bundle`, `candidates_bundle` | Extracted results for independent pairwise matching and source-set intersection; optional candidate arrays containing both directions of 144 top-five neighbor searches. | Portable Figure 2 verification |
| `scratch_dir`, `local_cache_dir` | Writable training/cache scratch and local download cache. No fixed cluster, username, scheduler or conda environment is required by the runner. Large jobs can be dispatched per manifest index. | Compute/training stages |
| Curated selections and layout inputs | Food/Music `*_selection.json`, `*_edges.csv`, activating examples and `latex_examples/*_table.tex`; target-3290 family specification and selected text/cache row IDs; color `sources.json`, 40 local JPEGs and three Unicode text snippets or appropriate fonts; cross-model `main_examples` and `diverse_examples` selections. Supply these through `inputs`. | Exact illustrative tables/diagrams |
| Manifold inputs | Date ordered embeddings and PCA archives, 12 selected/matched feature IDs and 366×12 activations. Color ordered 865-row embedding table/array, selected eight-feature unthresholded activation matrix, feature metadata and hue table. Retain original hashes even though only 374 unique filtered colors are plotted. | Fig. 27 |
| Fonts | Unicode fonts are needed when rendering multilingual text into images: DejaVu/FreeSerif, Noto CJK, Amiri/Noto Arabic, Noto Thai or equivalents. `render_color_examples_table.py --font /path/font.otf` checks glyph coverage. Graphviz is needed by the Graphviz hierarchy renderers. | Color example images and hierarchy layouts |

### Compute and storage

The empirical SAEs use batch size 2048, 1M optimizer steps per model;
Gemini 131K uses 2M steps and auxiliary coefficient 1.0 instead of 0.25. This is
2.048B or 4.096B presentations, not a single epoch over 90M rows. Training used
A100 80GB or A6000 GPUs. Full training runs require substantial GPU time. The synthetic grid alone processes **479.232B** training
examples (468 × 1.024B). Budget checkpoint, optimizer and evaluation storage.

The two full embedding arrays alone occupy about **1.28 TB at float16** (2.57 TB
at float32), before indexes, intermediate activations and copies. The exact
131K×131K uint32 co-occurrence matrix is 64 GiB before block overhead. A 131K
Gemini encoder plus decoder needs about 3.22 GB in float32, before gradients,
Adam states and activations. Compact graph verification and most plotting fit on
a CPU machine; full-corpus caches and high-width inference do not have the same
resource needs. Check free space before staging intermediate trees.

## Figure, table and number reproduction map

Workflow IDs below are arguments to `paper.py run`. The exact
argument arrays and required logical locations are in
[registry.json](registry.json); inspect with `plan` first. The guide lists each figure/table asset and its producer. Source hashes are generated during `prepare` and recorded in the work directory
in `reproduction_state.json`. Generated asset hashes are recorded in `run_logs/`. Outputs are written **inside the work directory** at the paths listed in the registry.

| Paper result | Numerical reconstruction and figure/table workflow | Inputs / conventions to preserve |
| --- | --- | --- |
| Fig. 2a; Fig. 7; 72.6/69.4/59.7/66.4% at width 4096 | `main-candidates` verifies the graphs and saved memberships; `persistence` regenerates paper count curves. | Independent full-source maxima, then source-ID intersection. Signed cosine/Pearson >=0.7, all larger widths, SciPy 1.15.3. |
| Fig. 2b; SAE 26–59%, PCA max 15%/median 1% | `main-numbers` re-solves graphs; `split-directions` renders SAE/PCA heatmaps. | Five mixtures; width 16K SAE, full d-component PCA; signed SAE/absolute PCA cosine. |
| Fig. 2c; all 90 mean-F1 cells | `hierarchy-probes` refits from sparse caches; `hierarchy-original` reaggregates and checks confusion counts and reference five-decimal table; `main-numbers` independently checks compact saved fits. | 72 fits (four prompt/dataset collections × two models × nine widths). Means pool category-model pairs. |
| Fig. 3a | `prefix-recall`, then `prefix-example`. F1 uses saved held-out fits; recall is recomputed from the a-detector's threshold on all word rows. | Recall is **not held-out-only**; every child F1 has its own detector. |
| Fig. 3b, Fig. 21, Tables 8–9; 66 nodes/64 displayed edges | `compact-hierarchies`, `food-hierarchy`, `music-hierarchy`, `hierarchy-examples`; run `scripts/containment/fit_table_height.py` for final layout sizing. | Preserve curation and source texts. Edge computation is separate from labels and selection. |
| Tables 3–4; all training corpus counts/seeds | `tools/inventory_training_data.py`; compare every dataset/model count with recipe provenance and `trainer` configs. `scripts/generate_model_configs.py` generates recipes. | Exact source shards required for a fresh inventory. Existing metadata hashes do not hash full embedding contents. |
| Table 5; control checkpoint identities | `release/provenance/control-checkpoints.json` records the nine checkpoint identities and SHA-256 hashes. | Six fixed-k plus three alternate-seed models; 1M steps and auxiliary coefficient 0.25. |
| Table 6 | `fixed-source-gemini`, `fixed-source-nemotron`, then `fixed-source-table`. | Only source switches to k128; all original larger targets stay. |
| Table 7 | `seed-gemini`; compare same-width and full-target persistence with `sparsity_seed_controls/gemini/results.json`. | Three alternate Gemini seeds; original source counts 303/2974/15881 and alternate 293/2979/15998. Same-width 470/3564/17856. |
| Fig. 6 | `exclusive-persistence`; upstream exhaustive exclusive pair generation via `scripts/stability/dictionary_exclusive_matches.py`. | Strict signed cosine >0.7, both endpoint degrees 1, intersect all larger comparisons. |
| Figs. 8–9 | `paper_persistent_threshold_sweep.py init`; compute each `{cosine,pearson}×{gemini,nemotron}`; `thresholds`. | Thresholds 0.5/0.6/0.7/0.8; preserve original t0.7 decisions/graphs and candidate manifests. |
| Figs. 10–11 | Generate original median counts with `summarize_sae_zero_median.py`; fixed-k counts with `fixed_k128_sae_prevalence.py`; KMeans counts with `kmeans_prevalence_iter100.py` and `kmeans_512_iter100.py`; `prevalence`, `prevalence-k128`. | Strict activation > pooled median; zeros excluded from median, included in denominator. Four KMeans widths mandatory, all 100 iterations. |
| Fig. 12 and six quoted P1 AUCs | `review_controls/update_width_persistence.py` applies matched memberships to saved rates; `prevalence-association`, then `prevalence-panels`. | Use `threshold=median_positive`, `membership=witness`, appropriate all/nonzero population in `width_effects.csv`. |
| Fig. 13 | Regenerate exclusive PCA with `split_16384_match_heatmaps.py --representations pca --t1 .7 --t2 .7 --device cpu --out-dir full_experiments/results/split_matches/exclusive_pca_absolute_0.7/matches --plot-dir full_experiments/results/split_matches/exclusive_pca_absolute_0.7/plots`; then `split-directions`. | Absolute PCA cosine at **both** endpoints. Maximum 15.0716%, reported as 15%. |
| Fig. 14; Gemini 22.0–48.9%, Nemotron 29.3–60.7% | Recompute distribution-specific candidates/rescores with `match_split_sae_activation_sketch.py` and `rescore_split_sae_activations_shared.py`; `split-activations`. | Each row's mixture evaluates both models. Reversing a heatmap cell changes the data distribution. |
| Fig. 15 and eight directional AUCs | `scripts/review_controls/median_counts.py --task i` for all ten source models, then `prevalence-association` and `prevalence-panels`. | Same source SAE feature on both distributions; minimum of the two rates, fixed full-corpus median threshold. |
| Fig. 16 | `fixed-k-probes`, then `hierarchy-k128`. | Six new checkpoints; 24 new fits (6 Prefix and 18 other datasets); reuse 65K/131K fits. |
| Figs. 17–18; 242/93/499 categories and 216/87/453 pairs | `individual-recovery`, `parent-child-recovery`. | **F1>0.6**, exact confusion-count arithmetic; distinct coordinates per parent/child pair. Cities uses separate prompts. |
| Figs. 19–20 | `prefix-grid`, `prefix-grid-k128`. | 25 panels a–z excluding x, 216 eligible children overall, q has one child; original nine/fixed five widths. |
| Figs. 22–23 | `synthetic-design`, `synthetic-train`, `analyze.py`, `synthetic-recovery`. | 468 final trajectories; 5148 milestones. Three-seed means and ranges. Prefix permits 10% misses; hierarchical families count as 3 atoms; ratio can exceed 1. |
| Figs. 24–25; 93.6–96.1%/91.9–99.1% and 61.5→91.8% | `persistence.py prepare/run` for t0.7; `persistence_threshold.py prepare/run --threshold .8` for each 78 group indices; `synthetic-persistence`. | 400K shared held-out rows per group, exhaustive signed cosine/Pearson, no sketch, inactive coordinates remain in denominator. |
| Table 1 | `prefix-molecules` first runs `run_all_a_prefix_linear_reconstruction.py --basis-direction decoder` without test-F1 filtering, then the table generator. | 23 columns,22 distinct directions; all columns in fit, only five positive contributions displayed. Use the `prefix_molecule_selection` probe cohort. |
| Table 2; 17 terms/error 0.049297/cosine 0.975040 | `transform-fit` recomputes directly from checkpoint rows. `transform-examples` rebuilds example TeX from fixed selections. | OMP tolerance is **sqrt(0.05) in L2**, not 0.05 L2. |
| Table 11 | `transform-examples` exports example tables from fixed selections. | Twelve selected examples per illustrated atom; cached activations, separate from decoder coefficients; five-atom selectivity >=0.09/<0.05 for lower main-table block. |
| Fig. 26; medians 0.1438/0.0084 | `prefix-cosines-compute`, then `prefix-cosines`. | Compute cosine matrices from checkpoints; 231 unordered off-diagonal pairs at each width. |
| Fig. 27 dates | `day_of_year/run.py` for input/model analysis; `plot_gemini_date_embeddings_pca.py`, `plot_gemini131k_month_pca.py` / `plot_gemini_smaller_dictionaries.py` generate original PCA/activation caches; `dates`. | 366 month/day strings,12 features selected at 65K then cosine-mapped to 131K; cache orientation preserved by final renderer. |
| Fig. 27 colors | `plot_gemini_sae_color_pca.py` and `plot_selected_color_feature_swatches.py` generate selected activation inputs; `colors` refits both PCAs on the same 374 filtered unique colors. | Deduplicate hex first; saturation>=.60,value>=.50,lightness<=.70; eight features; mean-center, do not standardize or threshold plotted activations. |
| Table 12 | `extract_selected_general_corpus_examples.py` generates general-corpus ranks; use the saved image selections in `sources.json`; `color-examples` renders all 40 text/40 artwork activation labels. | The renderer uses the saved final selections. Frozen images and Unicode fonts/snippets are required; rendering does not recompute global ranks. |
| Fig. 28 | `match_cross_family_width_sketch.py` with aligned 89,227,558 rows, then `cross-model`. | Same-width pairs only, all nine widths; thresholds 0.5–0.8, exhaustive matching on retained graphs. |
| Tables 13–24 | `semantic-examples` samples from retained binned-text inputs and renders tables; `semantic-render` renders saved `appendix_examples_15/selection_and_examples.json`. | Twelve illustrative pairs, 15 texts per model; bins (0.05,0.1]/>0.1 and the documented 71850 exception. |

## From raw resources rather than saved intermediate results

These steps explain the upstream dependency chain. They require the full training
resources and compute budgets above. Invoke programs through the `script` launcher;
it supplies the prepared tree's `PYTHONPATH`, `ATOMIC_THEORY_PATHS` and
`ATOMIC_PATH_REMAP`. The renderer recipes above are shorter routes
when trusted intermediates already exist.

1. **Prepare and embed the training corpora.** Use `data_prep/00_download_data.py`,
   `00_sample_paq.py`, `00_sample_miracl.py`, then the Gemini/Nemotron embedding
   scripts. Preserve original source versions, shard boundaries and ordering.
   Exact stored Gemini embeddings cannot be promised from a newly served model
   endpoint. Archived vectors are required for bitwise comparison.
2. **Train SAEs.** Generate all 35 configs with `--include-controls` and invoke
   `trainer/train.py --config RECIPE --setup-config SETUP --output-dir DEST`
   through the configured launcher. Use saved exact
   seeds, sampling, sparsity, steps and auxiliary coefficients. The control
   scripts identify six fixed-k and three alternate-seed runs separately. A fresh
   run tests reproducibility statistically; it does not guarantee the identical
   atom ordering/weights on another accelerator.
3. **Fit PCA/KMeans.** `scripts/compute_embedding_full_pca.py --config CONFIG
   --output-dir DEST` streams all corresponding rows. KMeans uses
   `train_kmeans_full_corpus.py` / `train_kmeans_full_corpus_ram.py`, then
   `continue_full_corpus_kmeans_widths.py` through 100 iterations;
   `kmeans_512_iter100.py` handles 512. Keep model metadata, centroids, objective
   history, counts and validation markers.
4. **Compute activations and moments.** Entry points include
   `cache_full_sparse_activations.py`, `cache_all_experiment_sae_activations.py`
   and the Nemotron cache/view builders. Preserve COMPLETE markers and shared
   text-row alignment. Use the bounded stage interfaces exposed by each script;
   do not interchange CSR, hierarchy COO and fixed-TopK shard formats.
5. **Discover Pearson candidates.** `match_gemini_width_sketch.py`,
   `match_nemotron_width_sketch.py`, `match_cross_family_width_sketch.py`, and split
   counterparts expose prepare/sketch/candidates/rescore/reduce stages. Execute
   every declared task index. Shared hashes and rows,512 sketch dimensions,
   32 proposed candidates, exact full-corpus rescore, top 5 retention **in both
   directions**. Candidate arrays and moments remain necessary even when a
   threshold graph is small. Recompute graph matching with the paper workflows.
6. **Hierarchy probes.** Prepare embeddings and **preserve saved train/test splits**.
   Reconstruct sparse inputs with `hierarchy_data_prep/02_compute_sparse_activations.py
   --model-path CHECKPOINT --embeddings EMBEDDINGS --rows ROWS --output CACHE
   --top-k K --batch-size 2048 --device cuda --dtype float32`. Run `hierarchy-probes`
   and `fixed-k-probes`. The package `encode` writes CSR NPZ; it is not a substitute
   for the hierarchy COO cache without an explicit conversion. Thresholds and feature
   IDs are selected on selection rows; all reported F1 uses unchanged test rows.
7. **Containment.** `compute_gemini_131k_cooccurrence.py prepare --cache-root CACHE
   --output-root COUNTS --threshold .05`, then `compute --output-root COUNTS
   --block-id INDEX` for every block and `finalize --output-root COUNTS`.
   `find_gemini_131k_parent_child_pairs.py prepare --matrix-root
   COUNTS --output-root PAIRS` followed by every block's `compute` and `reduce`
   applies support>=100, Wilson lower>=.70, reverse<=.50, lift>=5 and the single
   two-edge shortcut removal. Frozen curation is still required for exact examples.
8. **Synthetic experiments.** `synthetic-design` creates 468 configs without data
   inputs. `synthetic-train` lists all 468 independent argv invocations; dispatch
   chosen indices on your scheduler if needed. `run.py` retains optimizer/RNG
   state; `analyze.py` reconstructs checkpoint summaries.
   Compute persistence for all 78 groups at both thresholds before its report.
9. **Render figures and check numerical results.** Run the desired renderers, then
   `python paper.py run paper-numbers --config /path/to/resources.json` after
   staging its declared numerical inputs. This checks saved scalar values and
   records their input hashes; upstream training is validated separately.

## Validation and exactness

Integer counts, selected feature IDs, graph assignments, category cohorts and
splits must agree exactly. The compact hierarchy table uses 5.1e-6 tolerance for
five-decimal entries. OMP/cosine values should agree at the quoted precision;
near-ties, SVD signs and BLAS can alter unrounded values or visual orientations.
Compare plotted numeric tables, not PDF byte hashes: metadata, fonts and renderer
versions change PDF bytes without changing the experiment.

The runner's `run_logs/` records which stages were executed and the hashes of
their declared outputs. Saved-graph optimization, probe reaggregation, inference
from checkpoints, and fresh training answer different reproducibility questions.
Use the workflow's declared stage and input requirements to distinguish them.
Fresh downloads or newly served embedding models need not reproduce the reference
rows, splits, or vectors. Exact comparisons require the specified frozen inputs.


## Workflow dependencies

Run `aggregate-probes` to produce `all_summaries.tsv` before prefix analyses.
`prefix-cosines-compute` calculates decoder similarities; `prefix-cosines` renders
them. Figure 11 uses `prevalence-k128`. The containment family provides separate
stages for food/music diagrams and example tables.

Most corpus and checkpoint producers require the training/provider extras even
when reading saved tensors. The synthetic training stage expands one indexed
command over all 468 trajectories. Use `paper.py script` with a single `--index`
to run one task on a scheduler. Run prerequisites explicitly using the registry's
input requirements and the upstream recipes above.

# Evaluation definitions and reproducibility

The code supports analysis of saved results, evaluation from checkpoints, and
training from prepared embeddings. The following sections define matching,
probe fitting, data formats, and numerical interpretation.

## Saved-input formats

| External bundle | What it supports |
| --- | --- |
| results | Recompute 32 persistence intersections, 60 split-graph optima / 120 directional comparisons, and 90 pooled hierarchy mean-F1 cells |
| candidates | Reconstruct 72 Pearson threshold graphs from both directions of 144 retained top-five arrays |
| models | Encode embeddings with the 26 main SAEs exported as NumPy arrays; recompute decoder similarities |
| evaluation | Four hierarchy datasets, ordered rows, existing splits, embeddings, and taxonomy/prefix targets |
| recipes | Training configurations and corpus/checkpoint metadata |

Bundles have manifests with file sizes and SHA256 hashes. The compact
`atomic-features reproduce` command checks inputs, re-solves saved graphs, and
reaggregates saved probe fits on CPU. It writes CSVs, PDF plots, a TeX table, and
`report.json`, and fails on numerical disagreement.

## Matching and persistence

Maximum matching uses inclusive thresholds **>= 0.7**: signed decoder cosine for
SAEs/KMeans, absolute cosine for PCA, and signed Pearson correlation for
activation candidates. Exclusive matching uses **> 0.7** and requires degree one
at both endpoints.

For each source width, compute a maximum matching independently on each full
source-to-target graph, then intersect the matched source IDs. Source features
are not prefiltered and matching choices are not coordinated across targets.
The compact implementation certifies pairwise optima with vertex covers.
Original feature ordering, sorted CSR storage, and **SciPy 1.15.3** specify the
paper's tie-breaking; both saved memberships and counts are checked.

Pairwise maximum cardinality is invariant to matching ties, but the intersection
of matched source sets can change. Adding edges or lowering the threshold need
not increase this persistence statistic. Pearson graphs retain the union of
both directions' candidate lists after exact rescoring. Their pairwise maximum
cardinalities are lower bounds on exhaustive matching; persistence intersections
and candidate-restricted mutual-nearest-neighbor counts have no such bound.

PCA supplies orthogonal, sign-invariant directions; the SAE dictionaries are
overcomplete and their cosine comparison is signed. Normalizing counts by width
does not remove these structural differences. Dataset-mixture partitions also
do not guarantee that identical text is absent across sources.

## Hierarchy probes

Feature and threshold selection uses training rows; F1 evaluation uses the saved
held-out rows. Threshold comparison is `>=`. The package exposes `fit_probe` and
`evaluate_probe` separately. Means pool category-model trajectories. Regret
against the best test F1 across widths is a descriptive comparison, not a model
selection rule learned on training data.

Original row order, target membership, and train/test assignments are required
for exact reproduction. Newly preparing Wordfreq/GeoNames with seed 1729 does
not recreate the saved assignments. GBIF eligibility uses source species
counts, while evaluation weights name rows. Common-name targets can combine
multiple species, including species in both partitions, so that evaluation is
not a fully species-disjoint holdout.

The a-prefix illustration computes recall on all word rows. Its child F1 values
use independent held-out detectors. Parent/child recovery requires distinct
coordinates within each pair, not across the complete set of categories. City
parent and child questions use different prompts.

## SAE inference and numerical comparisons

Inference is `topk(relu((x - b_dec) @ W_enc))`; reconstruction adds `b_dec` after
the decoder multiplication. There is no input normalization or active encoder
bias. NumPy inference selects smaller feature IDs for exact positive ties;
optional PyTorch inference uses `torch.topk`, matching the training convention.
Hardware, BLAS, and near-tie roundoff can change selected IDs. Evaluation
embeddings retain their original dtype, including float32 GBIF Gemini inputs.

Integer counts, selected IDs, graph memberships, category cohorts, and splits
must agree exactly. The compact hierarchy table allows 5.1e-6 for five-decimal
entries. Compare continuous values at their stated precision. PCA signs,
renderer versions, fonts, and PDF metadata can change visual orientation or
file hashes without changing the underlying result.

## Interpreting descriptive and selected analyses

Prevalence above a pooled positive median is constrained by dictionary width
and TopK sparsity: the mean is bounded near `k / (2m)`, with ties potentially
reducing it further. The prevalence AUCs describe in-sample, feature-level
association with membership in a selected maximum matching, rather than merely
the existence of a qualifying neighbor. Correlated features, duplicated rows,
and matching ties affect interpretation.

Containment uses support and Wilson-bound screening over many dependent pairs.
These thresholds do not provide familywise statistical guarantees. Rare-feature
correlations and recovery estimates should be read alongside their support
counts. Fixed-k and alternate-seed controls have limited scope: the seed study
uses one alternate Gemini run at each of three widths, and the largest models
in the fixed-k comparisons reuse the original training settings.

Manifold coordinates, molecule examples, semantic pairs, and Food/Music branches
were selected for illustration. Recreating them requires the saved selections,
labels, and row IDs; they are not population performance estimates. The full
17-term transformation fit has squared error 0.049297 and cosine 0.975040; the
five displayed positive terms alone do not inherit that error. The prefix
basis has 23 columns but only 22 distinct directions, so individual coefficients
are not unique. OMP follows a greedy path and does not establish global minimality.

Synthetic bands show ranges over three seeds, not confidence intervals. Prefix
recovery permits 10% misses, and its hierarchical counting can yield a ratio
above one. Exact continuation requires compatible model, optimizer, and RNG
states, including any imported trajectory prefixes. A final presentation budget
alone does not establish continuation history.

Fresh provider calls or dataset downloads need not return the original vectors
or rows. Exact comparisons therefore use frozen embeddings, checkpoints, splits,
aligned row identities, and curated selections. Full empirical and synthetic
training require the compute budgets listed above.
