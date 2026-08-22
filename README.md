# fm-subtype-discovery

Type I error of covariance-preserving cluster nulls on learned representations.

Eva Bangsil, Nikhil Joshi — eva.bangsil@gmail.com

Paper: `PAPER_v5.md`. Full characterisation: `results/overnight/RESULTS.md`.
The repo name predates the result: no subtype structure was found, and the work is about the null.

## Summary

A covariance-preserving permutation null (Dinga et al., 2019) applied to L2-normalized
embeddings under a silhouette-driven *k*-search rejects structureless data at **20.5%** against
a nominal 5%. Two implementation errors cause it. **Geometry:** the Gaussian is fitted to
normalized embeddings but the draws are not re-projected onto the sphere, so they carry radial
variance the observed data cannot have. **Selection:** the observed statistic is a max over
*k* ∈ {2..6} while each null draw is scored at one fixed *k*.

Four constructions, selected by two booleans in `analysis/evaluate.py`:

| | Draws re-projected | Statistic per draw | *p*, real embedding |
|---|---|---|---|
| M | no | at observed *k*\* | 0.1685 |
| G | yes | at observed *k*\* | 0.4630 |
| S | no | max over *k* | 0.4033 |
| C | yes | max over *k* | 0.7009 |

Same data, same silhouette, same four clusters in every row. Both defaults in `evaluate.py`
are load-bearing; reverting either reintroduces one error.

[Live demo](https://evadiva3.github.io/fm-subtype-discovery/) — loads the real 28 × 119
embedding and runs the clustering, silhouette and every null draw client-side, nothing
precomputed. Source `docs/index.html`, one file, no build.

## Where the test fails

Structureless data by construction, 400 datasets × 200 draws per cell, d = 119. Rejection at
α = 0.05, M / C:

| target PR | n=28 | n=60 | n=119 | n=240 |
|---|---|---|---|---|
| 2.0 | .985 / .627 | 1.000 / .993 | 1.000 / 1.000 | 1.000 / 1.000 |
| 4.5 | .120 / .013 | .522 / .172 | .993 / .795 | 1.000 / .998 |
| 9.0 | .000 / .000 | .062 / .000 | .150 / .020 | .632 / .305 |
| 15.0 | .000 / .000 | .000 / .000 | .015 / .000 | .028 / .000 |

Full 30-cell grid in `results/overnight/e1_grid.csv`. C is at or below nominal in 9 of 30
cells. Its good behaviour at n = 28 is rank deficiency in the covariance estimate, not the
correction.

## The correction is not a guarantee

At **matched** measured participation ratio ≈ 4.5, n = 28, d = 119, varying only spectrum shape:

| spectrum | M | C | C-only | persist at 2000 draws |
|---|---|---|---|---|
| exponential decay | 0.130 | 0.015 | 2 / 1600 | 0 of 2 |
| power law | 0.200 | 0.205 | 89 / 1600 | 36 of 89 |
| one dominant eigendirection | 0.417 | **0.627** | — | — |
| two-block | 0.445 | **0.657** | 368 / 1600 | **244 of 368** |

On single-dominant-eigendirection spectra the corrected test rejects structureless data about
twice as often as the uncorrected one and the discordance reverses. Verified at four seeds and
re-tested at 2,000 draws. **A corrected-only rejection is not by itself a bug.** Earlier
releases said otherwise; that guidance is withdrawn.

So participation ratio is not sufficient to locate an analysis on the surface, and neither is
n/d — at fixed PR and fixed n/d, rejection rises steeply with d. The gap is specific to the
silhouette (gaps −0.005 / 0.000 / −0.033 for Calinski-Harabasz, Davies-Bouldin,
within-between, against +0.160 for silhouette) but survives every clustering algorithm tested.

## Setup

Python 3.10+, tested on 3.12–3.14.

```bash
git clone https://github.com/evadiva3/fm-subtype-discovery.git
cd fm-subtype-discovery
python3 -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
export OMP_NUM_THREADS=1
python -m pytest tests/ -q          # expect: 12 passed
```

- **`OMP_NUM_THREADS=1` is required, not general advice.** On a 28-point k-means, OpenMP
  coordination costs more than the arithmetic: 2.90 ms/fit single-threaded vs 74.07 ms at 32
  threads, byte-identical. Parallelize across processes.
- **`ray` has no wheel for 3.13+.** `src/train.py` imports it at module load, uses it only in
  the hyperparameter search. A stub whose `tune.report` raises is enough.
- **`umap-learn` is imported by `src/clustering.py`** though no null analysis uses it.
- **Depression data must be a repo sibling** (`../resting_state_dep_data/`), not inside it.
- `FileNotFoundError: Tuned hyperparameter 'dModel'` means `data/tune/bestParams.json` is
  missing. It is committed; pull again.

## Reproducing

**Tier 1 — clean clone, CPU.** `export OMP_NUM_THREADS=1` first.

| Command | Reproduces | Time |
|---|---|---|
| `analysis/null_progression.py 20000` | the four-rung ladder | 35 min |
| `MATCH_SELECTION=0 MATCH_GEOMETRY=0 NDATASETS=1000 NDRAWS=1000 analysis/calibration_run.py` | M Type I error, 0.205 | 5 min |
| `MATCH_SELECTION=1 MATCH_GEOMETRY=1 NDATASETS=1000 NDRAWS=1000 analysis/calibration_run.py` | C Type I error, 0.004 | 20 min |
| `analysis/type1_surface.py 200 200` | inflation tracks geometry | 3 min |
| `analysis/syntheticEmbeddings.py` then `analysis/separation_ratio.py` | detection floor, 0.584 / 1.166 / 1.725 | 30 min |
| `cd docs && python3 -m http.server 8000` | the demo locally | instant |

k-means init on the sphere is mildly scikit-learn-version sensitive, so the two sphere rungs
move a few draws in a thousand. No verdict moves.

**Tier 1b — the Type I error characterisation.** Drivers in
`experiments/ds002748_mdd/propagated_null/`, outputs to `results/overnight/`. Each checkpoints
per cell and skips completed cells on restart. ~2 h 15 m on 47 cores.

```
e1_grid.py            30-cell PR x n/d grid        e6_statistics.py     silhouette, CH, DB, within-between
e2_spectrum.py        5 spectra x 3 PR targets     e7_algorithms.py     k-means, Ward, GMM, spectral
e1_conly_check.py     verify C-only: 4 seeds       e8_normalization.py  none, L2, z-score, ZCA, PCA
e2_conly_check.py       + 2000-draw re-test        e9_seeds.py          5 replications
e3_radial_angular.py  6 nulls x 2 selection rules  e10_dimension.py     d in {32,64,128,256}
e4_k_family.py        Type I error vs effective K  e11_consensus.py     consensus, no training
e5_null_families.py   6 alternative nulls          make_figures.py, make_tables.py
```

**Tier 2 — needs the depression timeseries (5.6 MB).** 72 `*_rest_ts.npy` plus
`participants.tsv` in a sibling `resting_state_dep_data/`. Verify with
`python3 -c "import sys;sys.path.insert(0,'experiments/ds002748_mdd');from dataset_mdd import MDD_ROOT;print(MDD_ROOT,MDD_ROOT.exists())"`.
Needs GPU torch and `torch_geometric`. `subject_filter_mdd.py` needs fMRIPrep confounds that
are not shipped; all 72 participants passed the motion rule, so the `propagated_null/` scripts
override it and assert 72 = 51 + 21.

| Command | Reproduces | Time |
|---|---|---|
| `mdd_validate_forward.py` | forward-path validation, run first | 2 min |
| `mdd_propagated_null.py 0 500` | propagated null, *p* = 0.8124 | 3.5 h serial |
| `mdd_fixed_arch_control.py 0 20` | 20-seed fixed-architecture control | 1 h |

The null parallelizes across draws. Give each worker its own `MDD_NULL_ROOT` or they overwrite
each other's surrogate cohorts; run disjoint ranges and merge `draws/`. Ten workers on one GPU
did 500 draws in 2.5 h — the model is small enough that the GPU saturates on kernel launches.

**Tier 3 — needs preprocessed imaging we cannot ship.** Both OpenNeuro datasets, fMRIPrep, a
GPU, days. Layout in `docs/FM_Research_Data_Requirements.md`. Order: `preprocessing/` →
`src/hyperparameter_search.py` → `src/train.py` → `src/clustering.py` → the `analysis/`
scripts → `figures.py`. Depression: `experiments/ds002748_mdd/run_mdd_pipeline.sh`, then
`run_mdd_analysis.sh`.

## Results

| | Fibromyalgia (ds004144, 28 patients) | Depression (ds002748, 51 patients) |
|---|---|---|
| Architecture | 119 / 8 heads / 3 layers | 68 / 4 heads / 3 layers |
| Clustering | *k* = 4, sil 0.2433, *p* = 0.7009 | *k* = 5, sil 0.142, *p* = 0.3843 |
| Propagated null (raw data space) | *p* = 0.9062 | *p* = 0.8124 |
| Independent architectures | 0 of 15 (*p* 0.272–0.825) | — |
| Fixed-architecture control (seed only) | 0 of 20 (*p* 0.083–0.974) | 0 of 20 (*p* 0.086–0.927) |
| Cross-run ARI | 0.289 (14 runs) / 0.230 (20 runs) | 0.226 (20 runs) / 0.233 across seeds |
| Within-checkpoint bootstrap | 0.5676 | 0.370 |
| Untrained vs trained silhouette | 0.4600 vs 0.2433 | 0.387 vs 0.142 |
| Clinical / severity | 0 of 10 survive FDR | no gradient, all R² negative |

Cross-retrain agreement is 0.226–0.289 across both cohorts, two search spaces and four
protocols; the convention for a reproducible partition is 0.7. The within-checkpoint bootstrap
measures different variability and runs 1.6–2.0× higher.

Fibromyalgia baselines: PCA + k-means 0.057 (*p* = 0.983), flat upper-triangle 0.060 (0.886),
group ICA 0.377 (0.197), supervised SVM 0.503 accuracy (0.403, below majority class).

## What bounds the negative result

**Detection floor**, 1,440 planted-cluster realizations: power is 0 through offset 6, 13% at 8,
38% at 10, 80% at 12, 100% at 20. In transferable units, structureless data sits at separation
ratio 0.584, power starts at 1.166 and reaches 80% at 1.725. At offset 0 the control lands on
the real result exactly (0.2433, *p* = 0.6773).

**Information loss.** The connectivity features identify subjects at 93.1% (chance 1.7%) and
decode task condition at twice chance. Clustering the encoder's embeddings of the same graphs
recovers neither (ARI 0.061, 0.000); supervised decoding from those embeddings falls to 11.8%
and 18.3%. The loss is in the encoder, not the clustering.

So: no subtype structure this method could detect at this sample size. Not: no subtypes exist.

**Low-event rates.** 64 rates in `results/overnight/` rest on fewer than 10 rejection events,
including every corrected-arm figure at the fibromyalgia geometry. Five replications put the
corrected rate at 0.0067–0.0167, a 2.5× swing; pooled over 1,500 datasets it is
0.014 [0.009, 0.021]. Quote intervals, not point estimates.

## Layout

```
config.py, figures.py        paths/hyperparameters, publication figures
preprocessing/               fMRIPrep derivatives -> timeseries -> FC matrices
models/, src/                GATv2 encoder, NT-Xent; search, train, cluster
analysis/
  evaluate.py                ** the corrected permutation null **
  calibration_run.py         ** the 1,000 x 1,000 paired calibration **
  null_progression.py, type1_surface.py, raw_feature_ladder.py,
  separation_ratio.py, syntheticEmbeddings.py, baselines.py, + 10 more
experiments/ds002748_mdd/propagated_null/
  nullcal.py                 ** the four constructions as a library **
  geometry_surface.py, e1..e12 drivers, ov_common.py, make_figures.py
  mdd_propagated_null.py, mdd_validate_forward.py, mdd_fixed_arch_control.py
results/
  overnight/                 RESULTS.md, RUN_LOG.md, e*.json, CSVs, figures/
  figures/, mdd/             paper figures; canonical, multirun, propagated, fixed-arch
tests/, docs/index.html      12 pytest tests; browser demo
```

Most of `data/` and the fibromyalgia half of `results/` is gitignored — large and regenerable.
Committed so a fresh clone is not inert: `data/tune/bestParams.json` (without it `config`
cannot resolve), `data/outputs/` embeddings and labels (60 KB, enough for every tier-1
analysis), the OpenNeuro `*_events.tsv` / `*_confounds.tsv`, the Schaefer atlas, and
`results/mdd/**`, `results/figures/**`, `results/overnight/**` (each needs retraining or hours
of compute). The fibromyalgia checkpoint, the FC matrices and `data/clinical_clean.csv` are
not committed. **No subject-level clinical data is in this repository.**

Both datasets are public on OpenNeuro: **ds004144** (Balducci et al., 2022) and **ds002748**
(Bezmaternykh et al., 2021). The depression pipeline reads only parcellated timeseries;
re-parcellating would need the fMRIPrep derivatives, not the raw scans.

## If you extend this

- **The null is `analysis/evaluate.py`.** `_null_mvn` fits the Gaussian to the **L2-normalized**
  embeddings and re-projects each draw; `perm` defaults to `match_selection=True`.
- **`nullcal.py`** is the standalone library. `calibrate_paired` measures Type I error on your
  own embedding; `type1_surface_cell` measures any cell of the surface at your *n*, *d* and
  spectrum. Do this before interpreting a *p* — and vary the spectrum shape, not just the
  participation ratio.
- **Checkpoints carry `nodeMean`/`nodeStd`.** Inference must normalize with the training
  split's statistics; older checkpoints emit a fallback warning and are unusable for inference.
- **Batch size is fixed, not searched.** NT-Xent's denominator is the batch, so searching it
  ranks batch sizes, not architectures.
- **Open: NT-Xent temperature** pinned within 20% of its 0.05 floor in 12 of 15 searches. Range
  is now log-uniform [0.01, 1.0], effective next search. No conclusion depends on it.
- **Open: ZCA vs PCA whitening** in `e8_normalization.py` disagree where a rotation argument
  says they cannot. Suspected numerical conditioning in the ZCA inverse at n < d. Flagged, not
  used.

## AI usage

Generative AI was used as an editorial and analytical assistant: background research,
literature synthesis, prose and structural editing, code debugging and auditing, figures and
slides. All hypotheses, designs and conclusions are the authors', who reviewed and verified all
outputs and take responsibility for the content.

## References

Dinga R, et al. Evaluating the evidence for biotypes of depression. *NeuroImage Clin.* 2019;22:101796.
Kimes PK, et al. Statistical significance for hierarchical clustering. *Biometrics.* 2017;73(3):811-821.
Grabski IN, Street K, Irizarry RA. Significance analysis for clustering with single-cell RNA-sequencing data. *Nat Methods.* 2023;20:1196-1202.
Tozzi L, et al. Personalized brain circuit scores identify clinically distinct biotypes. *Nat Med.* 2024;30:2076-2087.
Balducci T, et al. A behavioral, clinical and brain imaging dataset ... females with fibromyalgia. *Sci Data.* 2022;9(1).
Bezmaternykh DD, et al. Resting state with closed eyes for patients with depression and healthy participants. OpenNeuro; 2021.
Brody S, Alon U, Yahav E. How attentive are graph attention networks? *ICLR* 2022.
Chen T, et al. A simple framework for contrastive learning of visual representations. *ICML* 2020.
