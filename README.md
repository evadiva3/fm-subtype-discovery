# Significance Without Structure

Null misspecification and non-reproducible subtypes in small-sample self-supervised
neuroimaging.

Eva Bangsil, Nikhil Joshi. Correspondence: eva.bangsil@gmail.com

The directory is still called `fm-subtype-discovery`, which is a historical accident. It
started as a fibromyalgia subtyping project and turned into a methods paper about when
small-sample subtyping can be trusted at all. The fibromyalgia pipeline is the case study,
not the point.

## What happened

The standard way to validate an unsupervised representation in small-sample neuroimaging is
to cluster the embeddings, report a silhouette, and test it against a covariance-preserving
permutation null (Dinga et al., 2019). We built that pipeline, a GATv2 encoder trained with
NT-Xent, and got a four-cluster fibromyalgia partition at p = 0.024.

Then it failed to reproduce. Retraining on byte-identical inputs gave a different answer.

Chasing that took a while, and the problem turned out not to be the clustering. It was the
null. There were two independent errors in it, and either one on its own was enough to
manufacture a significant result. With both corrected the same analysis gives p = 0.68, and
across two disorders none of 40 runs reaches nominal significance. The reproducibility
checks, which never invoke a null at all, had been right the whole time.

## The two errors

A permutation null is only valid if every draw goes through the same procedure that produced
the observed number. Ours failed that twice.

**Geometry.** The observed embeddings are L2-normalized onto the unit sphere before
clustering. The null draws were sampled in ambient space and never projected, so they carry
radial spread the real data cannot have. That makes them harder to cluster, which pushes null
silhouettes down, which pushes p down.

**Selection.** The reported silhouette is a maximum over *k* ∈ {2…6}, picked after seeing the
data. Each null draw was scored at one fixed *k*, so the selection happened outside the
resampling loop. Given the same search, 33.8% of null draws maximize at *k* = 2 and only
25.3% at the *k* = 4 we picked.

There was a third, smaller problem: the estimator was `c/n` with a strict inequality, which
can return a p of exactly 0. It is `(c+1)/(n+1)` now.

On the checkpoint where we first traced this, the progression runs 0.024, then 0.177 with the
geometry fixed, then 0.93 with both. On the rebuilt pipeline the corrected value is 0.6773.


## Headline results

### Fibromyalgia (ds004144): 58 scanned, 28 patients

| Quantity | Value |
|---|---|
| Canonical architecture | dModel 119, heads 8, output 15, layers 3 |
| Clustering | *k* = 4, silhouette 0.2433, p = 0.6773 |
| Independent architectures tested | 0 of 7 significant (p 0.272 – 0.803) |
| 20-run search-varying protocol | 0 of 20 significant (min 0.131, median 0.576, max 0.937) |
| Cross-run agreement (adjusted Rand) | 0.230 |
| Within-checkpoint bootstrap | 0.568, twice as high, and measuring the wrong variability |
| Clinical variables surviving FDR | 0 of 10 (smallest corrected p = 0.896) |
| Untrained vs trained silhouette | 0.4599 vs 0.2433 (silhouette rewards collapse) |
| Participation ratio, untrained / trained | 1.585 / 3.703 |

### Depression (ds002748): 72 scanned, 51 patients

| Quantity | Value |
|---|---|
| Canonical architecture | dModel 68, heads 4, output 8, layers 3 (independently searched) |
| Clustering | *k* = 5, silhouette 0.142, p = 0.384 |
| 20-run protocol | 0 of 20 significant (median 0.590) |
| Cross-run agreement | 0.226 |
| Severity gradient | none on any of three scales; every out-of-sample R² is negative |

Cross-run agreement comes out at 0.230 in one cohort and 0.226 in the other. That is a gap
of 0.004 between different clinical populations, different acquisition modalities (task
versus resting-state), and separately searched architectures.

### Classical baselines (fibromyalgia)

| Method | Value | p |
|---|---|---|
| PCA + k-means | silhouette 0.057 | 0.983 |
| Flat upper-triangle k-means | silhouette 0.060 | 0.886 |
| Group ICA + k-means | silhouette 0.377 | 0.197 |
| Supervised SVM (patients vs controls) | 0.503 accuracy | 0.403, below the majority-class rate |

---

## What bounds these negative results

A null result is only meaningful if the test had power to reject. Both checks are in the
repository.

*Calibration* (`analysis/calibration_run.py`). 200 datasets drawn from the null the
corrected test itself assumes, structureless by construction. A calibrated test at
α = 0.05 rejects ~5% of the time. Ours rejects 0.5%, so the realized Type I error is roughly tenfold below nominal. At n = 28 the sample covariance is rank-deficient and overfit, so
null draws recapitulate more apparent structure than the data does.

*Detection floor* (`analysis/syntheticEmbeddings.py`). 1,440 planted-cluster
realizations at controlled separation:

| Planting offset | Mean ARI | Power |
|---|---|---|
| 0 – 6 | ≤ 0.15 | 0 % |
| 8 | 0.39 | 13 % |
| 10 | 0.62 | 38 % |
| 12 | 0.81 | 80 % |
| 20 | 0.98 | 100 % |

At offset 0 the control reproduces the real result exactly (silhouette 0.2433,
p = 0.6773) across all 120 cells, which is the integrity check on the planting machinery.

**The defensible claim is therefore "no subtype structure this method could detect at this
sample size," not "no subtypes exist."** These analyses exclude structure that would be
unmistakable and say little about subtler structure.

---

## Repository layout

```
config.py                    Single source of truth for hyperparameters and paths
figures.py                   Publication figures, entry point
requirements.txt

preprocessing/               fMRIPrep derivatives → timeseries → FC matrices
  compute_fc_matrices.py       Ledoit-Wolf shrinkage, Fisher z
  subject_filter.py            Motion and completeness exclusions
  verify_setup.py

dataFiltering/               Clinical spreadsheet → clean CSV

models/                      Architecture
  gnn_encoder.py               3 × GATv2, multi-head graph attention
  attention_pool.py            Temperature-scaled condition attention
  contrastive_loss.py          NT-Xent
  augmentations.py             Node masking, edge noise
  dataset.py                   Graph construction, normalization

src/                         Training and clustering
  hyperparameter_search.py     Optuna + ASHA; optimizes NT-Xent loss, never silhouette
  train.py                     Joint encoder/pool training; persists normStats
  clustering.py                k-means, size guard, permutation test

analysis/                    Evaluation
  evaluate.py                  ** the corrected permutation null lives here **
  driver_utils.py              Shared clustering/eval helpers
  baselines.py, run_baselines.py
  bootstrap_stability.py       Within-checkpoint resampling
  syntheticEmbeddings.py       Planted-cluster positive control
  clinical_validation.py       Kruskal-Wallis + FDR, motion check
  severity_gradient_regression.py, knn_probe.py, confidence_intervals.py
  effective_rank.py            Participation ratio
  sensitivity_analysis.py      Edge-threshold sweep
  orchestrator.py
  calibration_run.py           ** the 200-dataset calibration experiment **

experiments/ds002748_mdd/    Depression replication (parallel pipeline)
tests/                       pytest suite
docs/                        Data requirements, repository architecture notes
notebooks/                   Exploratory only; no result depends on these
```

Everything under `data/` and `results/` is gitignored and regenerable.

**Presentation material lives outside this repository.** Conference videos, slide
figures, and the code that generates them were moved to `../presentation/`. They are
communication artifacts, not part of any result, and they were 400 MB of the repository.
Publication figures (`figures.py` → `results/figures/fig1…fig3`) remain here.

---

## Try it in your browser

[Live demo](https://evadiva3.github.io/fm-subtype-discovery/). No install, no data, nothing to
download.

[![The four null constructions running in the browser. The null distribution shifts right past the observed silhouette as each correction is applied, and the ladder table fills in.](docs/demo/thumbnail_1600.png)](https://evadiva3.github.io/fm-subtype-discovery/)

The page runs the whole argument client-side on the actual 28 × 119 embedding matrix from the
paper. Turn the two corrections on and off and watch the same partition stop being
significant. Nothing is precomputed. The k-means, the silhouette and every null draw are
computed in the page while you wait.

Hit "Run the full ladder" and it rebuilds the paper's central table live:

| Construction | Geometry matched | Selection matched | Paper *p* |
|---|---|---|---|
| Misspecified, as originally implemented | ✗ | ✗ | 0.1698 |
| Geometry corrected only | ✓ | ✗ | 0.4436 |
| Selection corrected only | ✗ | ✓ | 0.3966 |
| Fully corrected | ✓ | ✓ | 0.6773 |

It uses the same 20 k-means restarts as the paper, so it lands within Monte-Carlo noise of
those numbers. A 1,000-draw run comes back around 0.164, 0.435, 0.410 and 0.694. The null mean
silhouettes track just as closely: 0.215, 0.240 and 0.258 against 0.216, 0.242 and 0.259. Its
silhouette matches the Python to seven decimals (0.2433061).

The reason this works with no brain data is that the finding is about the null construction,
not about fibromyalgia. It needs one embedding matrix. No MRI, no checkpoint, no training.
Source is `docs/index.html`, a single file with no dependencies.

---

## Reproducing the full pipeline

The demo above needs nothing. Reproducing the *whole* study needs the imaging data, which is
public but large; see Data availability below. `data/` and `results/` are gitignored, so
a fresh clone has the code but not the derivatives.

```bash
pip install -r requirements.txt
export OMP_NUM_THREADS=1        # not optional, see below
```

```bash
# 1. preprocessing (assumes fMRIPrep derivatives on disk)
python preprocessing/verify_setup.py
python preprocessing/compute_fc_matrices.py

# 2. architecture search      → data/tune/bestParams.json
python src/hyperparameter_search.py

# 3. train                    → data/checkpoints/results/bestJointModel.pt
python src/train.py

# 4. cluster + significance   → data/outputs/
python src/clustering.py

# 5. analyses
python analysis/ablation_table.py
python analysis/bootstrap_stability.py
python analysis/run_baselines.py
python analysis/clinical_validation.py
python analysis/effective_rank.py
python analysis/severity_gradient_regression.py

# 6. positive control and calibration
python analysis/syntheticEmbeddings.py
python analysis/calibration_run.py

# 7. figures
python figures.py
```

Depression replication is in `experiments/ds002748_mdd/`, same order:
`hyperparameter_search_mdd.py` → `train_mdd.py` → `clustering_mdd.py`.

> **`OMP_NUM_THREADS=1` is not general advice.** scikit-learn's k-means parallelizes with
> OpenMP, and on a 28-point problem thread coordination costs more than the arithmetic:
> 2.90 ms/fit single-threaded against 74.07 ms at 32 threads. ~25× faster, byte-identical
> output.

---

## Notes for anyone extending this

**The null is in `analysis/evaluate.py`.** `_null_mvn` fits the Gaussian to the
**L2-normalized** embeddings and re-projects each draw onto the sphere; `perm` runs with
`match_selection=True` by default, so every draw repeats the full *k*-search under the
same minimum-cluster-size guard. Both behaviours are load-bearing, and reverting either
reintroduces one of the two errors.

**Checkpoints carry `nodeMean`/`nodeStd`.** Inference must normalize with the training
split's statistics, not all-subject statistics. Checkpoints predating this fix emit a
fallback warning and are not usable for inference.

**Batch size is fixed, not searched.** NT-Xent's denominator is the batch, so a trial with
a larger batch solves a harder discrimination problem and records a worse validation loss
for reasons unrelated to architecture. Searching it ranks batch sizes, not architectures.

**Open issue: NT-Xent temperature.** The selected temperature pinned at the search-range
floor in 7 of 7 independent searches, meaning the bound was binding and the optimum lies
below it. The range is now log-uniform [0.01, 1.0], but **this takes effect only on the
next search**. Every architecture above was selected under the old bound. Related
symptom: `tests/test_contrastive_loss.py::test_orthonormal_aligned_analytic` fails its
`q > 0` assertion because at the canonical temperature (0.0505) the analytic loss for
perfectly-aligned views is 1.5 × 10⁻⁸ and underflows to zero. That is a brittle test
meeting a very small temperature, not a defect in the loss. 11 of 12 tests pass.

---

## Data availability

Both datasets are public on OpenNeuro: **ds004144** (fibromyalgia task-fMRI, Balducci et
al., 2022) and **ds002748** (depression resting-state, Bezmaternykh et al., 2021).
Derivatives, checkpoints, and per-run outputs are gitignored; regenerate with the
commands above.

## AI usage

Generative AI tools were used as editorial and analytical assistants: background research,
literature synthesis, prose refinement, code debugging and auditing, and figure/slide
production. All research hypotheses, experimental designs, and scientific conclusions are
the authors'. All AI-generated output was reviewed and verified, and the authors take full
responsibility for the integrity of this work.

## Key references

Dinga R, et al. Evaluating the evidence for biotypes of depression. *NeuroImage Clin.* 2019;22:101796.
Tozzi L, et al. Personalized brain circuit scores identify clinically distinct biotypes in depression and anxiety. *Nat Med.* 2024;30:2076-2087.
Grabski IN, Street K, Irizarry RA. Significance analysis for clustering with single-cell RNA-sequencing data. *Nat Methods.* 2023;20:1196-1202.
Balducci T, et al. A behavioral, clinical and brain imaging dataset with focus on emotion regulation of females with fibromyalgia. *Sci Data.* 2022;9(1).
Bezmaternykh DD, et al. Resting state with closed eyes for patients with depression and healthy participants. OpenNeuro; 2021.
Schaefer A, et al. Local-global parcellation of the human cerebral cortex. *Cereb Cortex.* 2018;28(9).
Brody S, Alon U, Yahav E. How attentive are graph attention networks? *ICLR* 2022.
Chen T, et al. A simple framework for contrastive learning of visual representations. *ICML* 2020.
