# fm-subtype-discovery

Type I error of covariance-preserving cluster nulls on learned representations.

Eva Bangsil, Nikhil Joshi. Correspondence: eva.bangsil@gmail.com

The repository name predates the result. No subtype structure was recovered in either cohort;
the work reports the properties of the null.

## Overview

A covariance-preserving permutation null (Dinga et al., 2019) applied to L2-normalized
embeddings under a silhouette-driven *k*-search rejects structureless data at 20.5 percent
against a nominal 5 percent. Two implementation errors produce this. The first is geometric:
the Gaussian is fitted to normalized embeddings but the draws are not re-projected onto the
sphere, so they carry radial variance the observed data cannot have. The second is selective:
the observed statistic is a maximum over *k* ∈ {2..6} while each null draw is scored at one
fixed *k*. Correcting both gives 0.4 percent at the same geometry.

## Null constructions

Two binary choices define four tests, selected by two boolean arguments in
`analysis/evaluate.py`. All four use the same observed statistic; only the reference
distribution differs.

| | Draws re-projected | Statistic per draw | *p*, fibromyalgia embedding |
|---|---|---|---|
| M | no | at observed *k*\* | 0.1685 |
| G | yes | at observed *k*\* | 0.4630 |
| S | no | max over *k* | 0.4033 |
| C | yes | max over *k* | 0.7009 |

The silhouette, the partition and the selected *k* are identical across the four rows. Both
defaults in `evaluate.py` are load-bearing; reverting either reintroduces one error.

A browser implementation is at https://evadiva3.github.io/fm-subtype-discovery/. It loads the
28 × 119 embedding and computes the clustering, the silhouette and every null draw
client-side. Source is `docs/index.html`.

## Type I error over representation geometry

Structureless data by construction: a single unimodal Gaussian with exponentially decaying
eigenvalues, so every rejection is a false positive. 400 datasets × 200 draws per cell,
d = 119. Rejection at α = 0.05, M / C:

| target PR | n = 28 | n = 60 | n = 119 | n = 240 |
|---|---|---|---|---|
| 2.0 | .985 / .627 | 1.000 / .993 | 1.000 / 1.000 | 1.000 / 1.000 |
| 4.5 | .120 / .013 | .522 / .172 | .993 / .795 | 1.000 / .998 |
| 9.0 | .000 / .000 | .062 / .000 | .150 / .020 | .632 / .305 |
| 15.0 | .000 / .000 | .000 / .000 | .015 / .000 | .028 / .000 |

C is at or below nominal in 9 of the 30 cells. Its low rejection rate at n = 28 is a
consequence of rank deficiency in the covariance estimate rather than of the correction: at
n = 200 it rejects structureless data at 0.508.

## Dependence on spectrum shape

At matched measured participation ratio (≈ 4.5), n = 28, d = 119, varying only the shape of
the eigenvalue spectrum. Corrected-only rejections are pooled over four seeds; "persist" counts
those still discordant when the draw count is raised from 200 to 2,000.

| spectrum | M | C | C-only | persist |
|---|---|---|---|---|
| exponential decay | 0.130 | 0.015 | 2 / 1600 | 0 of 2 |
| power law | 0.200 | 0.205 | 89 / 1600 | 36 of 89 |
| one dominant eigendirection | 0.417 | 0.627 | 374 / 1600 | 241 of 374 |
| two-block | 0.445 | 0.657 | 368 / 1600 | 244 of 368 |

On concentrated spectra the corrected construction rejects structureless data roughly twice as
often as the uncorrected one, and the discordance reverses direction. A corrected-only
rejection is therefore not by itself evidence of an implementation error. Earlier releases of
this repository stated otherwise; that guidance is withdrawn.

Re-projecting a draw onto the sphere shifts the null mean by +0.0084 under exponential decay
and by -0.0085 and -0.0079 under the concentrated spectra, against a null standard deviation of
0.033 to 0.045. The sign of that shift predicts the reversal. Sweeping a family that
interpolates between the spectra at fixed participation ratio, the shift crosses zero, the two
rejection rates cross, and the discordance reverses at a leading-eigenvector variance fraction
of approximately 0.40.

| representation | PR (sphere) | PC1 (sphere) | position |
|---|---|---|---|
| fibromyalgia, trained | 4.37 | 0.326 | below the crossing, margin ≈ 0.08 |
| depression, trained | 8.62 | 0.176 | below the crossing |
| fibromyalgia, untrained | 1.59 | 0.784 | above the crossing |

The fibromyalgia embedding lies below the crossing, so *p* = 0.7009 is a conservative reading.
PC1 on the sphere is worth reporting alongside the participation ratio, though the crossing was
measured at one geometry and should be re-measured rather than assumed to transfer.

Participation ratio alone does not locate an analysis on this surface, and neither does n/d: at
fixed participation ratio and fixed n/d, rejection rises with d. The M-C gap is specific to the
silhouette (+0.005 for Calinski-Harabasz and -0.005 for Davies-Bouldin against +0.159 for the
silhouette, at 1,200 datasets) but survives every clustering algorithm tested, and under a
within/between dispersion ratio the corrected construction is significantly worse than the
uncorrected one on the exponential spectrum itself. Detail in `PAPER_v5.md` §5.

## Installation

Python 3.10 or later; tested on 3.12 to 3.14.

```bash
git clone https://github.com/evadiva3/fm-subtype-discovery.git
cd fm-subtype-discovery
python3 -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
export OMP_NUM_THREADS=1
python -m pytest tests/ -q          # expect: 12 passed
```

Four notes on the environment:

- `OMP_NUM_THREADS=1` is required. On a 28-point k-means, OpenMP coordination costs more than
  the arithmetic: 2.90 ms per fit single-threaded against 74.07 ms at 32 threads, with
  byte-identical output. Parallelism should be across processes.
- `ray` has no wheel for Python 3.13 or later. `src/train.py` imports it at module load but
  uses it only in the hyperparameter search; a stub whose `tune.report` raises is sufficient.
- `src/clustering.py` imports `umap-learn`, though no null analysis uses it.
- The depression timeseries must sit in a sibling directory (`../resting_state_dep_data/`),
  not inside the repository.

`FileNotFoundError: Tuned hyperparameter 'dModel'` indicates a missing
`data/tune/bestParams.json`, which is committed.

## Reproduction

### Tier 1: clean clone, CPU only

Set `OMP_NUM_THREADS=1` first.

| Command | Reproduces | Time |
|---|---|---|
| `analysis/null_progression.py 20000` | the four constructions | 35 min |
| `MATCH_SELECTION=0 MATCH_GEOMETRY=0 NDATASETS=1000 NDRAWS=1000 analysis/calibration_run.py` | M Type I error, 0.205 | 5 min |
| `MATCH_SELECTION=1 MATCH_GEOMETRY=1 NDATASETS=1000 NDRAWS=1000 analysis/calibration_run.py` | C Type I error, 0.004 | 20 min |
| `analysis/type1_surface.py 200 200` | Type I error by geometry | 3 min |
| `analysis/syntheticEmbeddings.py`, then `analysis/separation_ratio.py` | detection floor, 0.584 / 1.166 / 1.725 | 30 min |
| `cd docs && python3 -m http.server 8000` | the browser implementation locally | instant |

k-means initialisation on the sphere is mildly sensitive to the scikit-learn version, so the
two sphere constructions can move by a few draws in a thousand. No verdict changes.

### Tier 1b: the Type I error characterisation

Drivers are in `experiments/ds002748_mdd/propagated_null/`. Each writes JSON to
`results/overnight/`, which is generated at run time and is not tracked. Each checkpoints per
cell and skips completed cells on restart. Approximately 2 h 15 m on 47 cores.

```
e1_grid.py            PR x n/d grid, 30 cells      e9_seeds.py             5 replications
e2_spectrum.py        5 spectra x 3 PR targets     e10_dimension.py        d in {32,64,128,256}
e1_conly_check.py     C-only verification:         e11_consensus.py        consensus, no training
e2_conly_check.py       4 seeds + 2000 draws       e12_representations.py  stored embeddings
e3_radial_angular.py  6 nulls x 2 selection rules  e13_mechanism.py        sign of the shift
e4_k_family.py        error vs effective |K|       e14_whitening.py        E8 diagnostic
e5_null_families.py   6 alternative nulls          e15_boundary.py         sign-change sweep
e6_statistics.py      silhouette, CH, DB, w/b      e16_nwb_check.py        E6 verification
e7_algorithms.py      k-means, Ward, GMM, spectral e17_consensus_null.py   consensus vs a null
e8_normalization.py   none, L2, z-score, ZCA, PCA  make_figures.py, make_figures2.py,
                                                   make_tables.py, make_supplement.py
```

### Tier 2: requires the depression timeseries (5.6 MB)

72 `*_rest_ts.npy` files plus `participants.tsv` in a sibling `resting_state_dep_data/`.
Verify with:

```bash
python3 -c "import sys;sys.path.insert(0,'experiments/ds002748_mdd');from dataset_mdd import MDD_ROOT;print(MDD_ROOT,MDD_ROOT.exists())"
```

Requires GPU torch and `torch_geometric`. `subject_filter_mdd.py` expects fMRIPrep confounds
that are not distributed; all 72 participants pass the motion criterion, so the
`propagated_null/` scripts override the filter and assert 72 = 51 + 21.

| Command | Reproduces | Time |
|---|---|---|
| `mdd_validate_forward.py` | forward-path validation; run first | 2 min |
| `mdd_propagated_null.py 0 500` | propagated null, *p* = 0.8124 | 3.5 h serial |
| `mdd_fixed_arch_control.py 0 20` | 20-seed fixed-architecture control | 1 h |

The null parallelises across draws. Each worker requires its own `MDD_NULL_ROOT`, or workers
overwrite one another's surrogate cohorts; run disjoint ranges and merge the `draws/`
directories. Ten workers on one GPU completed 500 draws in 2.5 hours.

### Tier 3: requires preprocessed imaging not distributed here

Both OpenNeuro datasets, fMRIPrep and a GPU. File layout in
`docs/FM_Research_Data_Requirements.md`. Order: `preprocessing/`, then
`src/hyperparameter_search.py`, `src/train.py`, `src/clustering.py`, the `analysis/` scripts,
then `figures.py`. Depression: `experiments/ds002748_mdd/run_mdd_pipeline.sh`, then
`run_mdd_analysis.sh`.

## Results

| | Fibromyalgia (ds004144, 28 patients) | Depression (ds002748, 51 patients) |
|---|---|---|
| Architecture | 119 / 8 heads / 3 layers | 68 / 4 heads / 3 layers |
| Clustering | *k* = 4, silhouette 0.2433, *p* = 0.7009 | *k* = 5, silhouette 0.142, *p* = 0.3843 |
| Propagated null | *p* = 0.9062 | *p* = 0.8124 |
| Independent architectures | 0 of 15 (*p* 0.272-0.825) | n/a |
| Fixed-architecture control | 0 of 20 (*p* 0.083-0.974) | 0 of 20 (*p* 0.086-0.927) |
| Cross-run ARI | 0.289 (14 runs), 0.230 (20 runs) | 0.226 (20 runs), 0.233 across seeds |
| Within-checkpoint bootstrap | 0.5676 | 0.370 |
| Untrained / trained silhouette | 0.4600 / 0.2433 | 0.387 / 0.142 |
| Clinical, severity | 0 of 10 survive FDR | no gradient; all out-of-sample R² negative |

Cross-retrain agreement falls between 0.226 and 0.289 across both cohorts, two search spaces
and four protocols, against a conventional threshold of 0.7 for a reproducible partition. The
within-checkpoint bootstrap measures a different quantity and runs 1.6 to 2.0 times higher.

Fibromyalgia baselines: PCA with k-means, silhouette 0.057 (*p* = 0.983); flat upper-triangle
k-means, 0.060 (0.886); group ICA with k-means, 0.377 (0.197); supervised SVM, 0.503 accuracy
(0.403), below the majority-class rate.

## Bounds on the negative result

Detection floor, from 1,440 planted-cluster realizations: power is zero through offset 6, 13
percent at 8, 38 percent at 10, 80 percent at 12 and 100 percent at 20. In transferable units,
structureless data sits at separation ratio 0.584, power becomes non-zero at 1.166 and reaches
80 percent at 1.725. At offset 0 the control returns the observed result exactly (0.2433,
*p* = 0.6773).

Information loss: the connectivity features identify subjects at 93.1 percent against a chance
rate of 1.7 percent, and decode task condition at twice chance. Clustering the encoder's
embeddings of the same graphs recovers neither (ARI 0.061 and 0.000), and supervised decoding
from those embeddings falls to 11.8 and 18.3 percent. The loss is in the encoder rather than
the clustering.

These data therefore establish that no subtype structure exists at a scale this method could
detect, not that no subtype structure exists.

Sixty of the rates reported in the characterisation rest on fewer than ten rejection events,
including every corrected-arm figure at the fibromyalgia geometry. Five independent
replications place the corrected rate between 0.0067 and 0.0167, a factor of 2.5; pooled over
1,500 datasets it is 0.014 with 95 percent interval [0.009, 0.021]. Intervals rather than point
estimates should be quoted.

## Repository layout

```
config.py, figures.py        paths and hyperparameters; publication figures
preprocessing/               fMRIPrep derivatives -> timeseries -> FC matrices
models/, src/                GATv2 encoder, NT-Xent; search, train, cluster
analysis/
  evaluate.py                  the corrected permutation null
  calibration_run.py           the 1,000 x 1,000 paired calibration
  null_progression.py, type1_surface.py, raw_feature_ladder.py,
  separation_ratio.py, syntheticEmbeddings.py, baselines.py, and 10 more
experiments/ds002748_mdd/propagated_null/
  nullcal.py                   the four constructions as a library
  geometry_surface.py, e1-e17 drivers, ov_common.py, make_figures.py
  mdd_propagated_null.py, mdd_validate_forward.py, mdd_fixed_arch_control.py
results/
  figures/                     manuscript figures
  mdd/                         canonical, multirun, propagated null, fixed-architecture
tests/                       12 pytest tests
docs/index.html              browser implementation
```

Most of `data/` and the fibromyalgia portion of `results/` is gitignored, being large and
regenerable, and `results/overnight/` is generated by the Tier 1b drivers rather than tracked.
Committed so that a fresh clone is not inert: `data/tune/bestParams.json`, without which
`config` cannot resolve; the embeddings and labels in `data/outputs/` (60 KB, sufficient for
every Tier 1 analysis); the OpenNeuro `*_events.tsv` and `*_confounds.tsv` files; the Schaefer
atlas; and `results/mdd/**` and `results/figures/**`. The fibromyalgia checkpoint, the FC
matrices and `data/clinical_clean.csv` are not committed. No subject-level clinical data is
held in this repository.

Both datasets are public on OpenNeuro: ds004144 (Balducci et al., 2022) and ds002748
(Bezmaternykh et al., 2021). The depression pipeline reads only parcellated timeseries.

## Implementation notes

The null is implemented in `analysis/evaluate.py`. `_null_mvn` fits the Gaussian to the
L2-normalized embeddings and re-projects each draw; `perm` defaults to `match_selection=True`.

`nullcal.py` is the standalone library version. `calibrate_paired` measures Type I error on a
supplied embedding, and `type1_surface_cell` measures any cell of the surface at a supplied
*n*, *d* and spectrum. Both should be run before a *p*-value is interpreted, varying the
spectrum shape and not only the participation ratio.

Checkpoints carry `nodeMean` and `nodeStd`. Inference must normalize with the training split's
statistics; older checkpoints emit a fallback warning and cannot be used for inference.

Batch size is fixed rather than searched. The NT-Xent denominator is the batch, so searching
over it ranks batch sizes rather than architectures.

Two open items. The NT-Xent temperature pinned within 20 percent of its 0.05 floor in 12 of 15
searches; the range is now log-uniform on [0.01, 1.0], effective from the next search, and no
conclusion depends on it. Separately, the ZCA and PCA whitening arms in `e8_normalization.py`
disagree where a rotation argument says they cannot; `e14_whitening.py` establishes that the
cell is degenerate rather than numerically unstable, and neither arm is used as a result.

## AI usage

Generative AI was used as an editorial and analytical assistant for background research,
literature synthesis, prose and structural editing, code debugging and auditing, and figure
preparation. All hypotheses, designs and conclusions are the authors', who reviewed and
verified all outputs and take responsibility for the content.

## References

Dinga R, et al. Evaluating the evidence for biotypes of depression. *NeuroImage Clin.* 2019;22:101796.
Kimes PK, et al. Statistical significance for hierarchical clustering. *Biometrics.* 2017;73(3):811-821.
Grabski IN, Street K, Irizarry RA. Significance analysis for clustering with single-cell RNA-sequencing data. *Nat Methods.* 2023;20:1196-1202.
Tozzi L, et al. Personalized brain circuit scores identify clinically distinct biotypes. *Nat Med.* 2024;30:2076-2087.
Balducci T, et al. A behavioral, clinical and brain imaging dataset with focus on emotion regulation of females with fibromyalgia. *Sci Data.* 2022;9(1).
Bezmaternykh DD, et al. Resting state with closed eyes for patients with depression and healthy participants. OpenNeuro; 2021.
Brody S, Alon U, Yahav E. How attentive are graph attention networks? *ICLR* 2022.
Chen T, et al. A simple framework for contrastive learning of visual representations. *ICML* 2020.
