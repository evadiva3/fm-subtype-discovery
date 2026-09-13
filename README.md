# fm-subtype-discovery

Type I error of covariance-preserving cluster nulls on learned representations.

Eva Bangsil, Nikhil Joshi. Correspondence: eva.bangsil@gmail.com

Manuscript: `PAPER.md`. Supplement: `PAPER_SUPPLEMENT.md`. Section numbers below refer to
the manuscript.

The repository name predates the result. No subtype structure was recovered in either cohort,
and what the work reports is the behaviour of the null.

## Overview

A covariance-preserving permutation null of the kind introduced by Dinga et al. (2019), applied
to L2-normalized embeddings under a silhouette-driven *k*-search, rejects structureless data at
20.5 percent against a nominal 5 percent. Two procedure mismatches produce that inflation. The
first is geometric: the Gaussian is fitted to normalized embeddings but the draws are not
re-projected onto the sphere, so the null occupies a manifold the observed data cannot. The
second is selective: the observed statistic is a maximum over *k* in {2..6} while each null draw
is scored at the single count the observed data selected. Matching both brings the rate to 0.4
percent at the same geometry.

The realized Type I error reported here is measured on 62,789 datasets generated from a single
unimodal Gaussian, so every rejection is a false positive by construction. That hypothesis is
narrower than "the data contain no subgroups": it is the hypothesis the covariance-preserving
null implements, and a construction that fails to hold its size under its own generating
assumption cannot be expected to hold it under a weaker one.

## The four null constructions

Two binary choices define four tests, selected by two boolean arguments in
`analysis/evaluate.py`. All four share the same observed statistic, so any difference between
them is a difference in the reference distribution alone.

| | Draws re-projected | Statistic per draw | *p*, fibromyalgia embedding |
|---|---|---|---|
| M | no | at the observed *k*\* | 0.1685 |
| G | yes | at the observed *k*\* | 0.4630 |
| S | no | maximum over *k* | 0.4033 |
| C | yes | maximum over *k* | 0.7009 |

The silhouette, the partition and the selected *k* are identical across the four rows. Both
defaults in `evaluate.py` are load-bearing, and reverting either reintroduces one mismatch.

A browser implementation is at https://evadiva3.github.io/fm-subtype-discovery/. It loads the 28
by 119 embedding and computes the clustering, the silhouette and every null draw client-side.
Source is `docs/index.html`.

## Type I error over representation geometry

Data are drawn from a single unimodal Gaussian with exponentially decaying eigenvalues, tuned
per cell so that the measured sphere participation ratio meets its target. 400 datasets by 200
draws per cell at d = 119; rejection at α = 0.05, reported as M / C.

| target PR | n = 28 | n = 60 | n = 119 | n = 240 |
|---|---|---|---|---|
| 2.0 | .985 / .627 | 1.000 / .993 | 1.000 / 1.000 | 1.000 / 1.000 |
| 4.5 | .120 / .013 | .522 / .172 | .993 / .795 | 1.000 / .998 |
| 9.0 | .000 / .000 | .062 / .000 | .150 / .020 | .632 / .305 |
| 15.0 | .000 / .000 | .000 / .000 | .015 / .000 | .028 / .000 |

The region in which the matched construction is usable is a corner rather than a band. It sits
at or below nominal in nine of the thirty cells, and every one of those nine has a participation
ratio of at least 6 or a sample size of at most 60. Its low rejection rate at n = 28 follows
from rank deficiency in the covariance estimate, and is not evidence that matching works: at n =
200 it rejects structureless data at 0.508. A rate near zero is likewise not evidence of
calibration, since at the well-conditioned end the matched construction returns a mean *p* of
1.000 at a KS distance of 1.000 from uniform, which is degenerate conservatism.

The two mismatches interact rather than add. With both present rejection is 0.1525; matching the
geometry alone gives 0.0525, matching the selection alone gives 0.0500, and matching both gives
0.0175, so neither error inflates once the other has been removed (Section 4.10).

Full per-cell statistics for all thirty cells, including measured ratios, event counts, Wilson
intervals, mean *p*, KS distances, null means and both discordance counts, are in Section S9.1
of the supplement.

## Dependence on spectrum shape, and the reversal

At matched measured participation ratio near 4.5, n = 28 and d = 119, varying only the shape of
the eigenvalue spectrum. Matched-only rejections are pooled over four independent seeds;
"persist" counts those still discordant when the draw count is raised from 200 to 2,000.

| spectrum | M | C | C-only | persist |
|---|---|---|---|---|
| exponential decay | 0.130 | 0.015 | 2 / 1600 | 0 of 2 |
| power law | 0.200 | 0.205 | 89 / 1600 | 36 of 89 |
| one dominant eigendirection | 0.417 | 0.627 | 374 / 1600 | 241 of 374 |
| two-block | 0.445 | 0.657 | 368 / 1600 | 244 of 368 |

On concentrated spectra the matched construction rejects structureless data roughly twice as
often as the unmatched one, and the discordance reverses direction. A matched-only rejection is
therefore not by itself evidence of an implementation error. Earlier releases of this repository
stated otherwise, and that guidance is withdrawn.

The mechanism can be measured directly. Re-projecting a draw onto the sphere shifts the null
mean by +0.0084 under exponential decay and by -0.0085 and -0.0079 under the two concentrated
spectra, against a null standard deviation of 0.033 to 0.045, and the sign of that
quarter-standard-deviation shift predicts the reversal. Sweeping a one-parameter family that
interpolates between the spectra at fixed participation ratio, three separately computed
signatures change together: the shift crosses zero, the two rejection rates cross, and the
discordance direction reverses.

Mapping that crossing across 27 geometries shows it does not move with dimension, agreeing to
within 0.022 in leading-eigenvector variance fraction across a fourfold range in d, while it
does move with sample size and with conditioning. The transferable diagnostic is therefore the
sign of the shift, which costs two null distributions at a single *k*. A threshold on any scalar
summary of the spectrum is not.

| representation | PR (sphere) | PC1 (sphere) | position |
|---|---|---|---|
| fibromyalgia, trained | 4.37 | 0.326 | conservative side, margin ≈ 0.08 |
| depression, trained | 8.62 | 0.176 | conservative side, wide margin |
| fibromyalgia, untrained | 1.59 | 0.784 | past the sign change |

The fibromyalgia embedding lies on the conservative side, so *p* = 0.7009 is a conservative
reading and not an inflated one, which is the direction that supports the negative result.

## Further boundaries

Participation ratio does not locate an analysis on this surface, and neither does n/d. At fixed
participation ratio and fixed n/d, rejection rises steeply with d, so the controlling variable
is absolute n and the boundaries quoted above belong to d = 119 (Section 4.5).

The inflation belongs to the spherical constraint and not to dimensionality reduction in
general. At a participation ratio between 3.5 and 4.5, raw and z-scored data show no inflation
under any of the four constructions, while L2 normalization at essentially the same value shows
the full ladder (Section 4.7).

The gap between the two constructions is specific to the silhouette. At 1,200 datasets it is
+0.159 for the silhouette, +0.005 for Calinski-Harabasz and -0.005 for Davies-Bouldin, and it
inverts to -0.026 under a within-between dispersion ratio, where the matched construction is
significantly worse than the unmatched one on the exponential spectrum itself: pooled over four
seeds, 0.0581 [0.0477, 0.0707] against 0.0275 [0.0205, 0.0367], with 78 matched-only rejections
against 29 and 80.8 percent surviving at 2,000 draws. The gap survives every clustering
algorithm tested, including Ward, Gaussian mixtures and spectral clustering (Section 4.6).

Section 7.1 of the manuscript condenses these axes into a two-stage decision procedure. Stage 1
establishes which mismatches are present and therefore what the null must reproduce; stage 2
grades the risk that matching will nonetheless fail, ordered by cost, with the diagnostics
requiring no simulation placed before those requiring a calibration run.

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

Four properties of the environment are worth stating in advance.

`OMP_NUM_THREADS=1` is required. On a 28-point *k*-means, OpenMP coordination costs more than
the arithmetic: 2.90 ms per fit single-threaded against 74.07 ms at 32 threads, with
byte-identical output. Parallelism belongs across processes, not within them.

`ray` has no wheel for Python 3.13 or later. `src/train.py` imports it at module load but uses
it only in the hyperparameter search, so a stub whose `tune.report` raises is sufficient.
`src/clustering.py` imports `umap-learn`, though no null analysis uses it. The depression
timeseries must sit in a sibling directory (`../resting_state_dep_data/`), not inside the
repository.

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

*k*-means initialisation on the sphere is mildly sensitive to the scikit-learn version, so the
two sphere constructions can move by a few draws in a thousand. No verdict changes.

### Tier 1b: the Type I error characterisation

Drivers are in `experiments/ds002748_mdd/propagated_null/`. Each writes JSON to
`results/overnight/`, which is generated at run time and is not tracked, checkpoints after every
cell and skips completed cells on restart. Approximately 2 h 15 m on 47 cores.

```
e1_grid.py            PR x n grid, 30 cells        e10_dimension.py        d in {32,64,128,256}
e2_spectrum.py        5 spectra x 3 PR targets     e11_consensus.py        consensus partitions
e1_conly_check.py     matched-only verification:   e12_representations.py  stored embeddings
e2_conly_check.py       4 seeds + 2000 draws       e13_mechanism.py        sign of the shift
e3_radial_angular.py  6 nulls x 2 selection rules  e14_whitening.py        whitening diagnostic
e4_k_family.py        error vs effective |K|       e15_boundary.py         sign-change sweep
e5_null_families.py   6 alternative nulls          e16_nwb_check.py        E6 verification
e6_statistics.py      silhouette, CH, DB, w/b      e17_consensus_null.py   consensus vs a null
e7_algorithms.py      k-means, Ward, GMM, spectral e18_boundary_surface.py crossing, 27 geometries
e8_normalization.py   none, L2, z-score, ZCA, PCA  make_figures.py, make_figures2.py,
e9_seeds.py           5 replications               make_tables.py, make_supplement.py
```

Section S10 of the supplement maps each table and figure in the manuscript to the driver that
produces it and to that driver's output files.

### Tier 2: requires the depression timeseries (5.6 MB)

72 `*_rest_ts.npy` files plus `participants.tsv` in a sibling `resting_state_dep_data/`. Verify
with:

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

The forward-path validation is a precondition, not a formality: without it the null traverses a
different pipeline than the observed data and the comparison is void. The null parallelises
across draws, but each worker requires its own `MDD_NULL_ROOT` or workers overwrite one
another's surrogate cohorts. Run disjoint ranges and merge the `draws/` directories; ten workers
on one GPU completed 500 draws in 2.5 hours.

### Tier 3: requires preprocessed imaging not distributed here

Both OpenNeuro datasets, fMRIPrep and a GPU. File layout in
`docs/FM_Research_Data_Requirements.md`. Order: `preprocessing/`, then
`src/hyperparameter_search.py`, `src/train.py`, `src/clustering.py`, the `analysis/` scripts,
then `figures.py`. Depression: `experiments/ds002748_mdd/run_mdd_pipeline.sh`, then
`run_mdd_analysis.sh`.

## Results on the two cohorts

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

No clustering result reaches significance at any *k* for any encoder in either cohort. Testing
fifteen independently searched architectures raises rather than lowers the bar: the smallest of
the fifteen *p*-values is 0.272 against a Bonferroni threshold of 0.0033, and under the global
null the expected minimum of fifteen uniform draws would be 0.0625.

Cross-retrain agreement falls between 0.226 and 0.289 across both cohorts, two search spaces and
four protocols, against a conventional threshold of 0.7 for a partition offered as a discovered
subtype. The within-checkpoint bootstrap measures a different quantity and runs 1.6 to 2.0 times
higher. Aggregating across runs does recover a shared component: consensus partitions are stable
to resampling the runs at 0.6774 and 0.6079, and permuting each run's own labels, which
preserves its cluster count and sizes exactly, drops that to about 0.131 over 300 surrogates
each, outside the surrogate range entirely. That test establishes that the runs agree more than
chance, not that the consensus corresponds to subgroups in the data.

Fibromyalgia baselines: PCA with *k*-means, silhouette 0.057 (*p* = 0.983); flat upper-triangle
*k*-means, 0.060 (0.886); group ICA with *k*-means, 0.377 (0.197); supervised SVM, 0.503
accuracy (0.403), below the majority-class rate. Group ICA returns much the highest raw
silhouette of the four and is still far from significance, because a low-dimensional strongly
anisotropic representation is exactly the geometry that produces large silhouettes on
structureless data.

## Bounds on the negative result

Detection floor, from 1,440 planted-cluster realizations: power is zero through offset 6, 13
percent at 8, 38 percent at 10, 80 percent at 12 and 100 percent at 20. In transferable units,
structureless data sits at a separation ratio of 0.584, power becomes non-zero at 1.166 and
reaches 80 percent at 1.725. At offset 0 the control returns the observed result exactly
(0.2433, *p* = 0.6773), which confirms the harness reduces to the canonical analysis when
nothing is planted.

Information loss. The connectivity features entering the encoder identify subjects at 93.1
percent against a chance rate of 1.7 percent and decode task condition at twice chance.
Clustering the encoder's embeddings of the same graphs recovers neither grouping (ARI 0.061 and
0.000), and supervised decoding from those embeddings falls to 11.8 and 18.3 percent. The loss
is in the encoder, not in the clustering.

These data therefore establish that no subtype structure exists at a scale this method could
detect, which is a weaker clinical claim and a stronger methodological one than the absence of
subtype structure.

Sixty of the rates reported in the characterisation rest on fewer than ten rejection events,
including every matched-arm figure at the fibromyalgia geometry. Five independent replications
place the matched rate between 0.0067 and 0.0167, a factor of 2.5, and the audit of all sixty is
in Section S7 of the supplement. Every such rate should be quoted as an interval.

## Repository layout

```
config.py, figures.py        paths and hyperparameters; publication figures
preprocessing/               fMRIPrep derivatives -> timeseries -> FC matrices
models/, src/                GATv2 encoder, NT-Xent; search, train, cluster
analysis/                    23 scripts operating on a stored representation
  evaluate.py                  the matched permutation null
  calibration_run.py           the 1,000 x 1,000 paired calibration
  null_progression.py, type1_surface.py, raw_feature_ladder.py, e2e_calibration.py,
  separation_ratio.py, syntheticEmbeddings.py, run_baselines.py, ablation_table.py,
  clinical_validation.py, severity_gradient_regression.py, and others
  provenance_scripts/          14 drivers behind the fibromyalgia propagated null,
                               the feature-validity decoding and the audit gap fills
experiments/ds002748_mdd/propagated_null/
  nullcal.py                   the four constructions as a library
  e1-e18 (20 scripts)          the Type I error characterisation; ov_common.py is
                               the shared harness, geometry_surface.py the earlier sweeps
  make_figures.py, make_figures2.py, make_tables.py, make_supplement.py
  mdd_propagated_null.py, mdd_validate_forward.py, mdd_fixed_arch_control.py
packaging/                   nullcal as a standalone distribution, with its test suite
results/
  figures/                     manuscript figures
  mdd/                         canonical, multirun, propagated null, fixed-architecture
  overnight/                   Tier 1b output; generated at run time, not tracked
tests/                       12 pytest tests
docs/index.html              browser implementation
```

Most of `data/` and the fibromyalgia portion of `results/` is gitignored, being large and
regenerable, and `results/overnight/` is generated by the Tier 1b drivers and not tracked.

Committed so that a fresh clone is not inert: `data/tune/bestParams.json`, without which `config`
cannot resolve; the embeddings and labels in `data/outputs/` (418 KB, of which the condition-graph
embeddings are 378 KB, sufficient for every Tier 1 analysis); the Schaefer atlas;
`data/Clinical_fm_66.xlsx`, which is the variable dictionary for ds004144 and holds no
subject-level values; and `results/mdd/**` and `results/figures/**`. The depression artifacts are
committed rather than regenerated because each requires retraining.

Not committed: the fibromyalgia checkpoint, the FC matrices, `data/clinical_clean.csv`, and the
per-subject `*_events.tsv` and `*_Confounds.tsv` under `data/Subjects/`, which fall under the
`data/*` rule. The event files come from the OpenNeuro dataset and the confound files from
fMRIPrep, so both are regenerated by the Tier 3 route rather than distributed here. No
subject-level clinical data is held in this repository.

Both datasets are public on OpenNeuro: ds004144 (Balducci et al., 2022) and ds002748
(Bezmaternykh et al., 2021). The depression pipeline reads only parcellated timeseries.

## Implementation notes

The null is implemented in `analysis/evaluate.py`. `_null_mvn` fits the Gaussian to the
L2-normalized embeddings and re-projects each draw, and `perm` defaults to
`match_selection=True`.

`nullcal.py` is the standalone library version, packaged under `packaging/` with a 21-test
suite. `calibrate_paired` measures Type I error on a supplied embedding and `type1_surface_cell`
measures any cell of the surface at a supplied *n*, *d* and spectrum. Both should be run before
a *p*-value is interpreted, varying the spectrum shape and not only the participation ratio.

Checkpoints carry `nodeMean` and `nodeStd`. Inference must normalize with the training split's
statistics; older checkpoints emit a fallback warning and cannot be used for inference.

Batch size is fixed, not searched: the NT-Xent denominator is the batch, so searching over it
would rank batch sizes instead of architectures.

Two items remain open. The NT-Xent temperature pinned within 20 percent of its 0.05 floor in 12
of 15 searches, so those searches cannot be described as converged; the range is now log-uniform
on [0.01, 1.0], effective from the next search, and no conclusion depends on it. Separately, the
ZCA and PCA whitening arms in `e8_normalization.py` disagree where a rotation argument says they
cannot. `e14_whitening.py` establishes that the cause is degeneracy and not numerical
conditioning, since the two distance matrices agree to 5.3e-14 at rank 27 and a median condition
number of 4.0e3 while the observed silhouette is zero to machine precision. Neither arm is used
as a result, and Section S8 of the supplement reports the diagnostic in full.

## AI usage

Generative AI was used as an editorial and analytical assistant for background research,
literature synthesis, prose and structural editing, code debugging and auditing, and figure
preparation. All hypotheses, designs and conclusions are the authors', who reviewed and verified
all outputs and take responsibility for the content.

## References

Dinga R, et al. Evaluating the evidence for biotypes of depression. *NeuroImage Clin.*
2019;22:101796. Kimes PK, et al. Statistical significance for hierarchical clustering.
*Biometrics.* 2017;73(3):811-821. Chen Y, Witten D. Selective inference for k-means clustering.
*J Mach Learn Res.* 2023;24(152):1-41. Gao LL, Bien J, Witten D. Selective inference for
hierarchical clustering. *J Am Stat Assoc.* 2024;119(545):332-342. Grabski IN, Street K,
Irizarry RA. Significance analysis for clustering with single-cell RNA-sequencing data. *Nat
Methods.* 2023;20:1196-1202. Tozzi L, et al. Personalized brain circuit scores identify
clinically distinct biotypes. *Nat Med.* 2024;30:2076-2087. Balducci T, et al. A behavioral,
clinical and brain imaging dataset with focus on emotion regulation of females with
fibromyalgia. *Sci Data.* 2022;9(1). Bezmaternykh DD, et al. Resting state with closed eyes for
patients with depression and healthy participants. OpenNeuro; 2021. Brody S, Alon U, Yahav E.
How attentive are graph attention networks? *ICLR* 2022. Chen T, et al. A simple framework for
contrastive learning of visual representations. *ICML* 2020.
