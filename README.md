# Significance Without Structure

Null misspecification and non-reproducible subtypes in small-sample self-supervised
neuroimaging.

Eva Bangsil, Nikhil Joshi. Correspondence: eva.bangsil@gmail.com

## AI usage

Generative AI tools were utilized throughout this project as editorial and analytical
assistants. AI supported contextual background research, literature synthesis, and conceptual
clarification. It was also used to refine prose, structure the manuscript, assist with code
debugging and auditing, and to help create figures, videos, and presentation slides. Despite
this assistance, all core research hypotheses, experimental designs, and scientific
conclusions remain exclusively our own. We have reviewed and verified all AI-generated
outputs and take full responsibility for the integrity and content of this work.

## What this is

We tried to find subtypes of fibromyalgia patients from brain connectivity data. We found
four, they looked clean, and they weren't real. What made them look real was a bug in the
statistical test, not anything in the scans or the clustering.

So this stopped being a fibromyalgia paper and became a methods one, about how easy it is to
get a convincing false positive out of a very standard analysis. The repo is still called
`fm-subtype-discovery` from before we knew that.

You can run the whole argument in your browser without installing anything. Setup for the
rest is below.

## Try it in your browser

[Live demo](https://evadiva3.github.io/fm-subtype-discovery/). Nothing to install.

[![The four null constructions running in the browser. The null distribution shifts right past the observed silhouette as each correction is applied, and the ladder table fills in.](docs/demo/thumbnail_1600.png)](https://evadiva3.github.io/fm-subtype-discovery/)

The page loads the real 28 x 119 embedding matrix from the paper and does the clustering, the
silhouette and every null draw in front of you. Nothing is precomputed. Press "Run the full
ladder" and in about twenty seconds you get this:

| Construction | Geometry matched | Selection matched | Paper *p* |
|---|---|---|---|
| Misspecified, as originally implemented | No | No | 0.1698 |
| Geometry corrected only | Yes | No | 0.4436 |
| Selection corrected only | No | Yes | 0.3966 |
| Fully corrected | Yes | Yes | 0.6773 |

Same data, same silhouette, same four clusters in all four rows. The only thing that changes
is what we compare against, and the result goes from significant to not.

It uses the same 20 k-means restarts the paper does, so you should land within Monte-Carlo
noise of those numbers. A 1,000-draw run comes back near 0.164, 0.435, 0.410 and 0.694. The
silhouette agrees with the Python to seven decimals (0.2433061).

Source is `docs/index.html`. One file, no dependencies, no build step.

## Setup

Python 3.10 or newer. Tested on 3.12 and 3.14, macOS and Linux. Nothing on this page needs a
GPU.

```bash
git clone https://github.com/evadiva3/fm-subtype-discovery.git
cd fm-subtype-discovery

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

That pulls PyTorch, PyTorch Geometric, scikit-learn, nilearn and Ray Tune. Takes a few
minutes and about 2 GB, nearly all of it PyTorch.

Check it worked:

```bash
export OMP_NUM_THREADS=1
python -m pytest tests/ -q
```

You want `12 passed`. If you instead get `FileNotFoundError: Tuned hyperparameter 'dModel' is
unavailable`, your clone is missing `data/tune/bestParams.json`. It is committed, so pull
again.

## Run it locally

Three things work right after setup. No imaging data, no downloads. The clone ships the
trained embeddings and the architecture spec, which is all these need.

**Serve the demo yourself** instead of using the hosted one:

```bash
cd docs && python3 -m http.server 8000
```

Then open `http://localhost:8000`.

**Rerun the ladder in Python.** Same four constructions as the demo, about two minutes:

```bash
export OMP_NUM_THREADS=1
python analysis/null_progression.py 1000
```

Expect roughly 0.170, 0.444, 0.397 and 0.677, and a file in `results/null_corrected/`.

**Measure how often each construction fires on data with no groups in it.** About three
minutes:

```bash
python analysis/type1_surface.py 200 200
```

Writes `results/type1_surface.json`. The misspecified construction should reject far more than
the nominal 5 percent. The corrected one should reject far less.

## What happened

The usual way to claim you have found subtypes is: learn an embedding per patient, cluster
them, score how separated the clusters are with a silhouette, then check that score against a
null to show it didn't happen by chance. The null is built by making fake datasets that share
the real one's covariance but have no actual groups in them, clustering those the same way,
and counting how often the fake data scores as well as the real data (Dinga et al., 2019).

That is what we built. A GATv2 graph encoder trained with NT-Xent on connectivity graphs from
28 patients. Four clusters, silhouette 0.24, p = 0.024. It looked publishable.

Then we retrained it. Same inputs, byte for byte, only the random seed different, and we got
a different set of clusters. Agreement between runs was 0.23 on the adjusted Rand index,
where 1.0 means identical and 0 means chance. Real structure doesn't move like that.

Working out why took a while, and it wasn't the clustering. It was the null. Two separate
mistakes in how we built it, and either one on its own was enough to turn noise into a
result. Fix both and the same analysis gives p = 0.68. Across two disorders and 40 runs end
to end, nothing comes out significant.

The uncomfortable part is that the reproducibility checks never touch a null at all, and they
had it right from the start. The significance test was the broken thing, and it was the thing
we trusted.

## The two errors

The rule for a permutation null is that every fake dataset has to go through the exact same
procedure the real number came out of. Ours broke that in two places.

**Geometry.** Before clustering we scale every embedding to unit length. That's routine, and
it means only the direction of an embedding counts, not how long it is. Picture all 28
patients sitting on the surface of a sphere.

Our null draws never got scaled. They were generated in the full space, so instead of sitting
on the surface they were scattered through the inside of the sphere. Points spread through a
volume are harder to group tightly than points on a surface, so the fake data scored lower
than it should have, and beating it looked easier than it was. We were comparing the real
result against a null we'd accidentally handicapped.

**Selection.** We didn't decide the number of clusters up front. We tried *k* from 2 to 6 and
kept whichever scored best, which was 4. Choosing after seeing the results inflates the score
on its own, because the best of five tries beats a single try on average even when there's
nothing there.

That's fine as long as the null gets the same five tries. Ours didn't. Each fake dataset was
scored at one fixed *k*. Real data got five shots, the null got one. When you do give the null
the same search, 33.8% of draws do best at *k* = 2 and only 25.3% at the *k* = 4 we reported,
which is a decent measure of how much of our result was the search rather than the brains.

There was a third, smaller thing. The p-value used `c/n` with a strict inequality, so it could
return exactly 0, which isn't a valid p-value. It's `(c+1)/(n+1)` now.

Either of the first two is enough on its own. On the checkpoint where we first caught this, p
goes 0.024, then 0.177 with the geometry fixed, then 0.93 with the selection fixed as well.
On the rebuilt pipeline the corrected value is 0.6773. Through all of that the silhouette
never moves. Only the comparison does.

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

Saying we found nothing is only interesting if the test could have found something. A test
that never fires would give us this exact result on real structure too. So we checked it both
ways: does it fire when it shouldn't, and does it notice structure we deliberately put there.

**Does it fire on nothing?** (`analysis/calibration_run.py`) We made 200 datasets with no
groups in them at all, ran the corrected test on each, and counted. At α = 0.05 a well-behaved
test fires about 5% of the time. Ours fires 0.5%, roughly ten times too rarely.

That's a sample size problem. With 28 patients the estimated covariance is wobbly, so the fake
datasets come out carrying more apparent structure than the real data has, which pushes the
bar too high. Good news for our negative result, in that we're definitely not over-claiming.
Less good in that the test is blunter than its α implies.

**Does it notice real structure?** (`analysis/syntheticEmbeddings.py`) The other direction. We
took the real embeddings and shoved groups of patients apart by a known amount, so we knew
structure was there and knew exactly how strong. Then we ran the whole pipeline and watched.
1,440 of these:

| Planting offset | Mean ARI | Power |
|---|---|---|
| 0 – 6 | ≤ 0.15 | 0 % |
| 8 | 0.39 | 13 % |
| 10 | 0.62 | 38 % |
| 12 | 0.81 | 80 % |
| 20 | 0.98 | 100 % |

ARI is how well the recovered clusters match the ones we planted. Power is how often the test
actually called it. The pipeline is blind until the groups are quite far apart and doesn't
become reliable until offset 12, at which point you could see the clusters in a scatter plot
without any of this machinery.

At offset 0, where nothing is planted, the control lands on the real result exactly
(silhouette 0.2433, p = 0.6773) across all 120 cells. That's the sanity check on the planting
code itself: with no offset it should collapse back to the original analysis, and it does.

Together those two put a fence around what we can claim. The honest version is "no subtype
structure this method could detect at this sample size," not "no subtypes exist." Anything
obvious, we'd have caught. Anything subtle, we can't speak to, and neither can any paper
running this design on this many patients.

---

## Repository layout

```
config.py                    Single source of truth for hyperparameters and paths
figures.py                   Publication figures, entry point
requirements.txt

preprocessing/               fMRIPrep derivatives to timeseries to FC matrices
  compute_fc_matrices.py       Ledoit-Wolf shrinkage, Fisher z
  subject_filter.py            Motion and completeness exclusions
  clinical_to_csv.py           Clinical spreadsheet to clean CSV
  verify_setup.py

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
  null_progression.py          The four null constructions, end to end
  type1_surface.py             Type I error by representation geometry
  separation_ratio.py          Detection floor in interpretable units
  end_to_end_positive_control.py
  provenance.py                Hashes every artifact, writes PROVENANCE.md
  provenance_scripts/          The 14 drivers that produced the shipped artifacts

experiments/ds002748_mdd/    Depression replication (parallel pipeline)
tests/                       pytest suite
docs/
  index.html                   The browser demo (single file, no dependencies)
  demo/                        Embedding matrix and thumbnail the demo loads
  FM_Research_Data_Requirements.md
  repo_structure.md
notebooks/                   Exploratory only; no result depends on these
```

Most of `data/` and `results/` is gitignored, because it is large and regenerable from the
commands above. A small set is committed on purpose so the repository is not inert on a fresh
clone:

| Committed | Why |
|---|---|
| `data/tune/bestParams.json` | The chosen architecture. Without it `config` cannot resolve and nothing imports. |
| `data/outputs/*.npy`, `data/outputs/K-Means-Labeling.csv` | The trained 28 × 119 embeddings and their cluster labels, 60 KB total, enough to rerun every null analysis. |
| `data/Subjects/*_events.tsv`, `*_confounds.tsv` | Straight from the public OpenNeuro release, small, and needed to rebuild the FC matrices. |
| `data/schaefer200MNI.nii.gz` | The parcellation atlas. |

The trained checkpoint (1 MB), the per-condition FC matrices, and `data/clinical_clean.csv` are
not committed. **No subject-level clinical data is in this repository.**

**Presentation material lives outside this repository.** Conference videos, slide
figures, and the code that generates them were moved to `../presentation/`. They are
communication artifacts, not part of any result, and they were 400 MB of the repository.
Publication figures (`figures.py` → `results/figures/fig1…fig3`) remain here.

---

## Reproducing the full pipeline

Everything above runs on files that ship with the repo. Rebuilding the study from the raw
scans is a much bigger job. You need both OpenNeuro datasets, fMRIPrep, and a GPU for the
architecture search, and it takes days rather than minutes. Downloads are under Data
availability below, and `docs/FM_Research_Data_Requirements.md` spells out which files you
need and how to lay them out.

```bash
export OMP_NUM_THREADS=1        # not optional, see below
```

```bash
# 1. preprocessing (assumes fMRIPrep derivatives on disk)
python preprocessing/clinical_to_csv.py
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

The depression replication lives in `experiments/ds002748_mdd/` and runs in the same order:

```bash
cd experiments/ds002748_mdd
./run_mdd_pipeline.sh        # search, train, cluster
./run_mdd_analysis.sh        # severity regression, ablations
```

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
Preprocessed derivatives, the trained checkpoint, and per-run outputs are gitignored;
regenerate them with the commands above. The small artifacts needed to run the analyses
without a full rebuild are committed, listed under Repository layout.

## Key references

Dinga R, et al. Evaluating the evidence for biotypes of depression. *NeuroImage Clin.* 2019;22:101796.
Tozzi L, et al. Personalized brain circuit scores identify clinically distinct biotypes in depression and anxiety. *Nat Med.* 2024;30:2076-2087.
Grabski IN, Street K, Irizarry RA. Significance analysis for clustering with single-cell RNA-sequencing data. *Nat Methods.* 2023;20:1196-1202.
Balducci T, et al. A behavioral, clinical and brain imaging dataset with focus on emotion regulation of females with fibromyalgia. *Sci Data.* 2022;9(1).
Bezmaternykh DD, et al. Resting state with closed eyes for patients with depression and healthy participants. OpenNeuro; 2021.
Schaefer A, et al. Local-global parcellation of the human cerebral cortex. *Cereb Cortex.* 2018;28(9).
Brody S, Alon U, Yahav E. How attentive are graph attention networks? *ICLR* 2022.
Chen T, et al. A simple framework for contrastive learning of visual representations. *ICML* 2020.
