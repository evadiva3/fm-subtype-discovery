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

## Try it in your browser

[Live demo](https://evadiva3.github.io/fm-subtype-discovery/). Nothing to install.

[![The four null constructions running in the browser.](docs/demo/thumbnail_1600.png)](https://evadiva3.github.io/fm-subtype-discovery/)

It loads the real 28 x 119 embedding matrix and does the clustering, the silhouette and every
null draw in front of you. Nothing is precomputed. Press "Run the full ladder", wait twenty
seconds:

| Construction | Geometry matched | Selection matched | *p* |
|---|---|---|---|
| Misspecified, as originally implemented | No | No | 0.1698 |
| Geometry corrected only | Yes | No | 0.4436 |
| Selection corrected only | No | Yes | 0.3966 |
| Fully corrected | Yes | Yes | 0.6773 |

Same data, same silhouette, same four clusters in every row. Only the comparison changes, and
the result goes from significant to not.

Those are 1,000-draw values; at 20,000 they read 0.1685 / 0.4630 / 0.4033 / 0.7009. Silhouette
agrees with the Python to seven decimals (0.2433061).

Source is `docs/index.html`. One file, no dependencies, no build step.

---

## Setup

Python 3.10+. Tested on 3.12, 3.13, 3.14.

```bash
git clone https://github.com/evadiva3/fm-subtype-discovery.git
cd fm-subtype-discovery
python3 -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

export OMP_NUM_THREADS=1
python -m pytest tests/ -q
```

You want `12 passed`. If you get `FileNotFoundError: Tuned hyperparameter 'dModel'`, your clone
is missing `data/tune/bestParams.json`. It is committed, so pull again.

Four things that will otherwise cost you an hour:

- **`OMP_NUM_THREADS=1` is not general advice.** scikit-learn's k-means parallelizes with
  OpenMP, and on a 28-point problem thread coordination costs more than the arithmetic:
  2.90 ms/fit single-threaded against 74.07 ms at 32 threads. ~25x faster, byte-identical.
  Parallelize across processes, keep each single-threaded.
- **`ray` has no wheel for Python 3.13+.** `src/train.py` imports it at module load and only
  uses it inside the hyperparameter search. A stub whose `tune.report` raises is enough.
- **`umap-learn` is imported by `src/clustering.py`** even though no null analysis uses it.
- **The depression data must be a sibling of the repo**, not inside it. See tier 2.

---

## Reproducing

Three tiers by what each needs.

### Tier 1 — clean clone, CPU only

```bash
export OMP_NUM_THREADS=1
export FM_RESULTS_ROOT=/tmp/fmresults    # optional, keeps outputs out of the tree
```

| Command | Reproduces | Time |
|---|---|---|
| `python analysis/null_progression.py 1000` | the four-rung ladder, Tables 2A / 13 | 2 min |
| `python analysis/null_progression.py 20000` | same at the headline count | 35 min |
| `NDATASETS=1000 NDRAWS=1000 MATCH_SELECTION=0 MATCH_GEOMETRY=0 python analysis/calibration_run.py` | misspecified Type I error, 0.205 | 5 min |
| `NDATASETS=1000 NDRAWS=1000 MATCH_SELECTION=1 MATCH_GEOMETRY=1 python analysis/calibration_run.py` | corrected Type I error, 0.004 | 20 min |
| `python analysis/type1_surface.py 200 200` | inflation tracks geometry, not n | 3 min |
| `python analysis/syntheticEmbeddings.py` | planted-cluster control, Table 15 | 30 min |
| `python analysis/separation_ratio.py` | separation-ratio conversion, 0.584 / 1.166 / 1.725 | <1 min |
| `cd docs && python3 -m http.server 8000` | the demo on localhost:8000 | instant |

Run `syntheticEmbeddings.py` before `separation_ratio.py`: the latter converts offsets to
separation ratios on its own, but pinning those to the 1.166 and 1.725 power anchors needs the
detection curve the positive control writes. Without it you get the conversion table and a
"floor anchors unavailable" note.

Times are for 24 cores at `OMP_NUM_THREADS=1`. The ladder at 1,000 draws lands near
0.170 / 0.444 / 0.397 / 0.677. k-means init on the sphere is mildly scikit-learn-version
sensitive, so the two sphere rungs can move a few draws in a thousand. No verdict moves.

### Tier 2 — needs the depression timeseries (5.6 MB)

`dataset_mdd.py` hardcodes its data directory as a repo sibling, with no config override:

```
UTD-PROJECT/
├── fm-subtype-discovery/       <- this repo
└── resting_state_dep_data/     <- 72 *_rest_ts.npy + participants.tsv
```

Check before running anything:

```bash
python3 -c "
import sys; sys.path.insert(0,'experiments/ds002748_mdd')
from dataset_mdd import MDD_ROOT
print(MDD_ROOT, MDD_ROOT.exists())
"
```

`subject_filter_mdd.py` needs fMRIPrep confounds that are not shipped. Without them it returns
an empty cohort and `mddDataset()` dies on `stack expects a non-empty TensorList`. All 72
participants passed the motion rule and none was excluded (paper §3.1), so the filter has no
work to do; the scripts under `experiments/ds002748_mdd/propagated_null/` override it and
assert 72 = 51 depressed + 21 control before proceeding.

Needs GPU torch and `torch_geometric`; `mdd_validate_forward.py` also needs `nilearn`.

| Command | Reproduces | Time |
|---|---|---|
| `python experiments/ds002748_mdd/propagated_null/mdd_validate_forward.py` | forward-path validation, run this first | 2 min |
| `python experiments/ds002748_mdd/propagated_null/mdd_propagated_null.py observed` | the observed statistic on real data | 30 s |
| `python experiments/ds002748_mdd/propagated_null/mdd_propagated_null.py 0 500` | the propagated null, *p* = 0.8124 | 3.5 h serial |
| `python experiments/ds002748_mdd/propagated_null/mdd_fixed_arch_control.py 0 20` | 20-seed fixed-architecture control | 1 h |

The null is parallel across draws. Give each worker its own `MDD_NULL_ROOT` or they overwrite
each other's surrogate cohorts, run disjoint ranges, merge the `draws/` dirs. Ten workers on one
GPU did 500 draws in 2.5 hours — the model is small enough that the GPU saturates on kernel
launches, so ten workers buy well under 10x.

### Tier 3 — needs preprocessed imaging we cannot ship

Both OpenNeuro datasets, fMRIPrep, and a GPU for the search. Days, not hours.
`docs/FM_Research_Data_Requirements.md` has the file layout.

```bash
python preprocessing/clinical_to_csv.py
python preprocessing/verify_setup.py
python preprocessing/compute_fc_matrices.py
python src/hyperparameter_search.py        # -> data/tune/bestParams.json
python src/train.py                        # -> data/checkpoints/results/bestJointModel.pt
python src/clustering.py                   # -> data/outputs/
python analysis/ablation_table.py
python analysis/bootstrap_stability.py
python analysis/run_baselines.py
python analysis/clinical_validation.py
python analysis/effective_rank.py
python analysis/severity_gradient_regression.py
python analysis/raw_feature_ladder.py
python figures.py
```

`raw_feature_ladder.py` is tier 3 because it reads the per-condition FC matrices, which are
gitignored. Its result is committed at `results/raw_feature_ladder.json`.

Depression pipeline: `experiments/ds002748_mdd/run_mdd_pipeline.sh`, then `run_mdd_analysis.sh`.

---

## Results

### Fibromyalgia (ds004144): 58 scanned, 28 patients

| | |
|---|---|
| Canonical architecture | dModel 119, heads 8, output 15, layers 3 |
| Clustering | *k* = 4, silhouette 0.2433, p = 0.6773 (1,000 draws) / 0.7009 (20,000) |
| Propagated null, simulated in the raw data space | **p = 0.9062** (500 draws) |
| Independent architectures | 0 of 15 significant (p 0.272 – 0.825) |
| Fixed-architecture control, seed only | 0 of 20 (p 0.083 – 0.974) |
| 20-run search-varying protocol | 0 of 20 (min 0.131, median 0.576) |
| Cross-run agreement (ARI) | 0.289 (14 runs) / 0.230 (20 runs) |
| Within-checkpoint bootstrap | 0.5676, twice as high, measuring the wrong variability |
| Clinical variables surviving FDR | 0 of 10 (every corrected p = 0.896) |
| Untrained vs trained silhouette | 0.4600 vs 0.2433 |
| Participation ratio, untrained / trained | 1.585 / 3.703 |

### Depression (ds002748): 72 scanned, 51 patients

| | |
|---|---|
| Canonical architecture | dModel 68, heads 4, output 8, layers 3 (independently searched) |
| Clustering | *k* = 5, silhouette 0.142, p = 0.3843 |
| Propagated null | **p = 0.8124** (500 draws) |
| 20-run protocol | 0 of 20 (median 0.590) |
| Fixed-architecture control, seed only | 0 of 20 (p 0.086 – 0.927) |
| Cross-run agreement | 0.226 (20 runs); 0.233 across seeds at fixed architecture |
| Severity gradient | none on three scales; every out-of-sample R² negative |

Cross-retrain agreement lands between 0.226 and 0.289 across both cohorts, two search spaces
and four protocols. A reproducible subtype needs 0.7.

### Baselines (fibromyalgia)

| Method | Value | p |
|---|---|---|
| PCA + k-means | silhouette 0.057 | 0.983 |
| Flat upper-triangle k-means | silhouette 0.060 | 0.886 |
| Group ICA + k-means | silhouette 0.377 | 0.197 |
| Supervised SVM (patients vs controls) | 0.503 accuracy | 0.403, below majority class |

---

## What happened

The usual way to claim subtypes: learn an embedding per patient, cluster, report a silhouette,
test it against a null built from fake datasets that share the real covariance but have no
groups in them (Dinga et al., 2019).

That is what we built. GATv2 encoder, NT-Xent, 28 patients. Four clusters, silhouette 0.24,
p = 0.024. It looked publishable.

Then we retrained it. Same inputs byte for byte, different seed, different clusters. Agreement
across runs was 0.23 on the adjusted Rand index. Real structure doesn't move like that.

It wasn't the clustering. It was the null, in two places, either of which alone turns noise
into a result.

**Geometry.** We L2-normalize embeddings before clustering, so all 28 patients sit on the
surface of a sphere. The null draws never got normalized — they were scattered through the
inside. Points in a volume are harder to group tightly than points on a surface, so the fake
data scored low and beating it looked easy.

**Selection.** We tried *k* from 2 to 6 and kept the best, which was 4. Each null draw was
scored at one fixed *k*. Real data got five shots, the null got one. Give the null the same
search and 33.8% of draws do best at *k* = 2, only 25.3% at our *k* = 4.

A third, smaller thing: the p-value used `c/n` with a strict inequality and could return
exactly 0. It's `(c+1)/(n+1)` now.

Fix both and that checkpoint goes to p = 0.93; the canonical checkpoint this repo ships goes
from 0.1685 to 0.7009. Those are two different models — the 0.024 belongs only to the first.
Across two disorders and 40 end-to-end runs nothing is significant. The silhouette never moves,
only the comparison does.

The uncomfortable part is that the reproducibility checks never touch a null, and they had it
right from the start. The significance test was the broken thing, and it was the thing we
trusted.

## Correcting the null isn't enough

Every rung above still builds the null downstream of the representation, so connectivity
estimation, graph construction, training and pooling never enter it.

So we replaced it: surrogate cohorts simulated in the raw data space from a group-level
cross-spectral model, written to disk in the real subject layout, run through the unmodified
pipeline 500 times per cohort. Fibromyalgia **p = 0.9062**, depression **p = 0.8124**, and in
both the propagated null's mean silhouette exceeds the corrected embedding-space null's.

A propagated null retrains on every draw, so its spread mixes data variability with whatever
training nondeterminism the pipeline amplifies — 80% of the variance in fibromyalgia,
essentially none in depression. Run the paired check that separates them: re-run ~30 draws at
identical seeds and see whether the selected *k* holds.

## It only bites collapsed representations

Run the same four constructions on the raw connectivity edges — participation ratio 20.5
against the embedding's 4.37 — and the ladder disappears: p = 0.991 / 0.996 / 0.995 / 0.998,
null means agreeing to 0.004 against 0.043 in the embedding.

So covariance-preserving nulls aren't broken. They break on the anisotropic, effectively
low-dimensional representations contrastive encoders produce, which is where this literature
is heading. `results/raw_feature_ladder.json`.

## What bounds the negative result

**Does the test fire on nothing?** 1,000 structureless datasets under both constructions on
the same data. Misspecified rejects at **20.5%** against a nominal 5%, corrected at **0.4%**.
Of the discordant pairs, 201 were misspecified-only and **zero** corrected-only. The inflation
is one-directional.

The corrected test is conservative by 12.5x on the reported run, 95% CI 4.9x–31x. That point
estimate rests on four rejection events and comes back as 10x on re-execution, so quote the
interval, not the number.

**Does it notice real structure?** 1,440 realizations with clusters planted at known
separation:

| Offset | Mean ARI | Power |
|---|---|---|
| 0 – 6 | ≤ 0.15 | 0 % |
| 8 | 0.39 | 13 % |
| 10 | 0.62 | 38 % |
| 12 | 0.81 | 80 % |
| 20 | 0.98 | 100 % |

In transferable units: structureless sits at 0.584, power starts at 1.166, hits 80% at 1.725.
At offset 0 the control lands on the real result exactly (0.2433, p = 0.6773) across all 120
cells.

**And the pipeline can't retrieve structure that is present in its own inputs.** The
connectivity features identify subjects at 93.1% against 1.7% chance and decode task condition
at twice chance. Clustering the encoder's embeddings of the same graphs recovers neither (ARI
0.000 and 0.061); supervised decoding from those embeddings falls to 18.3% and 11.8%. The loss
is in the encoder, not the clustering.

So: no subtype structure this method could detect at this sample size. Not: no subtypes exist.
Anything obvious we'd have caught. Anything subtle we can't speak to, and neither can any
paper running this design on this many patients.

---

## Layout

```
config.py                    hyperparameters and paths, single source of truth
figures.py                   publication figures

preprocessing/               fMRIPrep derivatives -> timeseries -> FC matrices
models/                      GATv2 encoder, attention pool, NT-Xent, augmentations
src/                         hyperparameter_search.py, train.py, clustering.py

analysis/
  evaluate.py                  ** the corrected permutation null **
  null_progression.py          the four constructions, end to end
  calibration_run.py           ** the 1,000 x 1,000 paired calibration **
  type1_surface.py             Type I error by representation geometry
  raw_feature_ladder.py        ** the ladder in the raw connectivity space **
  separation_ratio.py          detection floor in interpretable units
  syntheticEmbeddings.py       planted-cluster positive control
  baselines.py, bootstrap_stability.py, clinical_validation.py,
  severity_gradient_regression.py, knn_probe.py, effective_rank.py,
  sensitivity_analysis.py, e2e_calibration.py, provenance.py, orchestrator.py

experiments/ds002748_mdd/    depression replication
  propagated_null/             ** the raw-data-space null for this cohort **
    mdd_propagated_null.py, mdd_validate_forward.py,
    mdd_fixed_arch_control.py, mdd_params.json

results/
  figures/                     fig1..fig8 as cited in the paper
  stability_summary.csv        the 14 stability runs behind Table 5
  raw_feature_ladder.json
  mdd/
    canonical_single/          checkpoint, bestParamsMdd.json, embeddings, labels
    multirun_v2/               20-run protocol, per-run embeddings and labels
    propagated_null/           500 draws, summary, paired re-run
    fixed_arch_control/        20 seeds, per-seed embeddings, permutation p

tests/                       pytest suite, 12 tests
docs/index.html              the browser demo
```

Most of `data/` and the fibromyalgia half of `results/` is gitignored — large and regenerable.
Committed on purpose so a fresh clone isn't inert:

| Committed | Why |
|---|---|
| `data/tune/bestParams.json` | the chosen architecture; without it `config` cannot resolve |
| `data/outputs/*.npy`, `K-Means-Labeling.csv`, `silhouette-scores.csv` | trained embeddings and labels, 60 KB, enough for every tier-1 analysis |
| `data/Subjects/*_events.tsv`, `*_confounds.tsv` | from the OpenNeuro release, needed to rebuild FC matrices |
| `data/schaefer200MNI.nii.gz` | the parcellation atlas |
| `results/mdd/**`, `results/figures/**`, `results/raw_feature_ladder.json`, `results/stability_summary.csv` | each needs retraining to regenerate |

The fibromyalgia checkpoint, the FC matrices and `data/clinical_clean.csv` are not committed.
**No subject-level clinical data is in this repository.**

## Data

Both datasets are public on OpenNeuro: **ds004144** (fibromyalgia task-fMRI, Balducci et al.,
2022) and **ds002748** (depression resting-state, Bezmaternykh et al., 2021).

The depression pipeline reads only the parcellated timeseries — 72 `*_rest_ts.npy` plus
`participants.tsv`, 5.6 MB. That is what the sibling `resting_state_dep_data/` needs to hold.
The raw BIDS NIfTI is archival; re-parcellating would need the fMRIPrep derivatives, not the
raw scans.

## If you extend this

**The null is in `analysis/evaluate.py`.** `_null_mvn` fits the Gaussian to the
**L2-normalized** embeddings and re-projects each draw onto the sphere; `perm` defaults to
`match_selection=True`, so every draw repeats the full *k*-search under the same guard. Both
are load-bearing. Reverting either reintroduces one of the two errors.

**Checkpoints carry `nodeMean`/`nodeStd`.** Inference must normalize with the training split's
statistics. Older checkpoints emit a fallback warning and are not usable for inference.

**Batch size is fixed, not searched.** NT-Xent's denominator is the batch, so a larger batch
records a worse validation loss for reasons unrelated to architecture. Searching it ranks
batch sizes, not architectures.

**Open issue: NT-Xent temperature.** Across the 14 stability searches plus the canonical run,
the selected temperature spans 0.0501 to 0.0668 and lands within 20% of the 0.05 floor in 13
of 15. The bound is binding and the optimum is below it. The range is now log-uniform
[0.01, 1.0], but that only takes effect on the next search — every architecture in the paper
was selected under the old bound. No conclusion depends on it: the null result is stable
across all fifteen architectures, dModel 19 to 119.

## References

Dinga R, et al. Evaluating the evidence for biotypes of depression. *NeuroImage Clin.* 2019;22:101796.
Tozzi L, et al. Personalized brain circuit scores identify clinically distinct biotypes in depression and anxiety. *Nat Med.* 2024;30:2076-2087.
Grabski IN, Street K, Irizarry RA. Significance analysis for clustering with single-cell RNA-sequencing data. *Nat Methods.* 2023;20:1196-1202.
Balducci T, et al. A behavioral, clinical and brain imaging dataset with focus on emotion regulation of females with fibromyalgia. *Sci Data.* 2022;9(1).
Bezmaternykh DD, et al. Resting state with closed eyes for patients with depression and healthy participants. OpenNeuro; 2021.
Schaefer A, et al. Local-global parcellation of the human cerebral cortex. *Cereb Cortex.* 2018;28(9).
Brody S, Alon U, Yahav E. How attentive are graph attention networks? *ICLR* 2022.
Chen T, et al. A simple framework for contrastive learning of visual representations. *ICML* 2020.
