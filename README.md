AI Usage Note: AI was utilized to write this README, debug code, define data requirements, organize the repository structure, assist in paper writing, and provide conceptual clarity on complex topics.

# Significance Without Structure

**Researcher Degrees of Freedom in Small-Sample Self-Supervised Neuroimaging Subtyping**

Eva Bangsil, Nikhil [surname]. Correspondence: eva.bangsil@gmail.com

> Repository note: the directory is named `fm-subtype-discovery` for historical reasons. It began as a fibromyalgia subtype-discovery project and was reframed, after the results below, into a methodological study of when small-sample self-supervised subtyping can and cannot be trusted. The fibromyalgia pipeline is now the primary case study, not the contribution.

---

## What this project shows in one paragraph

Small-sample neuroimaging subtyping validates unsupervised representations by clustering embeddings, reporting a silhouette score, and testing significance with a permutation null, increasingly a covariance-preserving null (Dinga et al., 2019). We built a self-supervised GATv2 pipeline for this task and found that a single converged run produces exactly the result the procedure rewards: a four-cluster fibromyalgia partition that is significant under the covariance-preserving null (p = 0.024), with moderate bootstrap stability (adjusted Rand index 0.68). Re-running the identical pipeline 20 times on byte-identical inputs, with only the unseeded architecture search varying, that significance does not reproduce: 0 of 20 runs survive multiple-comparison correction, the selected cluster count wanders from two to five, and cross-run cluster agreement is 0.23. The significance is a property of the architecture search, not of the data. The failure replicates in an independent resting-state depression cohort. The repository contains the pipeline, the multi-run reproducibility protocol, and every analysis behind these claims.

---

## Headline results

| Quantity | Fibromyalgia (ds004144, 28 patients) | Depression (ds002748, 51 patients) |
|---|---|---|
| Trained silhouette (selected k) | 0.306 (k = 4) | 0.135 (k = 5) |
| Single-checkpoint permutation p | 0.024 | 0.205 |
| Permutation p across full-pipeline reruns | median 0.139; 0 of 20 significant after correction | multi-run protocol in progress |
| Cross-run cluster agreement (ARI) | 0.23 | in progress |
| Within-checkpoint bootstrap ARI (for contrast) | 0.68 | 0.32 |
| Untrained encoder silhouette / effective rank | 0.464 / 1.39 | 0.330 / 2.03 |
| Trained encoder effective rank | 3.39 | 7.74 |

The two ARI rows are the crux: bootstrap stability computed on one fixed checkpoint (0.68) overstates reproducibility roughly threefold relative to reproducibility across the architecture search (0.23). The standard stability check holds the checkpoint fixed and measures the wrong invariance.

---

## Method at a glance

- **Data.** OpenNeuro `ds004144` (fibromyalgia, emotion-regulation task-fMRI) and `ds002748` (major depressive disorder, resting-state).
- **Preprocessing.** fMRIPrep 24.1.0, Schaefer-200 parcellation, Ledoit-Wolf shrinkage on the condition-epoch covariance, Fisher z-transform.
- **Graphs.** 200-node graphs, edges retained in the top twentieth percentile of absolute Fisher z (global thresholding, signed weights), five signal-derived node features per parcel.
- **Model.** Three-layer GATv2 encoder (4 heads, per-head hidden width 58, 128-dim embedding), global mean-pool readout, and for fibromyalgia a learned condition-attention pool over the seven task conditions. Trained with NT-Xent contrastive loss. See `canonical_hyperparameters.md` for the exact configuration.
- **Validation battery.** Silhouette and gap statistic across k in {2..6}; effective rank; covariance-preserving permutation null (1000 draws); bootstrap partition stability (1000 resamples); random-initialization control; a full-pipeline multi-run reproducibility protocol.

---

## Repository layout

```
fm-subtype-discovery/
├── config.py                       # fixed (non-tuned) constants: seed 42, epochs, k range, edge percentile, etc.
├── data/
│   ├── tune/bestParams.json        # canonical tuned config, shared by FM and MDD (see canonical_hyperparameters.md)
│   ├── Subjects/                   # per-subject connectivity, timeseries
│   └── checkpoints/results/
│       └── bestJointModel.pt       # canonical FM checkpoint (dModel = 128)
├── models/
│   └── gnn_encoder.py              # GATv2 encoder + condition-attention pool
├── src/
│   ├── hyperparameter_search.py    # Optuna / Ray Tune search (unseeded; the source of the instability studied here)
│   ├── sensitivity_analysis.py     # percentile and (author-defined) config sweeps
│   ├── train_mdd.py                # depression-cohort training (reuses the FM config)
│   └── dataset_mdd.py
├── analysis/
│   └── figures.py                  # legacy figure set (null histogram, ablation bars, gap curve)
├── figures.py                      # manuscript figures 1 and 2, generated from on-disk results
├── figures.ipynb                   # thin notebook wrapper around figures.py
├── results/
│   ├── ablation_table.csv          # untrained / mean-pool / trained comparison
│   ├── bootstrap_stability.csv     # within-checkpoint bootstrap ARI
│   ├── sensitivity_percentile_sweep.csv
│   ├── figures/                    # rendered fig1, fig2
│   ├── fm_fixed_arch_control/      # fixed-architecture, vary-seed control (isolates model-selection vs seed noise)
│   └── mdd/
│       ├── mdd_silhouette_by_run.csv
│       ├── MDD_MULTIRUN_SUMMARY.txt
│       ├── effective_rank_mdd.csv
│       ├── confidence_intervals_mdd.csv
│       └── multirun/by_run/        # per-run params, silhouette, label vectors
├── FM_20run_stability/             # the central result
│   ├── by_run/                     # per-run silhouette, k, perm_p, label vectors
│   ├── FINAL_SUMMARY.txt
│   └── deterministic/canonical_run/# archived byte-identical FC/timeseries (hash-verified)
├── canonical_hyperparameters.md    # verbatim FM and MDD config, with provenance and checksums
└── README.md
```
---

## Environment

```
python >= 3.10
torch, torch-geometric        # GATv2 encoder
nilearn                       # connectivity, Ledoit-Wolf, CanICA baseline
scikit-learn, scipy, numpy, pandas
ray[tune], optuna             # hyperparameter search
matplotlib                    # figures
```

Preprocessing is run separately under fMRIPrep 24.1.0 (Docker). Training was run on Apple Silicon (MPS) and an RTX 5080 (CUDA); the depression multi-run protocol uses a reduced batch (`ebs = 15`) as an MPS resource adaptation, documented in `canonical_hyperparameters.md`.

---

## Reproducing the central result

The claim of this project is a reproducibility claim, so the repository is built to be re-run rather than trusted. The key protocol runs the complete pipeline end to end many times on byte-identical inputs and inspects the distribution of the significance verdict, rather than any single run.

1. **Confirm the canonical configuration.** `data/tune/bestParams.json` and `canonical_hyperparameters.md` define the exact architecture and tuned values. Both cohorts use this same configuration; the depression encoder was not independently tuned.

2. **Single-checkpoint result.** Train once, cluster the patient embeddings, and compute the covariance-preserving permutation null. This reproduces the k = 4, silhouette 0.306, p = 0.024 fibromyalgia partition.

3. **Full-pipeline reproducibility protocol (the real result).** Re-run the complete pipeline 20 times with upstream inputs fixed and hash-verified, training and clustering seeded, and only the Optuna search left unseeded. Outputs land in `FM_20run_stability/by_run/`. `FINAL_SUMMARY.txt` reports the permutation-p distribution (min 0.005, median 0.139, max 0.793; 0 of 20 significant after correction), the selected-k distribution, and the cross-run adjusted Rand index (0.23).

4. **Fixed-architecture control.** To confirm the instability comes from model selection rather than training-seed noise, `results/fm_fixed_arch_control/` holds the architecture fixed at the canonical config and varies only the training seed. Comparing this distribution against the search-varying protocol isolates the source.

5. **Figures.** `figures.py` reads the CSVs above and regenerates manuscript Figures 1 and 2; `figures.ipynb` is a thin wrapper. No result values are hardcoded in the figure code.

Byte-identical FC and timeseries for the fibromyalgia protocol are archived under `FM_20run_stability/deterministic/canonical_run/` and are checksum-verified; see `canonical_hyperparameters.md` for the hashes.

---

## Data availability

Both datasets are public on OpenNeuro: `ds004144` (fibromyalgia; Balducci et al., 2022, *Scientific Data*) and `ds002748` (depression). Clinical variables for the fibromyalgia cohort are drawn from the associated Zenodo deposit. No participant-level data are redistributed in this repository.

---

## A note on scope

This work reports a negative, methodological result. It does not claim a fibromyalgia or depression subtype, and it does not claim the underlying model is broken; it claims that the standard validation chain, at these sample sizes, can certify structure that does not reproduce. The recommendation is procedural: report permutation-null significance as a distribution over full-pipeline retrains, assess reproducibility across the architecture search rather than only under resampling, and report effective rank alongside any silhouette.
---
## Contributions
Eva Bangsil and Nikhil Joshi 
