AI Usage Note: AI was utilized to write this README, debug code, define data requirements, organize the repository structure, and provide conceptual clarity on complex topics.

# FM Subtype Discovery

Graph neural network–based self-supervised learning to discover fibromyalgia (FM) subtypes from short-epoch task fMRI.

## Overview

This project trains a 3-layer Graph Attention Network (GATv2) with NT-Xent contrastive loss on functional connectivity (FC) graphs built from 7 emotion regulation conditions (Neutral-Observe, Negative-Observe, Negative-Reduce, Negative-Suppress, Happy-Observe, Happy-Suppress, Happy-Increase). A learned-temperature attention pool aggregates per-condition embeddings into a subject-level representation. K-means clustering on the pooled embeddings, combined with orthogonal complement projection to isolate within-FM structure from FM-HC group separation, is used to test for FM subtypes. Cluster validity is assessed via silhouette score, permutation testing, and the gap statistic; discovered subtypes are compared against clinical variables (HAMD, HAMA, TAS-20 and subscales, ERQ, VAS pain).

## Setup

```bash
conda activate UTD
pip install -r requirements.txt
python preprocessing/verify_setup.py
```

`verify_setup.py` computes framewise displacement per subject, flags motion and missing-file exclusions, and writes the exclusion manifest (`results/subject_exclusions.csv`) that the rest of the pipeline reads from. Run this before anything else — the downstream preprocessing and dataset steps depend on this manifest existing.

## Configuration

All paths and hyperparameters are centralized in `config.py`. Data paths default to environment variables (`FM_DATA_ROOT`, `FM_RESULTS_ROOT`) with local fallbacks — no CLI flags needed for paths. Model hyperparameters (`d_model`, `heads`, `layers`, `dropout`, `lr`, etc.) are read from `data/tune/bestParamTune.json`, produced by the hyperparameter search step below.

## Workflow

### Step 1 — fMRI Preprocessing

```bash
bash preprocessing/run_fmriprep.sh
```

### Step 2 — Extract ROI Time Series

```bash
python preprocessing/extract_roi_timeseries.py
```

### Step 3 — Compute FC Matrices

```bash
python preprocessing/compute_fc_matrices.py
```

Reads the exclusion manifest from Step 0 and skips excluded subjects. Applies Ledoit-Wolf shrinkage and a high-pass filter (0.01Hz, full run, pre-slicing) before computing per-condition FC matrices.

### Step 4 — Hyperparameter Search

```bash
python src/hyperparameter_search.py
```

Writes `data/tune/bestParamTune.json`, which `config.py` reads for all model hyperparameters from this point forward.

### Step 5 — Train the Model

```bash
python src/train.py
```

End-to-end joint training of the GATv2 encoder and condition attention pool via NT-Xent contrastive loss, using hyperparameters from Step 4.

### Step 6 — Cluster Subjects

```bash
python src/clustering.py
```

Extracts embeddings, applies attention pooling, runs FM-only k-means (both on raw and orthogonal-projected embeddings), and reports silhouette score, permutation p-value, and gap statistic per k.

### Step 7 — Clinical Validation, Sensitivity Analysis, and Figures

```bash
python analysis/clinical_validation.py
python analysis/sensitivity_analysis.py
python analysis/bootstrap_stability.py
```

See `notebooks/03_results.ipynb` for figure generation.

## Project Structure

```
fm-subtype-discovery/
├── config.py         # Centralized paths and hyperparameters
├── preprocessing/    # fMRIPrep, ROI extraction, FC computation, exclusion manifest
├── models/           # GNN encoder, attention pooling, loss, augmentations
├── src/              # Training loop, clustering, hyperparameter search, ablations
├── analysis/         # Clinical validation, evaluation metrics, interpretability, figures
├── notebooks/        # Exploratory notebooks
├── tests/            # Unit tests
├── paper/            # Manuscript draft and references
├── data/             # Local data, including the exclusion manifest and tuned params
└── results/          # Model checkpoints, embeddings, figures
```

## Running Tests

```bash
/opt/anaconda3/envs/UTD/bin/pytest tests/ -v
```

## Data

See `data/README.md` for local data paths and the exclusion manifest.

## Authorship

Eva Bangsil, Nikhil Joshi
