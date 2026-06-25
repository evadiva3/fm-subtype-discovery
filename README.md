# FM Subtype Discovery

Graph neural network–based self-supervised learning to discover fibromyalgia (FM) subtypes from multi-condition emotion regulation task fMRI.

## Overview

This project trains a 3-layer Graph Attention Network (GATv2) with NT-Xent contrastive loss on functional connectivity (FC) graphs built from 9 emotion regulation conditions (rest + 8 EPR tasks). K-means clustering on the learned embeddings reveals FM subtypes that are validated against clinical questionnaires (FIQ, PSQI, BDI, STAI, PCS).

## Setup

```bash
conda activate UTD
pip install -r requirements.txt
python preprocessing/verify_setup.py
```

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

### Step 4 — Train the Model

```bash
python src/train.py \
    --fc-dir ~/Desktop/DATA_clinical/fc_matrices \
    --clinical-csv ~/Desktop/DATA_clinical/clinical_data.csv \
    --device mps
```

### Step 5 — Hyperparameter Search (optional)

```bash
python src/hyperparameter_search.py \
    --fc-dir ~/Desktop/DATA_clinical/fc_matrices \
    --clinical-csv ~/Desktop/DATA_clinical/clinical_data.csv \
    --n-trials 50
```

### Step 6 — Cluster Subjects

```bash
python src/clustering.py \
    --checkpoint results/checkpoints/best_model.pt \
    --fc-dir ~/Desktop/DATA_clinical/fc_matrices \
    --clinical-csv ~/Desktop/DATA_clinical/clinical_data.csv \
    --k 3
```

### Step 7 — Clinical Validation & Figures

See `notebooks/03_results.ipynb`.

## Project Structure

```
fm-subtype-discovery/
├── preprocessing/   # fMRIPrep, ROI extraction, FC computation
├── models/          # GNN encoder, attention pooling, loss, augmentations
├── src/             # Training loop, clustering, hyperparameter search, ablations
├── analysis/        # Clinical validation, evaluation metrics, interpretability, figures
├── notebooks/       # Exploratory notebooks (Eva)
├── tests/           # Unit tests
├── paper/           # Manuscript draft and references
├── data/            # NOT in git — local data paths documented in data/README.md
└── results/         # NOT in git — model checkpoints, embeddings, figures
```

## Running Tests

```bash
/opt/anaconda3/envs/UTD/bin/pytest tests/ -v
```

## Data

See `data/README.md` for local data paths. Data is not committed to this repository.
