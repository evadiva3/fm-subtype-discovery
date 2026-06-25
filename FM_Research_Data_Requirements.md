# FM Subtype Discovery — Data Requirements Master Reference

## 1. PRIMARY DATA SOURCES

### 1.1 OpenNeuro ds004144 (Imaging)
- **URL:** https://openneuro.org/datasets/ds004144/versions/1.0.2
- **Access:** No account, approval, or DUA required — immediate download
- **Format:** BIDS

| Field | Value |
|---|---|
| FM patients | 33 (female) |
| Healthy controls | 33 (female) |
| Recruitment site | Mexico |
| Total subjects | 66 |

**Modalities to download:**
- `T1w` — structural MRI (all subjects)
- `T2w` — structural MRI (all subjects)
- `bold` resting-state fMRI — standard EPI
- `bold` task-fMRI — emotion processing/regulation task (EPRT)
  - Duration: 27.8 minutes
  - Volumes per subject: 834
  - TR: check dataset header
  - Task conditions: **Attend**, **Reappraise**, **Suppress**
  - Stimulus valences: **positive**, **negative**, **neutral**
  - Total condition cells: 3 strategies × 3 valences = **9 FC matrices per subject**
- `events.tsv` files — trial-level condition/timing labels (required for GLM contrasts and condition-level FC extraction)
- `participants.tsv` — subject-level metadata in BIDS root

**What NOT to download (not needed for this study):**
- Field maps (only needed if fMRIPrep flags them — grab if preprocessing fails without them)

---

### 1.2 Zenodo Clinical/Behavioral Data
- **DOI:** https://doi.org/10.5281/zenodo.6554870
- **Access:** Free download, no approval required

**Required variables — keep all of these:**

| Variable | Instrument | Used For |
|---|---|---|
| Pain intensity | Visual Analogue Scale (VAS) | Primary clinical outcome |
| Depression | Beck Depression Inventory (BDI) | Clinical validation |
| Anxiety | Beck Anxiety Inventory (BAI) | Clinical validation |
| Alexithymia | Toronto Alexithymia Scale (TAS-20) | Clinical validation |
| Emotion regulation | Emotion Regulation Questionnaire (ERQ) — subscales: Cognitive Reappraisal, Expressive Suppression | Clinical validation, maps to task conditions |
| Age | Sociodemographic | Covariate control |
| Education (years) | Sociodemographic | Covariate control |
| Disease duration | Sociodemographic | Clinical validation |
| Number of medications | Sociodemographic | Clinical validation |
| Subject ID | — | Merge key with imaging data |
| Group label (FM / HC) | — | Permutation testing, baseline SVM |

**Merge strategy:** Join Zenodo CSV to participants.tsv on subject ID. Verify ID format alignment (sub-XXXX in BIDS vs. numeric IDs in Zenodo — may need a crosswalk).

---

## 2. DERIVED / PROCESSED DATA (outputs of pipeline, not downloaded)

These are computed from raw data — do not need to download externally.

### 2.1 fMRIPrep Outputs (per subject)
- `*_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz` — preprocessed BOLD
- `*_confounds_timeseries.tsv` — motion + physiological confounds
- `*_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz` — brain mask
- `*_T1w_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz` — normalized T1

**What fMRIPrep applies:**
- Motion correction (6-parameter rigid body)
- Slice timing correction
- Spatial normalization to MNI152 (2mm³ resolution)
- Confound regression (24 motion params + WM/CSF signals minimum)

**Expected dropout:** 10–15% from excessive motion (>0.5mm FD threshold). Target: ≥28 FM subjects post-QC.

---

### 2.2 Atlas
- **Schaefer 200-parcel, 7-network parcellation** — fetched via `nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=200)`
- Produces 200 ROI time series per subject per condition block

---

### 2.3 Functional Connectivity Matrices
- Shape: `200 × 200` Pearson correlation matrix
- One matrix per condition cell (3 strategies × 3 valences = **9 per subject**)
- Also compute 1 resting-state FC matrix per subject (for ablation Study 3)
- Total matrices: 66 subjects × 10 conditions = **660 FC matrices**
- Storage estimate: each matrix at float32 = 200×200×4 bytes ≈ 160KB → total ~105MB

---

### 2.4 Graph Dataset (PyTorch Geometric)
Per subject graph node features (200 nodes × 5 features):

| Feature | Description |
|---|---|
| Mean BOLD signal | Per parcel, per condition block |
| BOLD variance | Per parcel, per condition block |
| Spectral power band 1 | Low frequency (0.01–0.08 Hz) |
| Spectral power band 2 | Mid frequency (0.08–0.15 Hz) |
| Spectral power band 3 | High frequency (0.15–0.25 Hz) |

Edge weights: pairwise FC correlation values (200×200 adjacency).

---

### 2.5 Model Outputs
- **Per-subject per-condition embeddings:** shape `(subjects × 9 conditions, 64)` — output of GAT encoder
- **Pooled subject embeddings:** shape `(subjects, 64)` — after attention pooling across conditions
- **Attention weights:** shape `(subjects, 200, 9)` — GAT node attention per parcel per condition
- **Cluster assignments:** shape `(33_FM_subjects,)` — k-means labels (k=2,3,4 runs)
- **Silhouette scores:** scalar per k
- **Null distribution:** 1000 permuted silhouette scores (permutation test)

---

## 3. DATA FILES TO CREATE / MAINTAIN LOCALLY

```
data/
  raw/
    ds004144/              ← OpenNeuro download (BIDS structure intact)
      sub-*/
        anat/
        func/
      participants.tsv
      dataset_description.json
    zenodo_clinical/       ← Zenodo download
      clinical_data.csv    ← or whatever filename Zenodo provides
  
  processed/
    fmriprep/              ← fMRIPrep output directory
      sub-*/
        func/
        anat/
    roi_timeseries/        ← extracted via nilearn
      sub-*_task-EPRT_cond-*.npy   (shape: n_volumes × 200)
      sub-*_task-rest.npy
    fc_matrices/           ← computed from time series
      sub-*_task-EPRT_cond-*.npy   (shape: 200 × 200)
      sub-*_task-rest.npy
  
  merged/
    subjects_master.csv    ← participants.tsv + Zenodo clinical, merged on subject ID
    qc_log.csv             ← subjects excluded, reason, FD stats
```

---

## 4. CLINICAL DATA COLUMNS — subjects_master.csv

Minimum required schema for the merged master file:

```
subject_id        | str   | BIDS subject identifier (e.g., sub-001)
group             | str   | "FM" or "HC"
age               | float | years
education         | float | years
disease_duration  | float | years (FM only; NaN for HC)
n_medications     | int   | (FM only; NaN for HC)
vas_pain          | float | 0–100 VAS score
bdi_total         | float | Beck Depression Inventory total
bai_total         | float | Beck Anxiety Inventory total
tas20_total       | float | Toronto Alexithymia Scale total
tas20_dif         | float | TAS-20 subscale: Difficulty Identifying Feelings
tas20_ddf         | float | TAS-20 subscale: Difficulty Describing Feelings
tas20_eot         | float | TAS-20 subscale: Externally Oriented Thinking
erq_reappraisal   | float | ERQ Cognitive Reappraisal subscale
erq_suppression   | float | ERQ Expressive Suppression subscale
qc_pass           | bool  | True if subject passed motion QC
mean_fd           | float | Mean framewise displacement (from fMRIPrep confounds)
cluster_label     | int   | Assigned post-clustering (FM only; NaN for HC)
```

---

## 5. WHAT THIS STUDY DOES NOT NEED

- Paid datasets (e.g., UK Biobank, ABCD) — not used
- Additional data collection — no participants to recruit
- External validation cohort — noted as Year 2 goal only
- DWI/DTI data — not in ds004144, not needed
- EEG/MEG — not collected
- Pharmacological trial data — not applicable

---

## 6. ANALYSIS INPUTS BY PIPELINE STAGE

| Stage | Input Data | Output |
|---|---|---|
| fMRIPrep | Raw BIDS (T1w + task BOLD + events) | Preprocessed BOLD + confounds |
| ROI extraction | Preprocessed BOLD + Schaefer atlas | 200 × T timeseries per condition |
| FC computation | ROI timeseries + events.tsv (condition blocking) | 9 × 200×200 FC matrices per subject |
| Graph dataset | FC matrices + node features | PyG Data objects |
| GAT training | Graph dataset (FM+HC or FM only — decide) | Trained encoder weights |
| Embedding extraction | Trained encoder + all subject graphs | 64-dim embeddings |
| Attention pooling | Per-condition embeddings | 1 pooled embedding per subject |
| Clustering | FM-only pooled embeddings | Cluster labels |
| Clinical validation | Cluster labels + subjects_master.csv | Mann-Whitney U results, effect sizes |
| Interpretability | GAT attention weights + Schaefer atlas | Brain maps per subtype |
| Ablations | Resting-state FC, raw FC matrices, PCA outputs | Baseline silhouette/effect sizes |

---

## 7. QUICK DOWNLOAD CHECKLIST

- [ ] Download ds004144 via `openneuro.org` or `aws s3 sync --no-sign-request s3://openneuro.org/ds004144 ds004144/`
- [ ] Download Zenodo clinical data: `wget https://zenodo.org/record/6554870/files/<filename>`
- [ ] Verify BIDS validator passes: `bids-validator ds004144/`
- [ ] Confirm 66 subjects present in `participants.tsv`
- [ ] Confirm `events.tsv` present for all task-fMRI runs (required for condition-level FC)
- [ ] Confirm clinical CSV subject IDs can be matched to BIDS sub-IDs
- [ ] Pull fMRIPrep Docker: `docker pull nipreps/fmriprep:latest`
- [ ] Fetch Schaefer atlas: `nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=200)`

---

## 8. KEY PRIOR PAPERS USING THIS DATASET (to read before writing)

1. Balducci et al. (2022) *Scientific Data* — original dataset descriptor
2. Garza-Villarreal et al. (2024) *Human Brain Mapping* — functional neurocircuitry in FM (likely used resting-state GLM)
3. Two other undisclosed papers using ds004144 — find via Google Scholar: `ds004144 OR "Balducci 2022 fibromyalgia"`

**Confirmed gap:** None of the 3 prior papers applied ML to task-fMRI. Task-fMRI analyzed only with GLM. No unsupervised subtype discovery in any modality.
