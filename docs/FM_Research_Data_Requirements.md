# Data requirements

What to download to rebuild this study from scratch, and what the files actually contain.

## 1. Primary data sources

### 1.1 OpenNeuro ds004144 (imaging)

URL: https://openneuro.org/datasets/ds004144/versions/1.0.2
Access: no account, approval or DUA. Download it directly. Format is BIDS.

| Field | Value |
|---|---|
| FM patients | 33 (female) |
| Healthy controls | 33 (female) |
| Recruitment site | Mexico |
| Total subjects | 66 |

Modalities to download:

- `T1w`, structural MRI, all subjects
- `T2w`, structural MRI, all subjects
- `bold` resting-state fMRI, standard EPI
- `bold` task-fMRI, the emotion processing and regulation task (EPRT). 27.8 minutes,
  834 volumes per subject, TR 2 s.
- `events.tsv`, trial-level condition and timing labels. Required for condition-level FC
  extraction.
- `participants.tsv`, subject-level metadata in the BIDS root.

The task design is worth reading carefully, because it is not a full crossing and an earlier
version of this document described it wrongly. There are **seven** conditions, not nine. Three
valences (Neutral, Negativo, Happy) are crossed with four regulation strategies (OBSERVAR,
REDUCIR, SUPRIMIR, INCREMENTAR), but only seven of the twelve cells were actually run:

| Condition (as it appears in `events.tsv`) | Trials | Seconds each |
|---|---|---|
| Neutral - OBSERVAR | 12 | 8 |
| Negativo - OBSERVAR | 12 | 8 |
| Negativo - REDUCIR | 12 | 8 |
| Negativo - SUPRIMIR | 12 | 8 |
| Happy - OBSERVAR | 12 | 8 |
| Happy - SUPRIMIR | 12 | 8 |
| Happy - INCREMENTAR | 12 | 8 |

Neutral only ever appears with OBSERVAR, and Happy is never paired with REDUCIR. So you get
**seven FC matrices per subject**, one per condition, each built from 12 trials × 4 volumes =
48 volumes concatenated. The trial-type strings are in Spanish and the code matches them
literally, so do not translate them.

Field maps are not needed. Grab them only if fMRIPrep fails without them.

---

### 1.2 Zenodo clinical and behavioural data
- DOI: https://doi.org/10.5281/zenodo.6554870
- Access: Free download, no approval required

Required variables, all of them:

| Variable | Instrument | Used For |
|---|---|---|
| Pain intensity | Visual Analogue Scale (VAS) | Primary clinical outcome |
| Depression | Beck Depression Inventory (BDI) | Clinical validation |
| Anxiety | Beck Anxiety Inventory (BAI) | Clinical validation |
| Alexithymia | Toronto Alexithymia Scale (TAS-20) | Clinical validation |
| Emotion regulation | Emotion Regulation Questionnaire (ERQ), subscales Cognitive Reappraisal, Expressive Suppression | Clinical validation, maps to task conditions |
| Age | Sociodemographic | Covariate control |
| Education (years) | Sociodemographic | Covariate control |
| Disease duration | Sociodemographic | Clinical validation |
| Number of medications | Sociodemographic | Clinical validation |
| Subject ID | n/a | Merge key with imaging data |
| Group label (FM / HC) | n/a | Permutation testing, baseline SVM |

Merge strategy: join Zenodo CSV to participants.tsv on subject ID. Verify ID format alignment (sub-XXXX in BIDS against numeric IDs in Zenodo, so you may need a crosswalk).

---

## 2. Derived data (produced by the pipeline, not downloaded)

These are computed from the raw data. Nothing here needs downloading.

### 2.1 fMRIPrep outputs (per subject)
- `*_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz`, preprocessed BOLD
- `*_confounds_timeseries.tsv`, motion and physiological confounds
- `*_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz`, brain mask
- `*_T1w_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz`, normalized T1

What fMRIPrep applies:
- Motion correction (6-parameter rigid body)
- Slice timing correction
- Spatial normalization to MNI152 (2mm³ resolution)
- Confound regression (24 motion params + WM/CSF signals minimum)

Expected dropout is 10–15% from excessive motion (>0.5mm FD threshold). Target: ≥28 FM subjects post-QC.

---

### 2.2 Atlas
- Schaefer 200-parcel, 7-network parcellation, fetched via `nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=200)`
- Produces 200 ROI time series per subject per condition block

---

### 2.3 Functional connectivity matrices

- Shape: `200 × 200`, Ledoit-Wolf shrunk correlation, Fisher z-transformed with the diagonal
  zeroed.
- One matrix per condition, so seven per subject. See the condition table in 1.1.
- Optionally one resting-state matrix per subject as well, for the ablation.
- That comes to 66 subjects × 8 matrices = 528 in total if you build the resting-state ones,
  462 without.
- At float32 each matrix is 200 × 200 × 4 bytes, about 160 KB, so budget roughly 85 MB.

Ledoit-Wolf rather than a plain Pearson correlation is not optional here. Each condition gives
48 volumes against 200 parcels, so the sample covariance is rank-deficient and a raw
correlation matrix is unstable.

---

### 2.4 Graph dataset (PyTorch Geometric)
Per subject graph node features (200 nodes × 5 features):

| Feature | Description |
|---|---|
| Mean BOLD signal | Per parcel, per condition block |
| BOLD variance | Per parcel, per condition block |
| Spectral power band 1 | 0.01–0.04 Hz |
| Spectral power band 2 | 0.04–0.1 Hz |
| Spectral power band 3 | 0.1–0.25 Hz |

Edge weights: pairwise FC correlation values (200×200 adjacency).

---

### 2.5 Model outputs

Shapes below are what the canonical run actually produces, with dModel 119 from the corrected
architecture search. Earlier drafts of this file quoted 64 and nine conditions; both were wrong.

- Per-condition graph embeddings: `(7, 119)` per subject, straight out of the GATv2 encoder.
- Pooled subject embeddings: `(28, 119)` for the FM patients, after the condition-attention
  pool. Saved as `data/outputs/Embeddings.npy`.
- Attention weights: `(28, 7)`, one softmax weight per condition per subject. These are the
  pooling weights, not per-parcel node attention.
- Cluster assignments: `(28,)`, k-means labels. k is searched over 2–6 and selected by
  silhouette under a minimum-cluster-size guard.
- Silhouette: one scalar per k.
- Null distribution: 1,000 draws by default. Read `analysis/evaluate.py` before changing how
  these are generated; the construction is the subject of the paper.

---

## 3. Data files to create / maintain locally

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

## 4. Clinical data columns (subjects_master.csv)

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

## 5. What this study does not need

- Paid datasets (UK Biobank, ABCD): not used
- Additional data collection: no participants to recruit
- External validation cohort: a year-two goal only
- DWI/DTI: not in ds004144, not needed
- EEG/MEG: not collected
- Pharmacological trial data: not applicable

---

## 6. Analysis inputs by pipeline stage

| Stage | Input Data | Output |
|---|---|---|
| fMRIPrep | Raw BIDS (T1w + task BOLD + events) | Preprocessed BOLD + confounds |
| ROI extraction | Preprocessed BOLD + Schaefer atlas | 200 × T timeseries per condition |
| FC computation | ROI timeseries + events.tsv (condition blocking) | 9 × 200×200 FC matrices per subject |
| Graph dataset | FC matrices + node features | PyG Data objects |
| GAT training | Graph dataset (FM+HC or FM only, decide) | Trained encoder weights |
| Embedding extraction | Trained encoder + all subject graphs | 64-dim embeddings |
| Attention pooling | Per-condition embeddings | 1 pooled embedding per subject |
| Clustering | FM-only pooled embeddings | Cluster labels |
| Clinical validation | Cluster labels + subjects_master.csv | Mann-Whitney U results, effect sizes |
| Interpretability | GAT attention weights + Schaefer atlas | Brain maps per subtype |
| Ablations | Resting-state FC, raw FC matrices, PCA outputs | Baseline silhouette/effect sizes |

---

## 7. Quick download checklist

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

1. Balducci et al. (2022), *Scientific Data*. The original dataset descriptor.
2. Garza-Villarreal et al. (2024), *Human Brain Mapping*. Functional neurocircuitry in FM, probably a resting-state GLM.
3. Two other undisclosed papers use ds004144. Find them via Google Scholar: `ds004144 OR "Balducci 2022 fibromyalgia"`

Confirmed gap: None of the 3 prior papers applied ML to task-fMRI. Task-fMRI analyzed only with GLM. No unsupervised subtype discovery in any modality.
