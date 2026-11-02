# Results

Every stored study, grouped by role. Table S54 in S1 File lists each folder and where it is cited.

| Folder | Contents |
|---|---|
| `recorded/s6_s9/` | One-dimensional sweeps, corrected eigenvector studies and the consensus correction (S6, S9) |
| `recorded/s18_s22/` | Recorded studies S18 to S20 and S22 |
| `recorded/s24_s29/` | Recorded studies S24 to S26 and S29 |
| `recorded/s31_s32/` | Recorded studies S31 and S32 |
| `recorded/s35_s36/` | Recorded study S35 and the S36 recomputation |
| `exploratory/screens_e1_e18/` | Exploratory Gaussian-null screens E1 to E18 |
| `exploratory/headline_recalibration/` | Historical headline recalibration and geometry checks (R1) |
| `exploratory/mechanism_runs/` | Mechanism runs on reducing the re-projection shift (R2), not reported |
| `exploratory/controls/` | Covariance, ambient, restart, paired verification and other single-file controls |
| `angular/` | v8 and v9 angular studies and the angular analyses of the saved representations |
| `validation/` | Guarded fixed-count and generating-law validation (v7) and its resolution rerun |
| `historical/` | Historical paired calibration, propagated null and decoding (`handoff/`), and the calibration re-execution |
| `cohorts/` | Depression cohort runs (`mdd/`) and fibromyalgia controls and manifests (`fm/`) |

## Old folder names

The protocols in `docs/` and the stored configs were written before the folders were renamed, so they use the old names. `analysis/moves.py` maps them.

| Old | Now |
|---|---|
| `revision_20260923` | `recorded/s6_s9` |
| `revision_20260924` | `recorded/s18_s22` |
| `revision_20260925` | `recorded/s24_s29` |
| `revision_20260926` | `recorded/s31_s32` |
| `revision_20261008` | `recorded/s35_s36` |
| `overnight` | `exploratory/screens_e1_e18` |
| `phase1` | `exploratory/headline_recalibration` |
| `phase2` | `exploratory/mechanism_runs` |
| single files in `results/` | `exploratory/controls/`, `angular/` (`audit_v9_claims.json`) or `cohorts/fm/` (the three cohort csv files) |
| `angular_v8`, `angular_v8_fourth`, `angular_v8_restarts` | `angular/v8`, `angular/v8_fourth`, `angular/v8_restarts` |
| `angular_v9`, `angular_v9_balanced` | `angular/v9`, `angular/v9_balanced` |
| `embeddings_v8`, `embeddings_v9` | `angular/embeddings_v8`, `angular/embeddings_v9` |
| `validation_v7`, `validation_v7_resolution` | `validation/v7`, `validation/v7_resolution` |
| `handoff_20260802` | `historical/handoff` |
| `calibration` | `historical/calibration` |
| `mdd` | `cohorts/mdd` |
| `fm_fixed_arch_control` | `cohorts/fm/fixed_arch_control` |
