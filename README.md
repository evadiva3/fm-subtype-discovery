# Cluster-reference calibration on normalized representations

This repository contains the code and evidence for Bangsil and Joshi, "Calibration and sensitivity of cluster tests on normalized neuroimaging representations". The study examines how Gaussian and angular cluster references behave after row normalization, how to calibrate them, and which kinds of structure they detect. The clinical cohorts illustrate the fitted pipelines; they do not establish disease subtypes.

The section and table numbers below refer to S1 File and S2 File of the submission. S2 File identifies the code and stored result for each table and figure. Section S30 of S1 File lists every folder in `results/`.

## Quick verification

Use Python 3.12 in an isolated environment, then run these commands from this directory:

```text
python -m pip install -r requirements/simulation-lock.txt
python reproduce.py verify
```

These commands recalculate the reported rates, check the stored records and source checksums, and run selected exact replays. They do not rerun every experiment. See [docs/REPRODUCING.md](docs/REPRODUCING.md) for fresh reruns, the S35 and S36 commands, the depression reproduction check (S21), tests and figures.

## Repository map

| Directory | Contents |
|---|---|
| `packaging/` | `nullcal` reference constructions and angular statistics, with package tests |
| `analysis/` | Simulation drivers, summaries and verifiers for the recorded studies |
| `experiments/ds002748_mdd/` | Depression pipeline and the exploratory screens E1 to E18 and phase 1 runs (`propagated_null/`) |
| `models/`, `src/`, `preprocessing/`, `config.py` | Encoder, training, clustering and preprocessing used to produce the stored embeddings |
| `results/` | Stored records for every study; see the register in Section S30 |
| `reproducibility/s16/` | Frozen independent validation (S16), with its code and 1,800 records |
| `reproducibility/audits/` | Reproduction checks, paired-count calculations and recovered historical records |
| `docs/` | Study protocols and the reproduction guide |
| `paper/figures/` | Figure scripts and the rendered figures |
| `archive/` | Superseded source and figure inputs that current code or S2 File depends on |
| `data/` | Applied inputs and saved embeddings; see `data/README.md` |
| `tests/` | Tests of the drivers and package |

Use a new output path for every fresh run so that stored results are not overwritten. Do not edit files whose SHA-256 is recorded in `results/` or `reproducibility/`; those records pin the files to their existing contents.

## License

The code is released under the MIT License; see [LICENSE](LICENSE). The OpenNeuro source datasets keep their own licenses.

## Disclosure
This project was conceived, designed and carried out by the authors, who set the research questions and hypotheses, wrote the core pipeline code, drafted the manuscript, and interpreted the results. Claude Opus 5.5 (Anthropic) and GPT-6 (OpenAI) supported the work by editing the writing for clarity, generating or substantially editing the files listed below, helping with debugging, auditing the code for errors, and organizing the repository. The authors reviewed all of this work and take full responsibility for the code, analyses and conclusions. The reported results can be reproduced with python reproduce.py verify.

Files generated or substantially edited with AI assistance:

Documentation
- README.md
- archive/README.md
- data/README.md
- packaging/README.md
- paper/figures/README.md
- reproducibility/s16/README.md
- results/overnight/RESULTS.md
- docs/REPRODUCING.md
- docs/S36_POSTHOC_OBSERVED_NOTE_20261009.md

Protocols
- docs/PROTOCOL_V8.md
- docs/PROTOCOL_V8_FOLLOWUP.md
- docs/PROTOCOL_V9.md
- docs/PROTOCOL_V9_APPLIED.md
- docs/PROTOCOL_V9_KERNEL_FOLLOWUP.md
- docs/FOLLOWUP_PROTOCOL_20260924.md
- docs/FOLLOWUP_PROTOCOL_S20_20260924.md
- docs/FOLLOWUP_PROTOCOL_S22_20260924.md
- docs/FOLLOWUP_PROTOCOL_S24_S26_20260925.md
- docs/FOLLOWUP_PROTOCOL_S29_20260925.md
- docs/FOLLOWUP_PROTOCOL_S31_S32_20260926.md
- docs/FOLLOWUP_PROTOCOL_S35_S36_20261008.md
- docs/FOLLOWUP_PROTOCOL_S36_AMENDMENT_20261009.md

Audit and check scripts
- analysis/audit_angular_gof_v9.py
- analysis/audit_balanced_v9.py
- analysis/audit_v9_claims.py
- analysis/audit_validation_v7.py
- analysis/seed_overlap_audit.py
- analysis/sign_rule_audit.py
- analysis/pooled_null_check.py
- analysis/cohort_observed_check.py
- analysis/cohort_restart_check.py
- analysis/cohort_restart_check_amendment.py
- reproducibility/audits/recheck_angular_records.py
- reproducibility/audits/recheck_counts.py

Analysis scripts
- analysis/mdd_geometry_calibrated.py
- analysis/shape_aware_minor_axis.py
- analysis/shape_aware_detection_floor.py
- analysis/shape_aware_at_article_geometry.py
- analysis/validate_cluster_nulls.py
- analysis/reproduce_mdd_from_timeseries.py
- analysis/sample_size_detection.py
- analysis/grid_recorded.py
- analysis/calibrated_cohort.py
- analysis/evaluate.py
- analysis/compare_mdd_retraining.py
- packaging/nullcal_validation.py
- reproducibility/s16/modules/nullcal_validation.py
- reproducibility/s16/run_validation.py
- experiments/ds002748_mdd/propagated_null/nullcal.py
- experiments/ds002748_mdd/propagated_null/ov_common.py
- experiments/ds002748_mdd/propagated_null/r1c_selective_inference.R
