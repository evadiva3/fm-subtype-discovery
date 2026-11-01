# Cluster-reference calibration on normalized representations

This repository contains the code and evidence for Bangsil and Joshi, "Calibration and sensitivity of cluster tests on normalized neuroimaging representations" (PLOS ONE submission). The study examines how Gaussian and angular cluster references behave after row normalization, how to calibrate them, and which kinds of structure they detect. The clinical cohorts illustrate the fitted pipelines; they do not establish disease subtypes.

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
| `experiments/ds002748_mdd/` | Depression pipeline, the exploratory screens E1 to E18 and the headline recalibration runs (`propagated_null/`) |
| `models/`, `src/`, `preprocessing/`, `config.py` | Encoder, training, clustering and preprocessing used to produce the stored embeddings |
| `results/` | Stored records for every study; see `results/README.md` and the register in Section S30 |
| `reproducibility/s16/` | Frozen independent validation (S16), with its code and 1,800 records |
| `reproducibility/audits/` | Reproduction checks, paired-count calculations and recovered historical records |
| `docs/` | Study protocols and the reproduction guide |
| `paper/build/`, `paper/figures/` | Figure scripts and the rendered figures |
| `archive/` | Original drivers behind the stored records, superseded source and figure inputs |
| `data/` | Applied inputs and saved embeddings; see `data/README.md` |
| `tests/` | Tests of the drivers and package |

Use a new output path for every fresh run so that stored results are not overwritten. Do not edit files whose SHA-256 is recorded in `results/` or `reproducibility/`; those records pin the files to their existing contents. When the results folders were renamed, the drivers that name those folders got the new paths, and their exact originals were kept in `archive/recorded_source/`. Verify checks the stored checksums against those copies.

## License

The code is released under the MIT License; see [LICENSE](LICENSE). The OpenNeuro source datasets keep their own licenses.

## Use of AI

The authors set the research questions, hypotheses and study design, wrote the core pipeline code (`models/`, `src/`, `preprocessing/`, `config.py` and the depression pipeline scripts in `experiments/ds002748_mdd/`), drafted the manuscript and interpreted the results. Generative AI tools (Claude and Claude Code, Anthropic; Codex, OpenAI) were used to:

- edit the writing for clarity, and draft this README and the other .md files
- draft or edit the simulation, verification and figure code and the tests, which the authors then edited and reviewed
- run some of the simulation studies
- help with debugging
- repo organization and file naming/headings
- audit the code for errors several times
- help organize the repository and the stored results

The authors reviewed every analysis choice and all AI-assisted code and text. The ideas, hypotheses and conclusions are the authors' own, and the authors take full responsibility for the code, analyses and conclusions. The reported results can be checked with `python reproduce.py verify`.

Eva Bangsil wrote the original manuscript draft and the initial versions of the student-authored scripts. The list below is a selected inventory; AI also assisted with small revisions to student-authored files not individually listed.

Selected files generated or substantially edited with AI assistance:

Documentation
- README.md
- archive/README.md
- data/README.md
- packaging/README.md
- paper/figures/README.md
- reproducibility/s16/README.md
- results/exploratory/screens_e1_e18/RESULTS.md
- docs/REPRODUCING.md
- docs/S36_POSTHOC_OBSERVED_NOTE_20261009.md
- results/README.md

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
- analysis/verify_corrected_power.py
- analysis/verify_geometry_surface.py
- analysis/verify_repository.py
- reproducibility/s16/verify_results.py
- reproduce.py

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
- analysis/analyze_saved_gof_v9.py
- analysis/geometry_surface.py
- analysis/scope_detection.py
- analysis/validate_angular_fourth_v8.py
- analysis/validate_angular_gof_v9.py
- analysis/validate_angular_null.py
- analysis/validate_balanced_v9.py
- packaging/angular_fourth.py
- packaging/angular_gof.py
- packaging/angular_null.py
- reproducibility/s16/modules/angular_gof.py
- reproducibility/s16/modules/angular_null.py

Experiment scripts
- experiments/ds002748_mdd/run_mdd_pipeline.sh

Tests
- tests/conftest.py
- tests/test_angular_fourth.py
- tests/test_angular_gof.py
- tests/test_asymmetric_detection.py
- tests/test_execution_regressions.py

Archived copies
- archive/recorded_source/ (exact copies of the drivers that wrote the stored records)
