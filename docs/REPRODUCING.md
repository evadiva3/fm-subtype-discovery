# Reproducing

Run everything from the repo root. Simulations use only synthetic data, so no imaging download is needed. Output folders must be new; drivers refuse to overwrite.

## Setup

Python 3.12:

```text
python -m pip install -r requirements/simulation-lock.txt
```

Other environments: `requirements/test-lock.txt` (tests), `requirements/figures.txt` (figures, plus Poppler), `requirements/applied.txt` (encoder and depression pipeline). Results match to floating-point tolerance, not bit for bit, across machines and library versions.

## Check stored results

```text
python reproduce.py verify
python reproduce.py test
```

`verify` recomputes the reported rates and intervals from the stored records and checks source checksums, with some exact replays. It covers the v9 angular studies, S9, S14 to S20, S22, S24 to S26, S29, S31, S32 and the consensus replay (S6.2). It does not rerun the simulations.

## Rerun studies

```text
python reproduce.py power --output results/my_power --workers 4
python reproduce.py geometry --output results/my_geometry --workers 4
python reproduce.py validation --output results/my_s16 --workers 4
python reproduce.py followup --output results/my_followup --workers 32
```

`power` is the corrected eigenvector studies, `geometry` is S9, `validation` is S16 from its frozen copy, and `followup` is S18 to S20, S22, S24 to S26, S29, S31 and S32 in order. `followup` takes hours. To run one study on its own, use the driver lines in `reproduce.py`, e.g.

```text
python analysis/headline_recorded.py --output results/my_s18 --workers 32
python analysis/verify_followup.py --directory results/my_followup --replay
```

S23 (post hoc held-out size) reads the S19, S20 and S22 records: `python analysis/pooled_null_check.py`.

S35 and S36:

```text
python analysis/mdd_geometry_calibrated.py --part a --output results/my_s35a --workers 32
python analysis/mdd_geometry_calibrated.py --part a --summarize --output results/my_s35a
python analysis/mdd_geometry_calibrated.py --part b --calibration results/my_s35a --output results/my_s35b --workers 32
python analysis/mdd_geometry_calibrated.py --part b --calibration results/my_s35a --summarize --output results/my_s35b
python analysis/cohort_restart_check.py --output results/my_s36 --workers 8
python analysis/cohort_restart_check_amendment.py --output results/my_s36_amendment --workers 8
python analysis/cohort_observed_check.py --output results/my_s36_observed
```

Consensus (S6.2): `python analysis/recheck_consensus.py --output results/my_consensus.json`.

## Depression reproduction (S21)

Needs `requirements/applied.txt` and the parcel time series in `../resting_state_dep_data/`:

```text
python analysis/reproduce_mdd_from_timeseries.py
set MDD_CTRL_ROOT=<new folder>
python experiments/ds002748_mdd/propagated_null/mdd_fixed_arch_control.py 0 1
python analysis/compare_mdd_retraining.py --fresh <new folder> --seed 0
```

The first line rebuilds the stored embedding from the canonical checkpoint. The last three retrain one seed and compare it with the saved run. The fMRIPrep derivatives and preprocessed fibromyalgia images were not kept, so preprocessing cannot be rerun from this repo. The R selective-inference comparison also needs R with `clusterpval`, `KmeansInference` and `jsonlite`.

## Figures

```text
python reproduce.py figures
```

See `paper/figures/README.md` for which file is which figure.

## Rules

Don't edit files whose SHA-256 is stored in `results/` or `reproducibility/` (protocols, drivers, inputs). Don't pool the historical headline run with S18.
