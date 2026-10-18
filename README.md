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
