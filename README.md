# Cluster-reference calibration on normalized representations

Code and evidence for Bangsil and Joshi, "Calibration and sensitivity of cluster tests on normalized neuroimaging representations" (PLOS ONE submission). The study examines how Gaussian and angular cluster references behave after row normalization, how they are calibrated, and what structure they are sensitive to. The clinical cohorts illustrate the fitted pipelines; they do not establish disease subtypes.

Section and table numbers below refer to S1 File and S2 File of the submission. S2 File lists the code and stored result behind each table and figure, and Section S30 of S1 File is the register of every folder in `results/`.

## Quick verification

Use Python 3.12. Create an isolated environment, then run from this directory:

```text
python -m pip install -r requirements/simulation-lock.txt
python reproduce.py verify
```

This recalculates reported rates and checks stored records, source checksums and selected exact replays. It does not rerun every experiment. See [docs/REPRODUCING.md](docs/REPRODUCING.md) for fresh reruns, the S35 and S36 commands, the depression reproduction check (S21), tests and figures.

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

Stored results are not overwritten; fresh runs must use a new output path. Files whose SHA-256 is recorded in `results/` or `reproducibility/` are pinned by those records and must not be edited.

## License

Code is released under the MIT License; see [LICENSE](LICENSE). The OpenNeuro source datasets keep their own licenses.
