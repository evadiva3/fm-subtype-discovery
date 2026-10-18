# Independent validation added during revision

This study uses a **newly tuned spectrum**. The original spectrum artifact was found in this repository later. S17 reports separate corrected reruns using that recovered spectrum. S16 remains an independent validation; it cannot replace the historical experiment.

The three numerical modules were recovered from an earlier local working folder. Their SHA-256 hashes match the archived configuration for the 7,200-analysis angular comparison reported in FINAL Tables S24 and S25. The fresh run records its own dependency versions. These differ from the versions in the earlier environment.

## Run

Use Python 3.12 in an isolated environment. Install `requirements.txt`, then run:

```text
python run_validation.py --datasets 200 --workers 4 --draws 199
```

Each analysis is saved as soon as it finishes. The run resumes from existing records only when the configuration and source checksums match. To rerun from scratch, copy the scripts/modules to a new directory without the results folder. Existing results are not overwritten with a different configuration.

## Frozen design

- n = 28, ambient d = 119, two equal-probability components, leading-eigenvector separation coefficient 4.
- A separate 100-dataset pilot tunes one leading eigenvalue and a constant residual block to mean centered normalized participation ratio 4.5. The tuned residual variance is recorded in `results/config.json`.
- Independent calibration and evaluation streams: 200 calibration null datasets, 200 held-out null datasets and 200 mixture datasets. Evaluation-null and mixture datasets share base noise and rotations for paired comparisons; calibration datasets are separate.
- Each source dataset is evaluated under nested fixed projections to q = 5 and q = 10, plus an oracle q = 5 projection containing the planted covariance eigenvector. The oracle serves as a simulation diagnostic; it is not an available clinical procedure.
- Four original arms: W, D4, K, E. Each evaluation uses 199 reference draws, with 200 auxiliary directions for E. This gives 1,800 analyses of 600 source arrays, plus the independent pilot and 200 rotation comparisons.
- Both nominal p ≤ .05 and independently calibrated p cutoffs are reported. For each arm/projection, the calibrated cutoff is the largest observed calibration p cutoff that yields no more than 10 rejections among 200 calibration datasets. Ties are retained. The held-out null rejection rate is used to check the resulting cutoff.
- Calibration-cutoff uncertainty is not included in the conditional binomial intervals. These estimates are exploratory and do not establish exact 5% size or general validity.
- Every dataset is kept in the records. Numerical errors are retained and reported with unconditional rejection-rate bounds, and guard failures are recorded explicitly.

`results/records/` contains seeds, source hashes, p-values, fitting diagnostics, guard outcomes, retained signal length and projected squared separation. `results/rotation_audit.json` records the numerical invariance comparisons. `results/summary.json` contains all rates and exact pointwise binomial intervals.
