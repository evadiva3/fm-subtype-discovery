# nullcal

Tests whether a set of embeddings or other feature vectors, one row per subject, contains cluster structure. It reports a p-value that stays calibrated when the rows have no clusters, even after the rows are normalized to unit length.

```python
import nullcal
r=nullcal.test(X)    #X: one row per subject, any number of features
print(r.p)           #small p: the rows are not directions of one centered population
print(r)             #statistic, chosen k, sample size and mean resultant length
```

Install from the repository root with `python -m pip install ./packaging` (Python 3.10 or newer).

## What `test` does

It normalizes each row, projects the rows to `q = 10` dimensions with a fixed random projection, removes their shared shape by Tyler whitening, and takes the best k-means silhouette over k = 2 to 6 with the minimum cluster size max(4, round(0.15n)). It compares that value with the same statistic computed on `n_draws = 199` sets of uniformly random directions. This is test W in the paper.

The p-value is exact, up to ties, when the rows are independent and are the directions of one zero-mean elliptical population, such as a centered Gaussian, whatever its covariance. It needs more rows than projected dimensions (n > q). With `n_draws` draws the smallest possible p is 1/(n_draws + 1), so 199 draws allow p = 0.005.

## Reading the result

- **A small p is not proof of subtypes.** It means the rows are not directions of one centered elliptical population. A single population whose center is away from the origin also gives small p values; a large `r.mean_resultant` points that way.
- **A large p is not proof that there are no subtypes.** At n = 28 the paper's simulations found low detection of centered mixtures. Detection rose with sample size.
- **Some structures are invisible to it.** Two groups placed symmetrically about the origin, one at +v and one at -v, are absorbed by the whitening step and are not detected (S19, S20). The fixed projection keeps only part of a structure that lives in a few of many coordinates, so detection is lowest when d is large and the groups differ along few directions.
- **Report the settings** you used (`q`, `n_draws`, `seed`, `projection_seed`), so others can repeat the result.

The Gaussian references `null_test` and `ladder` (M, G, S and C in the paper) are kept so that the archived runs can be reproduced. They are not calibrated after normalization and should not be used to decide whether clusters exist.

## Modules

The drivers used in the paper can also load this local source directly. Use the pinned environment in `requirements/simulation-lock.txt` for the corrected runs.

| Module | Role |
|---|---|
| `nullcal` | `test` (the calibrated angular test W) and the historical Gaussian cluster references |
| `nullcal_validation` | Diagnostic benchmark and known-generating-law controls |
| `angular_null` | Fixed projections, Tyler shape, whitening and refitted silhouette reference W |
| `angular_gof` | W, fourth-moment D4, kernel K and bootstrap E comparison |
| `angular_fourth`, `angular_balanced` | Earlier fourth-moment and kernel follow-up studies |

The ideal ACG argument assumes independent centered directions and requires the Tyler existence conditions to hold, with n greater than the projected dimension. Numerical shape fits report failures explicitly. An angular test does not by itself calibrate shared representation training or establish clinical subtypes. The manuscript describes the finite-sample evidence and the limits of these claims.

The historical protocols in `docs/PROTOCOL_*` and the source hashes in the result configurations identify earlier runs. Do not reformat or otherwise change numerical files with recorded hashes just to make the formatting consistent. Run `python reproduce.py verify` to check the evidence and `python reproduce.py test` for the tests.
