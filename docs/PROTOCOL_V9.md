# V9 follow-up protocol: power, dimension and established-method comparison

Written 15 September 2026, before running or viewing the v9 primary results.
This local plan follows the v8 weak-power findings and is not a public preregistration.
All earlier results remain unchanged. No commit, push or external release is authorized.

## Questions and population grid

Separate three possible limitations: projected dimension, sample size and departure
family. Generate independent d=20 source datasets at n=60 and n=240, with 200 datasets
per source cell. Analyze every source dataset at q=5,10,20. For q=20 use the identity;
the full dimension is identifiable since n>d. For q=5/10 retain the exact fixed v8
mixture projections, independent of all observations and shared across populations.

Six populations, each with independent rows, are fixed here:

1. `gaussian`: standard zero-mean Gaussian (ACG null after normalization).
2. `anisotropic`: zero-mean Gaussian with diagonal eigenvalues exp(-j/3), j=0..19 (ACG null).
3. `two_modes`: standard Gaussian noise plus equiprobable +/-4 e1 shifts (non-ACG alternative).
4. `six_modes`: standard Gaussian noise plus a uniformly chosen shift from +/-4 e1, +/-4 e2, +/-4 e3 (non-ACG alternative).
5. `offset`: standard Gaussian noise plus e1 (location alternative, not a subtype claim).
6. `shape_mixture`: equal mixture of zero-mean Gaussians with covariance diag(16,1,...,1) and diag(1,16,1,...,1) (non-ACG shape mixture).

Normalize rows before projection and renormalize afterward. No clustering labels are
used by any test. The source sample count is 6 x 2 x 200 = 2,400 unique new datasets;
three projections give 7,200 primary analyses. The earlier v7/v8 union is 4,000 unique
datasets, so the combined count becomes 6,400, subject to hash audit.

## Comparators

For every analysis use B=199 reference draws, plus-one upper-tail p-values including
ties, and rejection at p<=.05. Use independent reference streams per source dataset
and projection. Within a dataset, refitted-pivot constructions share uniform samples.

- W: the unchanged guarded maximum silhouette of Tyler-whitened, normalized directions,
  k=2..6, n_init=5 and clustering seed 0. Retain guard failures at p=1.
- D4: fourth-moment discrepancy, mean pairwise inner product to the fourth power minus
  3/[q(q+2)], calibrated by refitting Tyler in every isotropic reference sample.
- K: a fixed Gaussian-kernel discrepancy from uniformity on the whitened sphere,
  mean exp(-||w_i-w_j||^2). The omitted uniform cross and self expectations are
  constants on the unit sphere and do not affect ranks. Fix gamma=1; do not tune it.
  Calibrate by fully refitting isotropic references. This is an invariant adaptation,
  not an exact implementation of the published two-sample test.
- E: implement Ebner, Henze and Meintanis (2024), equation 22 with xi=2, gamma=1,
  auxiliary sample size m=200, and the composite parametric bootstrap in section 5.
  Generate observed Y from the Tyler-fitted ACG; for every bootstrap draw generate
  X* from the observed fit, refit Tyler on X*, generate Y* from that refit, and compute
  the raw-direction two-sample kernel statistic. Use B=199 plus-one p-values as our
  declared finite-simulation convention. This is not their exact published n=50,d=3
  simulation setting, and no finite-sample nuisance-free guarantee is claimed for E.

The K/D4/W invariance argument applies only to the exact unregularized estimator and
rotation-invariant statistic under independent ACG rows. E has a distinct reference
construction. Paired differences refer to the same observations, with independent
randomness for E's bootstrap. Do not label distinct-target tests as competitors.

## Reporting, multiplicity and audits

Report every cell and method, exact pointwise binomial 95% intervals, paired
discordances, fit/guard failures, and projected signal retention. Treat all comparisons
as descriptive unless specified otherwise. Combining the three fixed dimensions for
one test uses min(1,3 min(p5,p10,p20)); it does not adjust selection across tests or
training runs. Report the combined result, with no independence assumption for q.

Record full code/protocol hashes, environment, data/projection/reference addresses,
input hashes, numerical residuals and elapsed time. A configuration mismatch must fail
rather than mix results. Retain explicit failure records; do not silently drop failed
fits. Audit all hashes and summaries and fully replay one complete record per cell.
Unit checks verify the literal E kernel formula, full projection identity, invariance,
known uniform expectation of K, and deterministic finite-simulation decisions.

Reference: Ebner B, Henze N, Meintanis S. Statistical Papers 65, 3447-3475 (2024).
https://doi.org/10.1007/s00362-024-01529-1 . Use established methods with attribution;
do not claim a new kernel test or new Tyler theory.
