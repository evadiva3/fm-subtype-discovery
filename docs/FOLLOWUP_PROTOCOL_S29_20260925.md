# Recorded follow-up protocol: S29

Fixed on 25 September 2026, after the S18–S28 results were known and before any S29 record was generated. This is a prospective computational specification within this revision, not an external preregistration. No result-dependent stopping, cell selection or change of statistic is permitted. Numerical failures prevent final summaries. Root seed 20260925; execution pilots use root 2026092505 and are excluded from inference.

Question: do the conclusions of S19–S25 and S24 depend on the two features every earlier recorded study shares, namely Gaussian generating laws and group means placed on leading covariance eigenvectors?

## S29a: angular statistics beyond Gaussian eigenvector alternatives

Design as S22 (n = 28 and n = 100, d = 119, the two S19 spectra, the fixed nested S19 projection to q = 5 and 10, W, D4, K and E with 199 references and E auxiliary size 200), with experiment address 82 and bootstrap address 83. For each spectrum, generate 1,000 Gaussian calibration-null and 1,000 Gaussian evaluation-null datasets exactly as in S22. The first 250 evaluation datasets also carry two paired centered alternatives at δ = 4, added before normalization and recentered on the realized group sizes:

- random-direction three-group: means δ(cos θ a1 + sin θ a2), θ = 0, 2π/3, 4π/3, where u1, u2 are orthonormal directions drawn uniformly for each dataset from a separate stream and ak = (ukᵀΣuk)^½ uk, so δ is in standard deviations along each direction; sizes as the S22 balanced three-group family;
- six-group: means δ(cos θ m1 + sin θ m2), θ = jπ/3, j = 0,…,5, on the leading axes mj = √λj vj; sizes n − 5⌊n/6⌋ and five groups of ⌊n/6⌋.

Two non-Gaussian laws without groups, each with mean zero and covariance Σ, use their own data streams, 1,000 datasets each:

- skewed: independent coordinates e − 1, e ~ Exp(1), along the covariance eigenvectors;
- heavy-tailed: independent coordinates t/√2, t ~ Student t with 4 degrees of freedom, along the covariance eigenvectors.

Both are unimodal, centered and not elliptical, so their normalized rows do not follow the angular central Gaussian law that W, D4, K and E reference. Every cell is evaluated with the cutoffs calibrated on the Gaussian calibration sample. The prespecified primary quantities are calibrated detection of each centered alternative and the calibrated rejection rate of each non-Gaussian law (a false-positive rate under a misspecified but cluster-free law). Intervals as in S19.

## S29b: M and C calibration under non-Gaussian laws at the embedding geometry

Null geometry as S24: the mean and square-root covariance factor of the Gaussian fitted to the stored normalized fibromyalgia embedding. Generate 1,000 datasets from each of two laws with that mean and covariance, x = μ + zA, where A is the fitted factor (rows along the fitted principal axes) and z has independent coordinates, either the skewed or the heavy-tailed coordinates defined above; rows are normalized after drawing. M and C use the S24 implementations with 499 references, five restarts, the S18 guard and p = 1 for an observed guard failure. Experiment address 84. Each dataset is evaluated with the S24 cutoffs, estimated from the recorded S24 calibration sample, and a fresh boundary uniform. The prespecified primary quantity is the calibrated rejection rate of M and C under each law, with a conditional Wilson interval and a two-stage bootstrap interval (2,000 resamples of the S24 calibration and the S29b evaluation indices, bootstrap address 85). Nominal (p ≤ 0.05) rates are also reported.

## Scope

These are simulation studies of the stated laws. They do not bound error rates on real representations, whose law is unknown.
