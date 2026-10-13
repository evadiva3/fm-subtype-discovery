# Recorded follow-up protocol: S35 and S36

Fixed on 8 October 2026, before any S35 or S36 production record was generated. These are prospective computational specifications, not external preregistrations. No result-dependent stopping, cell selection, change of strength or change of statistic is permitted. Numerical failures prevent final summaries. Execution pilots use separate roots and are excluded from inference.

## S35: calibrated sensitivity and non-Gaussian controls at the depression embedding geometry

Question: do the S24 and S29b results, which were evaluated at the fitted fibromyalgia geometry only, hold at a second fitted geometry?

Null law: the Gaussian fitted to the stored normalized 51 × 68 canonical depression embedding (`results/mdd/canonical_single/Embeddings.npy`), with rows normalized after drawing, using the same fitting code as S18 and S24.

S35a repeats the S24 design exactly: 1,000 calibration-null and 1,000 evaluation-null datasets; the first 250 evaluation datasets also carry paired two-group alternatives (means ±δ m1, sizes 26/25) and three-group alternatives (means δ(cos θ m1 + sin θ m2), θ = 0, 2π/3, 4π/3, sizes 17/17/17, recentered), with mj = √λj vj from the fitted covariance and δ ∈ {0.5, 1, 1.5, 2}. The strengths are those of S24 and are not changed after any pilot. M and C use the nullcal implementations of S24 (five k-means restarts, k = 2 to 6, size guard max(4, round(0.15n)) = 8, 499 reference draws); an observed guard failure counts as a non-rejection. Calibration, randomized tied boundaries, the 3 to 7% descriptive tolerance and 2,000 two-stage bootstrap resamples follow S19 and S24. Root seed 20261008, experiment address 90, bootstrap address 91.

S35b repeats S29b: 1,000 datasets from each of the two cluster-free coordinate laws (skewed e − 1 with e ~ Exp(1); heavy-tailed t4/√2) with the fitted depression mean and covariance factor, normalized, tested with M and C as in S35a and evaluated at the S35a calibrated cutoffs. Experiment address 92, bootstrap address 93.

Prespecified primary quantities: held-out size of M and C; calibrated detection of each alternative and the paired M − C difference; calibrated rejection of the two non-Gaussian laws. Interpretation fixed in advance: the S24 conclusion is replicated at this geometry if both held-out sizes meet the 3 to 7% tolerance and calibrated C is at least as sensitive as M for every alternative (no paired M − C interval lies wholly above zero). Any other pattern is reported as found. The S29b conclusion is replicated if the skewed law is rejected well above 5% by both calibrated constructions.

## S36: cohort reference p-values and k-means restarts

Question: the cohort p-values in Table S64 (20,000 draws) and the S24 recomputation for the observed fibromyalgia embedding differ. Table S64 was produced with `analysis/evaluate.py`, whose k-means restart count is set by `config.kmeansNInit` (currently 20), whereas S24 uses five restarts. S36 tests whether the restart count explains the difference.

For each cohort (trained fibromyalgia embedding with `data/outputs/K-Means-Labeling.csv`; canonical depression embedding with `results/mdd/canonical_single/K-Means-Labeling.csv`), recompute M, G, S and C with `cluster_evaluate.perm` exactly as `analysis/null_progression.py` calls it (20,000 draws, random_state from `config.randomSeed`), once with `kmeansNInit` = 20 and once with 5. No other setting changes.

Criterion: Table S64 is reproduced by a restart setting if every recomputed p-value lies within three Monte Carlo standard errors, √(p(1 − p)/20,000), of the stored value. Library versions differ from the historical runs, so bitwise agreement is not expected. The result is reported whichever setting, if any, reproduces the table.
