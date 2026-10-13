# Recorded follow-up protocol: S24–S26

Fixed on 25 September 2026, before any S24–S26 records were generated. These are prospective computational specifications within this revision, not external preregistrations. No result-dependent stopping, cell selection or change of statistic is permitted. Numerical failures prevent final summaries. Each study uses root seed 20260925 and its own experiment address; execution pilots use separate roots and are excluded from inference.

## S24: sensitivity of M and C at calibrated size

Question: how sensitive are the historical construction M and the matched construction C once each is calibrated to 5%? The nominal-cutoff planted-group exercise cannot answer this, because C rejects 0.87% of null datasets at the embedding geometry.

Null law: the Gaussian fitted to the stored normalized 28 × 119 fibromyalgia embedding, exactly as in S18, with rows normalized after drawing. Generate 1,000 calibration-null and 1,000 evaluation-null datasets. The first 250 evaluation datasets also carry paired alternatives, built by adding component means before normalization. With λj and vj the eigenvalues and eigenvectors of the fitted covariance and mj = √λj vj:

- two-group: means ±δ m1, sizes 14/14;
- three-group: means δ(cos θ m1 + sin θ m2), θ = 0, 2π/3, 4π/3, sizes 10/9/9, recentered on the realized sizes;
- δ ∈ {0.5, 1, 1.5, 2}.

Amendment before production: the strengths were first set to δ ∈ {2, 4, 6}. A four-dataset execution pilot (separate root, excluded from inference) showed both constructions detecting the two-group alternative in every dataset already at δ = 2, so the planned strengths would have saturated. They were replaced by the four weaker strengths above before any production record was generated. No other design element was changed.

Row labels come from each dataset's random permutation and are shared across alternatives. M and C use the historical implementations (five k-means restarts, k = 2–6, guard as in S18) with 499 reference draws. An observed guard failure counts as a non-rejection (p = 1). Calibration, randomized tied boundaries, held-out size with the 0.03–0.07 descriptive tolerance, and 2,000 two-stage bootstrap resamples follow S19 (bootstrap address 79). The prespecified primary quantities are calibrated detection of M and C for each alternative, and their paired difference. Nominal (p ≤ 0.05) detection is also reported. Experiment address 78.

## S25: angular detection across sample size

Question: how does calibrated detection of centered cluster structure by W, D4, K and E change between n = 28 and n = 100 (S22)? S25 repeats the S22 design exactly at n = 50 and n = 200. For the four-group alternative, group sizes are n − 3⌊n/4⌋ and three groups of ⌊n/4⌋ (12/12/12/14 at n = 50; 50 each at n = 200); the other families use the S22 size rules. The S22 experiment address (76) with n in every hashed address gives new streams. Bootstrap address 81. The prespecified primary quantity is calibrated detection of each centered alternative as a function of n (28, 50, 100, 200), reported descriptively with separate intervals, because each n is a separate study.

## S26: fully recorded rerun of the conditioning × sample-size grid

Question: does the 30-cell grid of main Table 5 hold when every dataset outcome is recorded? Use the same 30 cells (target normalized participation ratio 2, 3, 4.5, 6, 9, 15 crossed with n = 28, 60, 119, 180, 240; d = 119), reusing each cell's tuned exponential-spectrum parameter τ as saved in `results/overnight/e1_grid.json` (no retuning), 400 datasets per cell, 200 reference draws, five restarts and strict p < 0.05 for M and C. Retain every dataset's seed, source hash, achieved participation ratio, observed statistic, selected k, guard outcome and p-values. Report rates with Wilson intervals, achieved participation ratio and M-only/C-only counts for every cell. Experiment address 80. The historical grid is retained, not pooled.

## Scope

These are simulation studies of the stated laws. S24 concerns the fitted embedding geometry and two planted families; S25 concerns the S22 laws; S26 concerns centered exponential spectra.
