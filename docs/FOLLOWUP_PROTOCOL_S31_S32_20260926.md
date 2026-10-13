# Recorded follow-up protocol: S31 and S32

Fixed on 26 September 2026, before any S31 or S32 record was generated. It was written after the S18 to S30 results were known, and after the exploratory screens behind main Tables 2 and 8, their Section S11 verification and the earlier generating-law runs in `results/ambient_control.json` were known. It is a prospective computational specification within this revision, not an external preregistration. No result-dependent stopping, cell selection or change of statistic is permitted. Numerical failures prevent final summaries. Root seed 20260926; execution pilots use root 2026092601 and are excluded from inference.

Both studies promote exploratory results to recorded studies. Every dataset seed is a hashed address of the root seed, the experiment address, the cell or block, the dataset index and a purpose code, so the data are independent of every earlier screen, which used seed bases of the form 6,000,000 + 1,000 i or the roots 20260916 and 20260917.

## S31: mechanism control

Question: in fresh data, does replacing the fitted normalized second moment by its exact population value move the reference away from nominal, while the actual generating law of the normalized rows stays near nominal?

Cells: main Table 8 cells 1 (exponential spectrum, n = 28), 3 (two-block spectrum, n = 28) and 6 (two-block spectrum, n = 180), all with d = 119. Each cell uses the spectrum parameter stored for it by the screen (`results/popcov_control.json` cells 1 and 3; `results/popcov_control_cells567.json` cell 6); nothing is retuned.

Data: for each dataset, Z is an n × d standard normal matrix, Q a random orthogonal basis, X = Z diag(√λ) Qᵀ, and the rows of X are L2-normalized. 400 datasets per cell. Experiment address 87.

Arms, each with 200 reference draws and p < 0.05; within a dataset all arms share the source data and the reference-draw seed:

- M fitted: Gaussian with the fitted mean and covariance of the normalized rows, draws not renormalized, scored at the observed k without the surrogate size guard;
- C fitted: the same fit, draws renormalized, guarded search over k = 2 to 6;
- M with H and C with H: as M and C, with the covariance replaced by the population second moment of the normalized rows, H = Q diag(c) Qᵀ, computed by quadrature, and the fitted mean retained;
- C generating law: mean zero and covariance Q diag(λ) Qᵀ, draws renormalized, guarded search.

An observed guard failure counts as a non-rejection and stays in the denominator.

Primary quantities: the rejection rate of each arm in each cell with a Wilson 95% interval, the paired change from C fitted to C with H, and the rate of the generating-law arm with a note of whether its Wilson interval contains 5%.

## S32: direction reversal

Question: does the concentrated-spectrum reversal, with C rejecting more often than M, hold in a fully recorded run?

Cell: the two-block spectrum at target participation ratio 4.5, n = 28, d = 119, the cell verified in Section S11, with its spectrum parameter taken from `results/overnight/e2_spectrum_shape.json` (cell `two_block_pr4.5`) and not retuned. 1,600 datasets in four seed blocks of 400 (the block index is part of the hashed address). M and C as in S31, 200 reference draws, p < 0.05. Every dataset whose M and C decisions disagree is retested on the same data with 2,000 new reference draws. Experiment address 88.

Primary quantities: the M and C rejection rates with Wilson intervals, the M-only and C-only counts with an exact two-sided McNemar test, and the fraction of discordant datasets that remain discordant at 2,000 draws. The retests describe only the initially discordant subset.

Relation to S11: S11 retains per-block counts and discordant records only. S32 is a new recorded run, not a relabeling of S11.

## Checks before any manuscript text uses these results

Stop and report if any of the following holds: the generating-law arm of S31 lies outside 3 to 7% in any cell; C with H does not reject more often than C fitted in cells 1 and 6; the S32 cell does not show C above M. At 400 datasets an exactly calibrated arm falls outside 3 to 7% in about 4% of cells, so whether each Wilson interval contains 5% is reported alongside.

## Scope

These are simulation studies of the stated Gaussian laws. They do not bound error rates on real representations.
