# Recorded follow-up protocol: scope extension (S22)

Fixed on 24 September 2026, before any S22 records were generated. It is a prospective computational specification within this revision, not an external preregistration. It was motivated by the S20 result: after independent calibration, none of W, D4, K or E detected centered cluster structure (antipodal or three-group), while K and E detected a displaced center. S20 used only balanced groups, at most three groups and n = 28. S22 asks whether that conclusion holds with unequal group sizes, four groups and a larger sample. No result-dependent stopping, cell selection or change of statistic is permitted.

## Design

Use d = 119, the two S19/S20 spectra, the S19 fixed nested projection with q = 5 and q = 10, and two sample sizes, n = 28 and n = 100. Each sample size is a separate study with its own calibration. For each spectrum and n, independently generate 1,000 calibration-null and 1,000 evaluation-null datasets, each with its own random orthogonal eigenbasis and row permutation.

The first 250 evaluation datasets carry five paired alternatives at strength δ = 4, built from m_j = √λ_j v_j (one standard deviation along eigenvector j). Row labels are assigned by the dataset's permutation rank. For every centered family, the component means are recentered using the realized group sizes, so the sample mixture mean is exactly zero:

| Alternative | Components | Group sizes (n = 28; n = 100) | Construction before recentering |
|---|---|---|---|
| tri | 3 | 10/9/9; 34/33/33 | δ(cos θ m1 + sin θ m2), θ = 0, 2π/3, 4π/3 |
| triuneq | 3 | 14/7/7; 50/25/25 | as tri, with half the rows in the first group |
| pairuneq | 2 | 21/7; 75/25 | +δ m1 and −δ m1 with a 3:1 split (centered, but not symmetric under negation) |
| tetra | 4 | 7/7/7/7; 25/25/25/25 | δ(±m1 ± m2 ± m3)/√3 at the vertices of a regular tetrahedron |
| shift | 1 | all rows | one Gaussian with mean δ(m1 + m2)/2 (unimodal, non-centered control, as in S20) |

Use the frozen S16 W, D4, K and E implementations, 199 references and E auxiliary size 200. Save projected between-group Mahalanobis scatter and projected squared mean shift for every dataset, plus all p-values, guard outcomes and seeds.

## Calibration, uncertainty and reporting

Calibration, randomized tied boundaries, held-out size, the 0.03–0.07 tolerance, conditional Wilson intervals and 2,000 two-stage bootstrap resamples follow S19/S20, using bootstrap address 77. Report every statistic, alternative, spectrum, projection and sample size.

**Primary quantities (prespecified):** calibrated detection of each centered alternative (tri, triuneq, pairuneq, tetra) at n = 100, and the paired contrast shift minus each centered alternative within each n. The S20 conclusion predicts low detection of centered alternatives and positive shift-minus-centered contrasts for K and E. Differences between n = 28 and n = 100 are reported descriptively, with separate intervals, because the two studies use different datasets. All other comparisons are descriptive. No universal ranking is claimed.

Numerical failures prevent final summaries. Root seed 20260924 and experiment address 76, with n in every hashed address, define the data, reference, bootstrap and boundary streams, distinct from S18 (71), S19 (72) and S20 (74). An execution pilot uses a separate root and is excluded from inference.

## Scope

S22 is a simulation study of the tested laws. It cannot establish behavior for other spectra, projections, strengths or clinical representations.
