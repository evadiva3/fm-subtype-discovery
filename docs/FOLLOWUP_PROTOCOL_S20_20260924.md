# Recorded follow-up protocol: non-antipodal alternatives (S20)

Fixed on 24 September 2026, before any S20 records were generated. It is a prospective computational specification within this revision, not an external preregistration. It was motivated by the S19 result: after independent calibration, none of W, D4, K or E detected origin-symmetric two-component mixtures above its held-out null rate. Every S19 mixture is unchanged when all observations are negated, as is every centered angular central Gaussian reference, and Tyler whitening can absorb a ±v concentration into the fitted shape. S20 tests whether detection returns when that symmetry is broken, using paired alternatives. No result-dependent stopping, cell selection or change of statistic is permitted.

## Design

Use n = 28, d = 119 and the two S19 spectra: (1, 0.01, …, 0.01) and exp(−j/10), j = 0,…,118. Use the S19 fixed nested projection matrix exactly (its seed address is reused), with q = 5 and q = 10. For each spectrum, independently generate 1,000 calibration-null and 1,000 evaluation-null datasets. Every dataset has its own random orthogonal eigenbasis v1,…,v119 and a random permutation of the 28 rows.

Alternatives use the first 250 evaluation indices and share each index's noise, eigenbasis and permutation across all alternatives, strengths and projections. Let m1 = √λ1 v1 and m2 = √λ2 v2, so each is one standard deviation along its eigenvector. At strength δ ∈ {2, 4}, component means are δ(cos θk m1 + sin θk m2), and row labels are the permutation rank modulo the number of components (balanced sizes).

| Alternative | Angles θk | Components and sizes | Overall mean |
|---|---|---|---|
| antipodal | 0, π | 2; 14/14 | zero; symmetric under negation (S19-type) |
| tri | 0, 2π/3, 4π/3 | 3; 10/9/9 | zero; not symmetric under negation |
| pair | 0, π/2 | 2; 14/14 | nonzero |
| shift | — | 1; mean δ(m1 + m2)/2 | nonzero; unimodal control |

With 28 rows the tri split is 10/9/9, so its realized overall mean is δ/28 times the first component mean rather than exactly zero; the saved mean-shift diagnostic reports it. The shift control has the same overall mean as pair. Its rejection indicates sensitivity to a non-centered unimodal law rather than to discrete groups.

Use the frozen S16 W, D4, K and E implementations, 199 references and E auxiliary size 200. D4 is invariant to sign changes of individual rows, so it can respond only to axial structure; this is noted, not used to select methods. For every dataset, save the projected between-component Mahalanobis scatter tr((PᵀΣP)⁻¹B), where B is the size-weighted covariance of the projected component means, and the projected squared Mahalanobis length of the overall mean. Save all p-values, guard outcomes and seeds.

## Calibration, uncertainty and reporting

Calibration, randomized tied boundaries, held-out size, the prespecified descriptive tolerance of 0.03–0.07, conditional Wilson intervals and 2,000 two-stage bootstrap resamples follow the S19 protocol exactly, using distinct bootstrap address 75. Report every statistic, alternative, strength, spectrum and projection.

**Primary contrast (prespecified):** for each statistic, the paired calibrated-detection difference tri − antipodal at δ = 4, with two-stage bootstrap intervals. The symmetry explanation predicts a clearly positive difference for W, K and E. **Secondary contrasts:** tri − antipodal at δ = 2, and pair − shift at δ = 2 and 4. All other comparisons are descriptive. No universal ranking is claimed, and no multiplicity-adjusted winner is selected.

Numerical failures prevent final summaries. Root seed 20260924 with experiment address 74 defines the data, reference, bootstrap and boundary streams, distinct from S18 (71) and S19 (72). A small execution pilot uses a separate root and is excluded from inference.

## Scope

S20 is a simulation study of the tested laws. It cannot establish performance for other spectra, projections or clinical representations. Rejection of a centered angular law can reflect a mean shift without discrete classes; the shift control is included for that reason, and detection alone does not establish subtypes.
