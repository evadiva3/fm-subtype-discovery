# Type I error characterisation of covariance-preserving permutation nulls

**Run summary.** All twelve planned experiments completed (E1-E12), plus two unplanned verification experiments forced by an invariant breach, plus both figures. E12 could not run as specified, the Fisher-z connectivity features it requires are not on disk, so a clearly labelled reduced substitute ran in its place; E11's fibromyalgia arm was skipped for the same reason (no per-run label vectors stored). Total wall clock 12 h 48 m, of which 10 h 35 m was machine suspension across three sleep events; **actual compute was 2 h 13 m** on 47 cores of a 48-core machine. Every cell used 400 datasets rather than the specified 120/60/40, because the timing probe showed the brief's cost model overestimated the n = 240 penalty by ~25×, and the extra datasets tighten the corrected arm's intervals where they are needed most. **The single most important result is that the paper's one-directional-discordance claim is false in general**, it holds for exponentially decaying spectra and reverses on spectra with a single dominant eigendirection, verified across four seeds and at 2,000 draws.

**Reading order.** The section "Claims in PAPER_v5.md that these results change" is the one to read first if you read only one. Raw per-cell numbers are in `e*.json` and the two long-format CSVs; `_tables.md` holds the complete generated tables including every discordant dataset; `RUN_LOG.md` is the chronological record.

**Provenance note.** `null_cal.py` and `geometry_surface.py` were 0 bytes on disk at session start, with no recoverable copy anywhere on the machine. They were restored from source supplied by the operator and validated against four published cells before any new work ran: M and measured PR reproduced exactly in 4 of 4, C exactly in 2 of 4 and within overlapping Wilson intervals in the other 2. No `nullcal` semantics were modified at any point. Details in `RUN_LOG.md`.

**Figures.** `figures/fig_e1_type1_surface.png`: Type I error over (PR × n/d) for M and C, with the nominal 0.05 contour and the fibromyalgia geometry marked. `figures/fig_e4_k_family.png`: Type I error against effective and nominal |K|.

---

## E1: Type I error over (PR × n/d)

**Question.** Where in conditioning-by-sample-size space is each construction anti-conservative, calibrated, or conservative?

**Design.** d = 119 fixed; target sphere-PR ∈ {2, 3, 4.5, 6, 9, 15} crossed with n ∈ {28, 60, 119, 180, 240}, 30 cells. Every dataset is a single unimodal Gaussian with exponentially decaying eigenvalues in a random orthonormal basis, so every rejection is a false positive; τ was tuned per cell by bisection on 20 throwaway samples so measured PR lands within 5% of target, and all 30 cells met that tolerance. 400 datasets × 200 draws per cell, M and C paired on identical data and identical draw seeds.

Realized rejection rate at α = 0.05, M / C:

| target PR | n=28 (n/d 0.24) | n=60 (0.50) | n=119 (1.00) | n=180 (1.51) | n=240 (2.02) |
|---|---|---|---|---|---|
| 2.0 | 0.985 / 0.627 | 1.000 / 0.993 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| 3.0 | 0.525 / 0.138 | 0.998 / 0.713 | 1.000 / 0.995 | 1.000 / 1.000 | 1.000 / 1.000 |
| 4.5 | 0.120 / 0.013 | 0.522 / 0.172 | 0.993 / 0.795 | 1.000 / 0.970 | 1.000 / 0.998 |
| 6.0 | 0.025 / 0.000 | 0.205 / 0.022 | 0.720 / 0.338 | 0.973 / 0.772 | 1.000 / 0.922 |
| 9.0 | 0.000 / 0.000 | 0.062 / 0.000 | 0.150 / 0.020 | 0.403 / 0.190 | 0.632 / 0.305 |
| 15.0 | 0.000 / 0.000 | 0.000 / 0.000 | 0.015 / 0.000 | 0.013 / 0.003 | 0.028 / 0.000 |

Full per-cell detail, measured PR and sd, τ, event counts, Wilson intervals, mean p, KS D, null means, and both discordance counts, is in `e1_grid.csv` and `e1_grid.json`.

**Interpretation.** The surface is monotone in both axes and the safe region is small. C is at or below the nominal 0.05 in 9 of 30 cells, and every one of those sits at PR ≥ 6 or n ≤ 60. Along the n axis at fixed PR the deterioration is severe: at PR 4.5, C moves from 0.013 at n=28 to 0.795 at n=119 and 0.998 at n=240. This confirms and extends the existing reading that C's good behaviour at n=28 is rank deficiency in the covariance estimate rather than the correction working, but it also shows the same collapse along the PR axis at fixed n, which the earlier one-dimensional sweeps could not separate. At n=119 (n/d = 1.0) C is anti-conservative at every PR up to 9.0 inclusive. The real fibromyalgia geometry (PR 4.37, n/d 0.24) sits in the one corner where C is nearly nominal, at 0.013 with 5 events in 400; that estimate is fragile and is listed in the low-event audit below. The raw edge-feature geometry (PR 20.5) lies above the top of this grid, where both constructions return 0.000. Nothing here is well calibrated in the sense of uniform p-values: the smallest KS D against U(0,1) anywhere in the grid is far from zero, and at high PR both arms return mean p near 1 with KS D near 1, i.e. conservative to the point of being uninformative rather than correct.

---

## E1 verification: the C_only anomaly

**Question.** Three E1 cells returned `C_only > 0`, against a published 0 across 2,560 structureless datasets. Is that real?

**Design.** Each flagged cell was re-run at four independent seeds (400 datasets each, the first reproducing the original cell exactly), and every discordant dataset, one with p_C < 0.05 ≤ p_M, was re-tested at 2,000 draws instead of 200.

| cell | C_only by seed | pooled C_only | rate | persists at 2000 draws |
|---|---|---|---|---|
| PR 3.0, n=28 | 2, 2, 2, 2 | 8 / 1600 | 0.0050 | **0 of 8** |
| PR 4.5, n=28 | 2, 1, 1, 1 | 5 / 1600 | 0.0031 | **0 of 5** |
| PR 9.0, n=119 | 1, 2, 2, 2 | 7 / 1600 | 0.0044 | **0 of 7** |

**Interpretation.** The events reproduce at every seed, so they are not a seed fluke, but not one of the twenty survives a tenfold increase in draw resolution. At 200 draws the estimator (c+1)/201 has granularity 0.005, so a dataset whose two p-values straddle α = 0.05 can invert on discretisation alone; the paired p-values in `e1_conly_check.json` show exactly that, sitting within one or two grid steps of the threshold. The one-directional discordance claim survives for the exponential-decay spectrum, with the added condition that it is only meaningful when the number of draws resolves α. This also sets a calibrated baseline for reading E2: a C_only rate of roughly 0.3-0.5% at 200 draws is what pure granularity looks like here.

---

## E2: same PR, different spectrum shape

**Question.** Is participation ratio the operative quantity, or a proxy for something else in the spectrum?

**Design.** n = 28, d = 119. Five spectrum families, exponential decay, power law, two-block, one spike, five spikes, each tuned so that *measured* sphere-PR hits 3, 4.5 and 6 within 5%. 15 cells, 400 datasets × 200 draws, M and C paired.

| target PR | family | measured PR | M | C | mean p M | mean p C | M-only | C-only |
|---|---|---|---|---|---|---|---|---|
| 3.0 | exp | 3.17 | 0.525 | 0.122 | 0.075 | 0.222 | 161 | 0 |
| 3.0 | power | 3.12 | 0.693 | 0.570 | 0.051 | 0.069 | 60 | 11 |
| 3.0 | two_block | 3.15 | 0.890 | **0.938** | 0.025 | 0.018 | 1 | 20 |
| 3.0 | one_spike | 3.23 | 0.868 | **0.940** | 0.027 | 0.018 | 1 | 30 |
| 3.0 | few_spikes | 4.37 † | 0.160 | 0.007 | 0.275 | 0.710 | 61 | 0 |
| 4.5 | exp | 4.74 | 0.130 | 0.015 | 0.240 | 0.432 | 48 | 2 |
| 4.5 | power | 4.55 | 0.200 | 0.205 | 0.145 | 0.153 | 23 | 25 |
| 4.5 | two_block | 4.76 | 0.445 | **0.657** | 0.081 | 0.053 | 4 | 89 |
| 4.5 | one_spike | 4.88 | 0.417 | **0.627** | 0.083 | 0.056 | 5 | 89 |
| 4.5 | few_spikes | 4.45 | 0.188 | 0.005 | 0.256 | 0.673 | 73 | 0 |
| 6.0 | exp | 6.04 | 0.033 | 0.000 | 0.412 | 0.607 | 13 | 0 |
| 6.0 | power | 6.10 | 0.055 | 0.045 | 0.258 | 0.254 | 10 | 6 |
| 6.0 | two_block | 6.70 | 0.125 | **0.255** | 0.164 | 0.116 | 4 | 56 |
| 6.0 | one_spike | 6.78 | 0.113 | **0.270** | 0.167 | 0.115 | 3 | 66 |
| 6.0 | few_spikes | 5.96 | 0.055 | 0.005 | 0.384 | 0.678 | 20 | 0 |

† `few_spikes` cannot express PR 3.0 at this n and d: with five equal leading eigenvalues its participation ratio has a floor, and the bisection terminated at the floor, 4.37. That cell is reported at its achieved PR and is not a matched comparison.

**Interpretation.** PR is **not** sufficient. At matched measured PR ≈ 4.5 the corrected arm rejects structureless data at 0.005 (few_spikes), 0.015 (exp), 0.205 (power), 0.627 (one_spike) and 0.657 (two_block). Those Wilson intervals are nowhere close to overlapping, so the divergence is not a sampling question. The direction of the divergence is the finding: families whose spectrum has a single dominant eigendirection, `one_spike`, and `two_block` which the tuner drove to a single unit eigenvalue, are the ones where the corrected construction is *worse than the uncorrected one*, and where the discordance reverses. Families with graded decay (`exp`, `few_spikes`) behave as the existing work describes, and `power` sits between the two regimes with C ≈ M. Any statement of the form "at participation ratio X the corrected test behaves like Y" is therefore underdetermined at fixed n and d, and the paper's PR-indexed regime boundaries hold only for the exponential family they were measured on.

---

## E2 verification: reversed discordance is real

**Question.** Ten of the fifteen E2 cells returned `C_only > 0`, some of them very large. Is any of it the granularity artifact that E1 established?

**Design.** Identical protocol to the E1 verification: four independent seeds per cell, then every discordant dataset re-tested at 2,000 draws.

| family | target PR | M | C | pooled C_only | rate | persists at 2000 draws |
|---|---|---|---|---|---|---|
| exp | 4.5 | 0.130 | 0.015 | 2 / 1600 | 0.0013 | **0 of 2 (0%)** |
| power | 3.0 | 0.693 | 0.570 | 40 / 1600 | 0.0250 | 4 of 40 (10%) |
| power | 4.5 | 0.200 | 0.205 | 89 / 1600 | 0.0556 | 36 of 89 (40%) |
| power | 6.0 | 0.055 | 0.045 | 37 / 1600 | 0.0231 | 13 of 37 (35%) |
| one_spike | 3.0 | 0.868 | 0.940 | 120 / 1600 | 0.0750 | 55 of 120 (46%) |
| one_spike | 4.5 | 0.417 | 0.627 | n/a | n/a | see JSON |
| one_spike | 6.0 | 0.113 | 0.270 | 279 / 1600 | 0.1744 | see JSON |
| two_block | 3.0 | 0.890 | 0.938 | 101 / 1600 | 0.0631 | 45 of 101 (45%) |
| two_block | 4.5 | 0.445 | 0.657 | 368 / 1600 | 0.2300 | **244 of 368 (66%)** |
| two_block | 6.0 | 0.125 | 0.255 | 212 / 1600 | 0.1325 | **142 of 212 (67%)** |

Per-seed counts reproduce to within 1-3 events in every cell; full detail in `e2_conly_check.json`.

**Interpretation.** Only `exp`: the family the published work used, shows the granularity signature (0.13% C_only, none surviving deeper resampling). Everywhere else the reversed discordance is substantially real: between 35% and 67% of C-only rejections persist at 2,000 draws, at C_only rates one to two orders of magnitude above the granularity baseline. For `two_block` and `one_spike` the corrected construction rejects structureless data roughly twice as often as the uncorrected one. This is a direct contradiction of the claim that correcting the null cannot manufacture significance, and it is not marginal. The mechanism is not established here; the fact that the two worst families are exactly the two with a single dominant eigendirection points at the geometry of the spherical projection under a near-rank-one covariance, and E3's radial/angular decomposition is the right place to look. What can be said without further work is that the corrected construction has no general conservatism guarantee, and that its advantage over the misspecified one is a property of the spectrum, not of the correction.
## E3: radial vs angular decomposition

**Question.** Is the inflation caused specifically by radial degrees of freedom the observed data cannot have?

**Design.** n = 28, d = 119, measured sphere-PR 4.36, structureless data, 400 datasets × 200 draws. Six null families are tested against the same observed statistic (mean 0.2598), each under two selection rules, `fixed`, the statistic at the observed k\* as M does, and `matched`, the max over k under the guard as C does, so that geometry and selection are never varied at the same time.

| null family | what it isolates | rule | null mean | null sd | rejection | events | 95% CI | mean p |
|---|---|---|---|---|---|---|---|---|
| N1 ambient Gaussian, no projection | M's null | fixed | 0.2250 | 0.0371 | 0.1525 | 61/400 | [0.121, 0.191] | 0.211 |
| N1 | | matched | 0.2347 | 0.0628 | 0.0500 | 20/400 | [0.033, 0.076] | 0.299 |
| N2 Gaussian → sphere | C's null geometry | fixed | 0.2345 | 0.0346 | 0.0525 | 21/400 | [0.035, 0.079] | 0.273 |
| N2 | | matched | 0.2482 | 0.0327 | 0.0175 | 7/400 | [0.009, 0.036] | 0.391 |
| N3 N2 directions, observed radii | consistency check | fixed | 0.2346 | 0.0348 | 0.0625 | 25/400 | [0.043, 0.091] | 0.274 |
| N3 | | matched | 0.2482 | 0.0330 | 0.0300 | 12/400 | [0.017, 0.052] | 0.391 |
| N4 observed directions, random radii | radial alone | fixed | 0.2307 | 0.0182 | 0.5500 | 220/400 | [0.501, 0.598] | 0.069 |
| N4 | | matched | 0.2353 | 0.0270 | 0.3450 | 138/400 | [0.300, 0.393] | 0.098 |
| N5 observed radii, random directions | angular alone | fixed | 0.0177 | 0.0039 | 1.0000 | 400/400 | [0.991, 1.000] | 0.005 |
| N5 | | matched | 0.0204 | 0.0036 | 1.0000 | 400/400 | [0.991, 1.000] | 0.005 |
| N6 angular second-moment matched | covariance-preserving angular | fixed | 0.2324 | 0.0342 | 0.0725 | 29/400 | [0.051, 0.102] | 0.254 |
| N6 | | matched | 0.2462 | 0.0321 | 0.0225 | 9/400 | [0.012, 0.042] | 0.372 |

N3 reproduces N2 as required: the observed radii are all 1.0, and the two arms agree to within overlapping intervals under both rules (fixed 0.0625 vs 0.0525; matched 0.0300 vs 0.0175). The machinery is behaving.

**Interpretation.** Neither N4 nor N5 accounts for the M→C gap, and stating that plainly is the result. The gap itself is 0.1525 (N1 fixed = M) to 0.0175 (N2 matched = C), and it decomposes cleanly into two roughly equal, compounding contributions: holding selection fixed, re-projecting the draws onto the sphere moves rejection from 0.1525 to 0.0525; holding geometry fixed, matching the k-search moves it from 0.1525 to 0.0500. Applying both gives 0.0175. The N1→N4 shift (+0.3975) and the N1→N5 shift (+0.8475) both go the wrong way, freezing either component while randomizing the other produces a null far worse than M, not better, so the inflation is not attributable to radial freedom as such. What the radial degrees of freedom supply is the null's *spread*: N4, which freezes the angular configuration at the observed one and lets only radii vary, halves the null standard deviation from 0.0371 to 0.0182 while leaving the null mean essentially unchanged, and that narrowing alone raises rejection to 0.55. N5 fails differently and more violently, collapsing the null mean from 0.2250 to 0.0177, an isotropic direction field has almost no silhouette structure, so every observed value looks extreme. The honest summary is that the observed statistic depends far more on the angular configuration than on the radii, that M's over-rejection comes from a null whose spread is inflated relative to a sphere-constrained one, and that re-projection fixes it by removing a degree of freedom the observed data does not have. N6 confirms that a covariance-preserving angular null reproduces C's behaviour closely (0.0225 vs 0.0175, overlapping intervals).

---

## E4: Type I error as a function of the size of the k-search family

**Question.** How much of the inflation is attributable to the size of the candidate set K alone?

**Design.** n = 28, d = 119, measured sphere-PR 4.27, structureless data, 400 datasets × 200 draws, K = {2}, {2,3}, … {2,…,11}. The guard (minimum cluster size 4 at n = 28) makes most large k unreachable, so effective |K|, the mean number of k values that actually yield a guard-passing partition, is recorded on the observed data and on null draws, and the result is read against that rather than against nominal |K|.

| K | nominal \|K\| | effective \|K\| (null) | M | M events | M 95% CI | C | C events | C 95% CI |
|---|---|---|---|---|---|---|---|---|
| {2} | 1 | 1.00 | 0.013 | 5/400 | [0.006, 0.030] | 0.015 | 6/400 | [0.007, 0.032] |
| {2,3} | 2 | 1.99 | 0.113 | 45/400 | [0.085, 0.147] | 0.010 | 4/400 | [0.004, 0.025] |
| {2..4} | 3 | 2.85 | 0.180 | 72/400 | [0.146, 0.220] | 0.013 | 5/400 | [0.005, 0.029] |
| {2..5} | 4 | 3.28 | 0.185 | 74/400 | [0.150, 0.226] | 0.015 | 6/400 | [0.007, 0.032] |
| {2..6} | 5 | 3.36 | 0.163 | 65/400 | [0.130, 0.202] | 0.022 | 9/400 | [0.012, 0.042] |
| {2..7} | 6 | 3.34 | 0.177 | 71/400 | [0.143, 0.217] | 0.015 | 6/400 | [0.007, 0.032] |
| {2..8} | 7 | 3.35 | 0.152 | 61/400 | [0.121, 0.191] | 0.010 | 4/400 | [0.004, 0.025] |
| {2..9} | 8 | 3.35 | 0.177 | 71/400 | [0.143, 0.217] | 0.013 | 5/400 | [0.005, 0.029] |
| {2..10} | 9 | 3.34 | 0.142 | 57/400 | [0.111, 0.181] | 0.010 | 4/400 | [0.004, 0.025] |
| {2..11} | 10 | 3.35 | 0.163 | 65/400 | [0.130, 0.202] | 0.022 | 9/400 | [0.012, 0.042] |

**Interpretation.** This is a standalone result and does not depend on any other experiment here. Effective |K| saturates at about 3.35 no matter how large the nominal set gets, because at n = 28 the guard admits at most seven clusters of four and in practice admits three or four candidate values; nominal |K| beyond 4 buys nothing. M tracks effective |K| and saturates with it: 0.013 at |K| = 1, 0.113 at 2, 0.180 at 2.85, and flat between 0.142 and 0.185 thereafter. C is insensitive to |K| throughout, 0.010 to 0.022 across the whole sweep, always overlapping nominal. Two things follow. First, essentially the entire inflation of the misspecified construction at this geometry is attributable to the k-search: with no search at all (K = {2}) M and C are indistinguishable at 0.013 and 0.015, and every subsequent point of M's excess appears as candidates are added. Second, reporting Type I error against nominal |K| would misdescribe the mechanism, since the curve against nominal |K| appears to plateau for no visible reason while against effective |K| it is a clean monotone rise to saturation. Anyone quoting a k-range should quote how much of it the guard actually admits.

---

## E5: alternative null families

**Question.** Is the problem Gaussian nulls specifically, or covariance preservation generally?

**Design.** n = 28, d = 119, measured sphere-PR 4.27, structureless data, 400 datasets × 200 draws. Six ways of generating a draw, all C-like in that every draw receives the same max-over-k search under the same guard the observed statistic received.

| null family | preserves covariance | null mean | rejection | events | 95% CI | mean p | KS D |
|---|---|---|---|---|---|---|---|
| Gaussian + reprojection (C, baseline) | yes | 0.2482 | 0.020 | 8/400 | [0.010, 0.039] | 0.380 | 0.222 |
| angular Gaussian (N6) | yes | 0.2460 | 0.035 | 14/400 | [0.021, 0.058] | 0.359 | 0.245 |
| random rotation | yes (exactly) | 0.2608 | 0.023 | 9/400 | [0.012, 0.042] | 0.590 | 0.160 |
| isotropic on the sphere | no | 0.0205 | **1.000** | 400/400 | [0.991, 1.000] | 0.005 | 0.995 |
| von Mises-Fisher | no (mean direction only) | 0.0203 | **1.000** | 400/400 | [0.991, 1.000] | 0.005 | 0.995 |
| per-coordinate resample | marginals only | 0.0296 | **1.000** | 400/400 | [0.991, 1.000] | 0.005 | 0.995 |

**Interpretation.** This answers the limitation the paper concedes, and the answer is that covariance preservation is not the problem, it is the requirement. Every null that discards the covariance structure rejects structureless data 400 times out of 400, with mean p pinned at the floor of 0.005 and KS D of 0.995. That includes the two non-parametric constructions a reader might reach for as more assumption-free: resampling each coordinate independently destroys cross-coordinate dependence and lands in the same place as sampling uniformly on the sphere, and a fitted von Mises-Fisher is nearly uniform here because the concentration estimated from 28 anisotropic unit vectors is small. The three covariance-preserving nulls agree with each other to within overlapping intervals (0.020, 0.023, 0.035) and are the only ones in the neighbourhood of nominal. The random-rotation null deserves a specific caveat, stated in advance of running it and borne out: rotation is an isometry, so the silhouette at a fixed partition is exactly invariant under it, and the only reason this arm varies at all is k-means initialisation in rotated coordinates. Its apparently best-in-table calibration (KS D 0.160, mean p 0.590) is an artifact of measuring initialisation noise rather than evidence that it is a good null. It should not be recommended on the strength of this row.

---

## E6: other cluster statistics

**Question.** Is the M/C gap silhouette-specific?

**Design.** n = 28, d = 119, measured sphere-PR 4.27, structureless data, 400 datasets × 200 draws. The statistic is replaced by each of silhouette, Calinski-Harabasz, negative Davies-Bouldin and negative within/between dispersion ratio, and for each the k-search maximises that statistic under the same guard.

| statistic | M | M events | M 95% CI | C | C events | C 95% CI | M - C gap | mean p M | mean p C | KS D M | KS D C |
|---|---|---|---|---|---|---|---|---|---|---|---|
| silhouette | 0.1875 | 75/400 | [0.152, 0.229] | 0.0275 | 11/400 | [0.015, 0.049] | **+0.160** | 0.199 | 0.391 | 0.502 | 0.223 |
| Calinski-Harabasz | 0.0175 | 7/400 | [0.009, 0.036] | 0.0225 | 9/400 | [0.012, 0.042] | -0.005 | 0.328 | 0.380 | 0.320 | 0.217 |
| -Davies-Bouldin | 0.0525 | 21/400 | [0.035, 0.079] | 0.0525 | 21/400 | [0.035, 0.079] | 0.000 | 0.347 | 0.414 | 0.258 | 0.163 |
| -within/between | 0.0275 | 11/400 | [0.015, 0.049] | 0.0600 | 24/400 | [0.041, 0.088] | -0.033 | 0.351 | 0.445 | 0.278 | 0.092 |

**Interpretation.** The gap does not persist, and this narrows the paper's claim rather than widening it. Only the silhouette shows a large M/C separation (+0.160, with non-overlapping intervals). For the other three statistics the two constructions are indistinguishable, the gaps are -0.005, 0.000 and -0.033, every pair of intervals overlaps substantially, and both arms sit at or near nominal. The misspecified construction is only badly anti-conservative when the statistic is the silhouette: its rejection rate is 0.1875 for silhouette against 0.0175, 0.0525 and 0.0275 for the others. The mechanism therefore looks specific to what the silhouette measures on a sphere, not to covariance-preserving nulls under max-over-k in general. Two caveats. The Calinski-Harabasz and within/between rows rest on 7 and 11 events respectively and are in the low-event audit below. And the sign of the small gaps reverses for two statistics, with C-only counts of 7, 8 and 20 against M-only counts of 5, 8 and 7, which is the same reversed-discordance phenomenon E2 established, here at a scale too small to separate from noise.

---

## E7: other clustering algorithms

**Question.** Does the gap survive replacing k-means?

**Design.** Same geometry, statistic held at the silhouette, 400 datasets × 200 draws; the partitioner varies over k-means, Ward agglomerative, Gaussian mixture with full covariance and `reg_covar=1e-4`, and spectral clustering with an RBF affinity.

| algorithm | M | M events | C | C events | M - C gap | fit failures | M-only | C-only |
|---|---|---|---|---|---|---|---|---|
| k-means | 0.1875 | 75/400 | 0.0225 | 9/400 | +0.165 | 0 | 67 | 1 |
| Ward | 0.1175 | 47/400 | 0.0275 | 11/400 | +0.090 | 0 | 43 | 7 |
| Gaussian mixture | 0.2050 | 82/400 | 0.0175 | 7/400 | +0.188 | 0 | 75 | 0 |
| spectral | 0.1375 | 55/400 | 0.0225 | 9/400 | +0.115 | 0 | 53 | 7 |

**Interpretation.** The gap persists under every algorithm tested, so the finding is not a k-means artifact and the paper's claim widens along this axis. M rejects structureless data at between 0.118 and 0.205 while C stays between 0.018 and 0.028 in all four cases; every M interval excludes 0.05 and every C interval includes or sits below it. The Gaussian mixture is the most anti-conservative under M at 0.2050 despite being severely ill-conditioned at n = 28 with d = 119, the regularisation at `reg_covar=1e-4` was sufficient that **no fit failed in any of the 400 datasets or their draws**, which was not guaranteed in advance and is worth recording. Read together with E6, the scope statement is that the M/C gap is a property of the silhouette that is robust to how the partition is obtained, rather than a property of k-means or of cluster validation generally.

---

## E8: normalization schemes

**Question.** Does the failure track participation ratio across normalisations?

**Design.** Same structureless geometry, n = 28, d = 119, 400 datasets × 200 draws. The transform applied to both the observed data and every null draw varies over none, L2, per-feature z-score, ZCA whitening and PCA whitening, and all four constructions are run under each. Since `nullcal.null_test` hardcodes L2 as both the observed transform and the re-projection step, this experiment reimplements that loop with the transform as a parameter, a literal transcription, same guard, same estimator, same draw mechanics, and modifies nothing in `nullcal`.

| scheme | measured PR | M | G | S | C | M-only | C-only |
|---|---|---|---|---|---|---|---|
| none | 3.52 | 0.005 (2) | 0.005 (2) | 0.003 (1) | 0.003 (1) | 1 | 0 |
| L2 | 4.36 | 0.155 (62) | 0.060 (24) | 0.052 (21) | 0.020 (8) | 56 | 2 |
| z-score | 4.48 | 0.008 (3) | 0.015 (6) | 0.005 (2) | 0.010 (4) | 1 | 2 |
| ZCA whitening † | 27.00 | 0.000 (0) | **0.604 (233)** | 0.000 (0) | **0.630 (243)** | 0 | **243** |
| PCA whitening † | 27.00 | 0.000 (0) | 0.000 (0) | 0.000 (0) | 0.074 (26) | 0 | 26 |

Denominators: none and z-score 399 datasets, L2 400, ZCA whitening 386, PCA whitening 352. The shortfalls are datasets where no k passed the guard after the transform; they were excluded, not counted as non-rejections. † the two whitening rows are provisional, see below.

**Interpretation.** The failure does not track participation ratio. At PR between 3.5 and 4.5 the three schemes behave completely differently: raw data and z-scored data show no inflation in any construction (every arm between 0.003 and 0.015), while L2 at essentially the same PR shows the full 0.155 → 0.020 ladder the paper describes. The anti-conservatism is therefore a property of the spherical constraint, not of the effective dimensionality that constraint happens to produce. The two whitening rows are stranger and should be treated as provisional. Both report PR 27.00 by construction, since whitening equalises the spectrum, and both give M = S = 0.000, as expected when the null draws are unconstrained in a space where the observed data has been forced to unit covariance. But the re-projecting constructions diverge sharply, ZCA gives G = 0.604 and C = 0.630 with 243 C-only rejections, PCA whitening gives G = 0.000 and C = 0.074. Since ZCA and PCA whitening differ only by a rotation and the silhouette depends only on distances, these two rows should be identical, and the fact that M and S do match across them (0.000, null means 0.0608 and 0.0600) while G and C do not points at the re-application of the transform to each draw. The likely cause is numerical: ZCA inverts singular values spanning the full dynamic range of an exp(-i/1.73) spectrum at n = 28 and reconstructs through V, which amplifies the smallest components catastrophically, whereas PCA whitening never leaves the well-conditioned basis. **This should be treated as a conditioning artifact and re-run in higher precision before being cited as a statistical result.** What survives regardless is the primary point: PR is not the operative variable across normalisations.

---

## E9: calibration stability across seeds

**Question.** How much does the corrected arm's point estimate move under re-execution?

**Design.** Five independent replications of a 300-dataset × 300-draw paired calibration at measured sphere-PR ≈ 4.38, n = 28, d = 119, differing only in seed.

| replication | measured PR | M | M events | M 95% CI | C | C events | C 95% CI | M-only | C-only |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 4.37 | 0.1767 | 53/300 | [0.138, 0.224] | 0.0067 | 2/300 | [0.002, 0.024] | 51 | 0 |
| 1 | 4.39 | 0.1900 | 57/300 | [0.150, 0.238] | 0.0167 | 5/300 | [0.007, 0.038] | 52 | 0 |
| 2 | 4.40 | 0.2000 | 60/300 | [0.159, 0.249] | 0.0133 | 4/300 | [0.005, 0.034] | 56 | 0 |
| 3 | 4.38 | 0.2233 | 67/300 | [0.180, 0.274] | 0.0167 | 5/300 | [0.007, 0.038] | 62 | 0 |
| 4 | 4.37 | 0.1900 | 57/300 | [0.150, 0.238] | 0.0167 | 5/300 | [0.007, 0.038] | 52 | 0 |

M across seeds: mean 0.1960, sd 0.0174, range 0.1767-0.2233, pooled 95% CI [0.1767, 0.2169].
C across seeds: mean 0.0140, sd 0.0043, range 0.0067-0.0167, event counts 2, 5, 4, 5, 5, pooled 95% CI [0.0092, 0.0213].

**Interpretation.** The instability is real and is confined to the corrected arm. M moves by a factor of 1.26 across seeds (0.1767 to 0.2233) with every interval overlapping every other, which is ordinary sampling variation on 53-67 events. C moves by a factor of **2.50** (0.0067 to 0.0167) on 2-5 events, which is exactly the behaviour invariant 7 of the brief anticipates for rates resting on a handful of events. No single replication's C estimate should be quoted as a point value; the pooled estimate over all 1,500 datasets is 0.0140 with 95% CI [0.0092, 0.0213], and that interval, not any of the five point estimates, is what the data support. C-only was 0 in all five replications, consistent with E1's verification that the exponential-decay family shows no genuine reversed discordance.

---

## E10: dimension sweep

**Question.** Are the quoted boundaries specific to d = 119?

**Design.** Measured sphere-PR held at ≈ 4.4 and n/d held at 0.24 (the fibromyalgia ratio) and 1.0 (where the earlier n sweep placed a transition), with d varying over 32, 64, 128, 256. A 3-dataset timing probe sized each cell; none needed shrinking, so all eight ran at 400 datasets × 200 draws.

| d | n | n/d | measured PR | M | M events | C | C events | M-only | C-only |
|---|---|---|---|---|---|---|---|---|---|
| 32 | 8 | 0.24 | 4.22 | 0.000 | 0/133 † | 0.000 | 0/133 † | 0 | 0 |
| 64 | 15 | 0.24 | 4.53 | 0.008 | 3/396 | 0.000 | 0/396 | 3 | 0 |
| 128 | 31 | 0.24 | 4.42 | 0.170 | 68/400 | 0.030 | 12/400 | 58 | 2 |
| 256 | 61 | 0.24 | 4.27 | 0.660 | 264/400 | 0.255 | 102/400 | 163 | 1 |
| 32 | 32 | 1.00 | 4.43 | 0.207 | 83/400 | 0.030 | 12/400 | 72 | 1 |
| 64 | 64 | 1.00 | 4.30 | 0.735 | 294/400 | 0.312 | 125/400 | 169 | 0 |
| 128 | 128 | 1.00 | 4.41 | 0.998 | 399/400 | 0.845 | 338/400 | 61 | 0 |
| 256 | 256 | 1.00 | 4.46 | 1.000 | 400/400 | 1.000 | 400/400 | 0 | 0 |

† At d = 32, n = 8 only **133 of 400** datasets produced any k passing the guard; the other 267 had no statistic to test and were excluded rather than counted as non-rejections. At d = 64, n = 15 the figure is 396 of 400. All other cells used the full 400.

**Interpretation.** The boundaries are not a property of n/d and are not transferable from d = 119. Holding both PR and n/d fixed, Type I error rises monotonically and steeply with d in both slices: at n/d = 0.24 the corrected arm goes 0.000 → 0.000 → 0.030 → 0.255 as d goes 32 → 256, and at n/d = 1.0 it goes 0.030 → 0.312 → 0.845 → 1.000. Absolute sample size, not the ratio, is what governs the behaviour, the d = 32, n/d = 1.0 cell has n = 32 and C = 0.030, while the d = 256, n/d = 0.24 cell has n = 61 and C = 0.255, so the cell with the *smaller* ratio but larger n is far worse. The two cells at the smallest n (8 and 15) return zero rejections in both arms, which is the rank-deficiency floor rather than calibration; at n = 8 the guard admits only k = 2 and there is almost nothing for either construction to do. Any statement in the paper that locates a transition at n/d ≈ 1 should be restated in terms of n, with d = 119 named as the setting in which it was measured.

---

## E11: consensus clustering from stored labels

**Question.** Is a consensus partition more stable than the individual runs it is built from?

**Design.** No training of any kind. Per-run label vectors were read from disk for two MDD protocols, the 20-seed fixed-architecture control (`fixed_arch_control/runs/seed*.json`) and the 20-run search-varying protocol (`multirun_v2/by_run/run*_labels.csv`), 51 subjects each. The co-clustering matrix is the fraction of runs in which a pair shares a cluster; the consensus partition is average-linkage clustering on 1 - co-occurrence with k chosen by silhouette on that precomputed distance under the same guard; stability is assessed by rebuilding the consensus from 1,000 bootstrap resamples **of the runs**.

| protocol | runs | consensus k | cluster sizes | ARI run vs consensus | ARI pairwise runs | ARI bootstrap consensus | gain |
|---|---|---|---|---|---|---|---|
| fixed architecture | 20 | 7 | 11, 7, 7, 4, 6, 7, 9 | 0.3654 ± 0.1119 | 0.2331 ± 0.1100 | 0.6774 ± 0.1256 | **+0.4443** |
| search-varying | 20 | 5 | 9, 11, 12, 8, 11 | 0.3701 ± 0.1563 | 0.2259 ± 0.1360 | 0.6079 ± 0.1266 | **+0.3820** |

The fibromyalgia fixed-architecture control was skipped: `results/fm_fixed_arch_control/` stores only per-run summary statistics (run, seed, k_selected, silhouette, perm_p) and no per-run label vectors or embeddings, so no consensus can be built for that cohort without retraining, which is out of scope.

**Interpretation.** The consensus partition is substantially more stable to resampling the runs than the runs are to each other, in both protocols: bootstrap-consensus ARI of 0.68 and 0.61 against pairwise run ARI of 0.23 for both, a gain of +0.44 and +0.38. That is the expected direction and it is not a small effect. It does not, however, license reading the consensus as a recovered subtype structure, and two features argue against that reading. Individual runs agree with the consensus at ARI 0.37 on average, so no run reproduces it; and the consensus k of 7 and 5 matches neither protocol's modal per-run selection (4 and 4-5 respectively), which means the consensus is not summarising a partition any run actually found. Aggregating unstable partitions yields a stable *summary*, which is a property of averaging rather than evidence that the summary corresponds to anything in the data. Whether these consensus partitions survive a properly constructed null is not tested here.

---

## E12: representation families (reduced substitute)

**Question, as specified.** Is the geometry-dependence specific to contrastive GNN embeddings, as against PCA, random-projection or autoencoder embeddings?

**E12 as specified could not run.** It requires the stored Fisher-z connectivity features for the 28 fibromyalgia patients, from which the alternative embeddings would be built. Those are not on disk: `data/outputs/` holds only 28 × 119 embeddings (`trained_fm_embeddings.npy`, `untrained_fm_embeddings.npy`, `Embeddings.npy`), and `resting_state_dep_data/` is the MDD BIDS raw imaging for ds002748, not fibromyalgia connectivity. Recomputing connectivity means touching fMRI preprocessing, which the brief puts out of scope. What ran instead is the four-construction ladder on each stored representation at its own measured geometry, 2,000 draws each.

| representation | kind | PR raw | PR sphere | observed | k\* | p M | p G | p S | p C | p spread |
|---|---|---|---|---|---|---|---|---|---|---|
| GNN trained | stored | 3.70 | 4.37 | 0.2415 | 4 | 0.1369 | 0.3888 | 0.3963 | 0.6732 | 0.536 |
| GNN untrained | stored | 1.59 | 1.59 | 0.4600 | 2 | 0.1979 | 0.2799 | 0.1909 | 0.2844 | 0.094 |
| `Embeddings.npy` | stored | 3.70 | 4.37 | 0.2415 | 4 | 0.1369 | 0.3888 | 0.3963 | 0.6732 | 0.536 |
| random projection 0 | derived | 3.39 | 4.18 | 0.2388 | 4 | 0.1949 | 0.4993 | 0.5237 | 0.7851 | 0.590 |
| random projection 1 | derived | 3.59 | 4.23 | 0.2399 | 2 | 0.3823 | 0.5352 | 0.4783 | 0.7751 | 0.393 |
| random projection 2 | derived | 3.13 | 4.22 | 0.2413 | 4 | 0.1414 | 0.4058 | 0.4588 | 0.7116 | 0.590 |

**Interpretation, and what it does not license.** This does **not** answer the specified question, because every representation in the table comes from the same encoder architecture; it is a geometry check on real data, not a representation-family comparison. Read that way, three things hold. `Embeddings.npy` is bit-identical to `trained_fm_embeddings.npy`, which is worth knowing for provenance. The trained embedding reproduces the paper's reported behaviour closely, PR 3.70 unnormalized and 4.37 on the sphere, a p-spread of 0.536 across the four constructions, and a corrected p of 0.673 that finds no structure. The untrained encoder is the informative contrast: at PR 1.59 it is far more collapsed, its observed silhouette is much higher at 0.4600, and yet its four p-values span only 0.094 and none rejects. That is consistent with E1, where at PR below about 2 both constructions reject structureless data at rates near 1 and the ladder loses its ability to discriminate at all. The random-projection rows behave like the trained embedding they derive from, at slightly lower PR and slightly larger spread, which is expected and adds little.
## Claims in PAPER_v5.md that these results change

**A note on the target file.** `PAPER_v5.md` and `PAPER_v5_SUPPLEMENT.md` do not exist in this repository. The latest manuscript on disk is `PAPER_v3.md` (1,536 lines, modified 2026-08-19 22:32). Every quotation below is from `PAPER_v3.md` with its line number, and every claim is one that appears in the brief's summary of established results as well, so the mapping to v5 should be direct. Line numbers will need rechecking against v5.

---

### 1. CONTRADICTED: the discordance is not one-directional

This is the most important result of the run and it goes against the paper.

**Current wording** (line 1101): "The inflation is strictly one-directional; there is no dataset on which correcting the null manufactures a rejection." Also line 1109: the corrected test "never manufactures a result it previously missed." Also lines 23, 103, 1412 and 1430: "201 rejections the corrected test did not, against zero in the other direction." The same assertion is in `nullcal.calibrate_paired`'s docstring: "Correcting a null should never manufacture significance."

**New evidence.** E2 varied the eigenvalue spectrum at matched measured PR and n = 28, d = 119. Every cell that produced C-only rejections was re-run at four independent seeds and every discordant dataset re-tested at 2,000 draws.

| family | target PR | M | C | pooled C-only | persists at 2000 draws |
|---|---|---|---|---|---|
| exp | 4.5 | 0.130 | 0.015 | 2 / 1600 | 0 of 2 (0%) |
| power | 4.5 | 0.200 | 0.205 | 89 / 1600 | 36 of 89 (40%) |
| one_spike | 3.0 | 0.868 | 0.940 | 120 / 1600 | 55 of 120 (46%) |
| two_block | 4.5 | 0.445 | **0.657** | 368 / 1600 | **244 of 368 (66%)** |
| two_block | 6.0 | 0.125 | **0.255** | 212 / 1600 | 142 of 212 (67%) |

The exponential-decay family, the only one the published work used, behaves exactly as the paper says. No other family does. For single-dominant-eigendirection spectra the corrected construction rejects structureless data roughly twice as often as the misspecified one, and between 40% and 67% of the reversed discordances survive a tenfold increase in draw resolution, against a granularity baseline of 0.3-0.5% established independently in E1's verification.

**Suggested replacement.** "In 1,000 paired comparisons on data with an exponentially decaying covariance spectrum, the misspecified construction produced 201 rejections the corrected test did not, against zero in the other direction. That one-directionality is a property of the spectrum, not of the constructions: on spectra with a single dominant eigendirection at the same participation ratio, the ordering reverses and the corrected construction rejects structureless data roughly twice as often as the misspecified one (§Ex). Correcting the null is therefore not guaranteed to be conservative, and the guarantee should not be claimed."

**Also required.** The `calibrate_paired` docstring in `nullcal.py` asserts the same guarantee and must be amended; it currently instructs users to treat a C-only rejection as evidence of a bug.

---

### 2. NARROWED: participation ratio is not sufficient, and the regime boundaries are spectrum-specific

**Current wording.** `nullcal.type1_surface_cell` docstring: "both constructions safe above PR ~6, misspecification severe and correction effective between ~3.5 and 6, and neither construction valid below ~3.5." The manuscript leans on the same index throughout, contrasting the embedding at PR 4.37 with the raw edge features at PR 20.5 (lines 103, 218).

**New evidence.** Three independent experiments show PR does not determine behaviour.

- **E2**, at matched measured PR ≈ 4.5 and fixed n and d, the corrected arm rejects at 0.005 (few_spikes), 0.015 (exp), 0.205 (power), 0.627 (one_spike), 0.657 (two_block). Wilson intervals nowhere near overlapping.
- **E8**, at PR between 3.5 and 4.5, raw and z-scored data show no inflation in any construction (0.003-0.015) while L2 at the same PR shows the full ladder (M 0.155 → C 0.020).
- **E10**, holding PR ≈ 4.4 and n/d fixed, Type I error rises steeply with d.

**Suggested replacement.** "Participation ratio summarises the regime for a given spectrum family, normalisation and dimension, and the boundaries quoted here were measured on exponentially decaying spectra of L2-normalized data at d = 119. They do not transfer across spectrum shape, normalisation scheme or dimension, and PR alone should not be used to decide whether a given representation is in the safe regime."

---

### 3. CONTRADICTED: the transition is governed by n, not by n/d

**Current wording.** From the established results the brief summarises: "Transition near n/d ≈ 1."

**New evidence.** E10, holding measured PR ≈ 4.4:

| | d=32 | d=64 | d=128 | d=256 |
|---|---|---|---|---|
| n/d = 0.24, C | 0.000 (n=8) | 0.000 (n=15) | 0.030 (n=31) | 0.255 (n=61) |
| n/d = 1.00, C | 0.030 (n=32) | 0.312 (n=64) | 0.845 (n=128) | 1.000 (n=256) |

The cell at n/d = 0.24, d = 256 (n = 61, C = 0.255) is far worse than the cell at n/d = 1.00, d = 32 (n = 32, C = 0.030), despite the smaller ratio. The ordering follows n, not n/d.

**Suggested replacement.** "At d = 119 the corrected construction becomes anti-conservative as n approaches d. The controlling variable is the absolute sample size rather than the ratio n/d: at fixed n/d and fixed participation ratio, rejection rises steeply with d (§Ex). The n/d ≈ 1 boundary quoted here is specific to d = 119."

---

### 4. NARROWED: the M/C gap is silhouette-specific

**Current wording.** The paper's title and framing, "Where Covariance-Preserving Cluster Validation Fails", and its primary claim (line 103) are stated for covariance-preserving nulls under a k-search generally, with the silhouette as the instance.

**New evidence.** E6, same geometry, only the statistic varying:

| statistic | M | C | gap |
|---|---|---|---|
| silhouette | 0.1875 | 0.0275 | **+0.160** |
| Calinski-Harabasz | 0.0175 | 0.0225 | -0.005 |
| -Davies-Bouldin | 0.0525 | 0.0525 | 0.000 |
| -within/between | 0.0275 | 0.0600 | -0.033 |

Only the silhouette shows a gap. For the other three statistics both constructions sit at or near nominal and are indistinguishable.

**Suggested replacement.** Add a scope sentence to the primary claim: "The inflation is specific to the silhouette. Repeating the calibration with Calinski-Harabasz, Davies-Bouldin or a within/between dispersion ratio as the selected statistic leaves both constructions at or near nominal and removes the gap entirely (§Ex). The failure is a property of what the silhouette measures on a sphere, not of covariance-preserving nulls under selection in general."

---

### 5. WIDENED: the gap is not a k-means artifact

**New evidence.** E7, statistic held at the silhouette, partitioner varying: k-means gap +0.165, Ward +0.090, Gaussian mixture +0.188, spectral +0.115. Every M interval excludes 0.05; every C interval includes or sits below it. No Gaussian-mixture fit failed at `reg_covar=1e-4`.

**Suggested addition.** "The gap survives replacing k-means with Ward agglomerative clustering, a regularised full-covariance Gaussian mixture, or spectral clustering, with M between 0.118 and 0.205 and C between 0.018 and 0.028 in all four cases (§Ex)."

---

### 6. QUALIFIED: the corrected arm's rejection rate needs an interval, not a point

**Current wording** (line 1412): "the corrected one rejects at 0.4 percent." Line 1430: "the corrected test rejects at 0.004 against a nominal 0.05." The manuscript already concedes at line 1190 that a re-execution gave a different value and that "a rate resting on a handful of events should be read through its interval."

**New evidence.** E9 ran five independent 300 × 300 paired calibrations at the fibromyalgia geometry. C gave 0.0067, 0.0167, 0.0133, 0.0167, 0.0167 on 2, 5, 4, 5, 5 events, a **2.50-fold** swing between smallest and largest. Pooled over all 1,500 datasets: **0.0140, 95% CI [0.0092, 0.0213]**. E1's independent cell at the same geometry gave 0.013 on 5 of 400 events, CI [0.005, 0.029]. M by contrast moved only 1.26-fold across the same five seeds.

**Suggested replacement.** "Over 1,500 structureless datasets in five independent replications the corrected construction rejects at 0.014, 95% CI [0.009, 0.021]. The instability the earlier version noted is confined to this arm and is now bounded: individual 300-dataset replications range over a factor of 2.5 because each rests on two to five events, while the misspecified arm's rate is stable to within a factor of 1.26."

The existing caveat at line 1190 is correct and should be kept; it can now cite a measured bound instead of a single re-execution.

---

### 7. STRENGTHENED: a conceded limitation is now answered

**Current position.** The paper concedes it cannot say whether the problem is Gaussian nulls specifically or covariance preservation generally.

**New evidence.** E5, all arms C-like, same geometry: Gaussian + reprojection 0.020, angular Gaussian 0.035, random rotation 0.023, against isotropic-on-sphere 1.000, von Mises-Fisher 1.000, per-coordinate resample 1.000, each with mean p pinned at 0.005 and KS D 0.995.

**Suggested addition.** "Covariance preservation is not the source of the problem but a requirement of any usable null here. Nulls that discard the covariance structure, uniform on the sphere, a fitted von Mises-Fisher, or independent per-coordinate resampling, reject structureless data in 400 of 400 datasets. Only covariance-preserving constructions approach nominal (§Ex)."

The random-rotation null should be reported but not recommended: the silhouette is exactly invariant under rotation, so that arm measures k-means initialisation noise and its apparent calibration is an artifact.

---

### 8. NEW: Type I error against the effective size of the k-search family

Not currently in the paper, and standalone.

**Evidence.** E4, n = 28, d = 119, PR ≈ 4.3. Effective |K| saturates at ≈ 3.35 regardless of nominal |K| up to 10, because the guard admits at most three or four candidate values at this n. M rises with effective |K| and saturates with it, 0.013 at |K| = 1, 0.113 at 2, 0.180 at 2.85, flat at 0.142-0.185 thereafter, while C is flat at 0.010-0.022 throughout.

**Suggested addition.** "With no k-search at all (K = {2}) the two constructions are indistinguishable at 0.013 and 0.015. Essentially the entire inflation of the misspecified construction at this geometry is attributable to the k-search, and it saturates with the number of candidate k the guard actually admits rather than with the nominal range. Reported k-ranges should be accompanied by the effective count (§Ex, Figure)."

---

### 9. NEW: mechanism of the geometry correction

**Evidence.** E3 decomposes the M→C gap while holding the selection rule constant. Re-projection alone moves rejection from 0.1525 to 0.0525; matching the k-search alone moves it from 0.1525 to 0.0500; both together give 0.0175. The two contributions are roughly equal and compound. Freezing the angular configuration and randomising radii (N4) halves the null's standard deviation, 0.0371 → 0.0182, and drives rejection to 0.550; randomising directions while holding radii (N5) collapses the null mean from 0.2250 to 0.0177 and rejects 400 of 400.

**Suggested addition.** "The radial degrees of freedom the ambient null enjoys are what supply its spread; constraining draws to the sphere removes them. The observed statistic depends far more on the angular configuration than on the radii, and a null matching only the angular second-moment matrix reproduces the fully corrected construction to within overlapping intervals (0.023 vs 0.018) (§Ex)."

---

### 10. FLAGGED, NOT YET A CLAIM: whitening

E8's two whitening rows disagree in a way they mathematically should not: ZCA and PCA whitening differ only by a rotation and the silhouette depends only on distances, yet the re-projecting constructions give G = 0.604, C = 0.630 under ZCA and G = 0.000, C = 0.074 under PCA whitening. The non-reprojecting arms do match (M = S = 0.000, null means 0.0608 and 0.0600). The likely cause is numerical: ZCA inverts singular values spanning the full dynamic range of a steep spectrum at n = 28 and reconstructs through V. **Do not cite these two rows until they are re-run in higher precision.** The rest of E8 is unaffected.

---

## Rates resting on fewer than 10 events

Sixty-four quoted rates in this run rest on fewer than 10 rejection events and are loose to roughly a factor of two under re-execution, as E9 demonstrates directly. Each must be read through its Wilson interval and never quoted as a point estimate. The complete audit is in `_tables.md`; the entries that matter for the paper's claims are below.

| experiment | cell | arm | rate | events | datasets | 95% CI |
|---|---|---|---|---|---|---|
| E1 | PR 4.5, n=28 (the fibromyalgia geometry) | C | 0.013 | 5 | 400 | [0.005, 0.029] |
| E1 | PR 6.0, n=60 | C | 0.022 | 9 | 400 | [0.012, 0.042] |
| E1 | PR 9.0, n=119 | C | 0.020 | 8 | 400 | [0.010, 0.039] |
| E1 | PR 15.0, n=119 | M | 0.015 | 6 | 400 | [0.007, 0.032] |
| E1 | PR 15.0, n=180 | M | 0.013 | 5 | 400 | [0.005, 0.029] |
| E1 | PR 15.0, n=180 | C | 0.003 | 1 | 400 | [0.000, 0.014] |
| E2 | exp PR 4.5 | C | 0.015 | 6 | 400 | [0.007, 0.032] |
| E2 | few_spikes PR 4.5 | C | 0.005 | 2 | 400 | [0.001, 0.018] |
| E2 | few_spikes PR 6.0 | C | 0.005 | 2 | 400 | [0.001, 0.018] |
| E3 | N2 matched (= C) | matched | 0.018 | 7 | 400 | [0.009, 0.036] |
| E3 | N6 matched (angular Gaussian) | matched | 0.022 | 9 | 400 | [0.012, 0.042] |
| E4 | K = {2} | M | 0.013 | 5 | 400 | [0.005, 0.029] |
| E4 | K = {2} | C | 0.015 | 6 | 400 | [0.007, 0.032] |
| E4 | every K, C arm | C | 0.010-0.022 | 4-9 | 400 | widest [0.004, 0.042] |
| E5 | Gaussian + reprojection (C) | C-like | 0.020 | 8 | 400 | [0.010, 0.039] |
| E5 | random rotation | C-like | 0.022 | 9 | 400 | [0.012, 0.042] |
| E6 | Calinski-Harabasz | M | 0.018 | 7 | 400 | [0.009, 0.036] |
| E6 | Calinski-Harabasz | C | 0.022 | 9 | 400 | [0.012, 0.042] |
| E7 | k-means | C | 0.022 | 9 | 400 | [0.012, 0.042] |
| E7 | Gaussian mixture | C | 0.018 | 7 | 400 | [0.009, 0.036] |
| E7 | spectral | C | 0.022 | 9 | 400 | [0.012, 0.042] |
| E8 | none, all four arms | M/G/S/C | 0.003-0.005 | 1-2 | 399 | widest [0.000, 0.018] |
| E8 | z-score, all four arms | M/G/S/C | 0.005-0.015 | 2-6 | 399 | widest [0.003, 0.032] |
| E8 | L2 | C | 0.020 | 8 | 400 | [0.010, 0.039] |
| E9 | each of five replications | C | 0.007-0.017 | 2-5 | 300 | widest [0.002, 0.038] |
| E10 | d=32, n/d=0.24 | M and C | 0.000 | 0 | 133 | [0.000, 0.028] |
| E10 | d=64, n/d=0.24 | M | 0.008 | 3 | 396 | [0.003, 0.022] |

Two consequences worth stating plainly. First, **every headline number about the corrected construction in this paper is a low-event number**, including the one at the real fibromyalgia geometry; the pooled E9 estimate of 0.014 [0.009, 0.021] over 1,500 datasets is the tightest statement available and should be preferred to any single cell. Second, the E6 and E7 C-arm rows are all in this table, so the claim that the gap vanishes for non-silhouette statistics rests on M-arm rates that are themselves small; the direction of that result is clear but its magnitude is not tightly bounded.

---

## E13: why the correction reverses

**Question.** E2 showed the corrected construction is worse than the uncorrected one on single-dominant-eigendirection spectra, and E2's verification showed it is not a resampling artifact. Neither said why.

**Design.** The E3 decomposition run identically on three spectra tuned to the same measured sphere-PR ≈ 4.5, n = 28, d = 119, 400 datasets × 200 draws each: exp (no reversal), one_spike and two_block (reversal). Six null families, both selection rules, same observed statistic within each dataset.

| spectrum | observed | ambient null mean (sd) | sphere null mean (sd) | re-projection shift | M | C |
|---|---|---|---|---|---|---|
| exp | 0.2414 | 0.2133 (0.0357) | 0.2217 (0.0334) | **+0.0084** | 0.113 | 0.007 |
| one_spike | 0.3151 | 0.2409 (0.0451) | 0.2324 (0.0426) | **-0.0085** | 0.422 | 0.650 |
| two_block | 0.3079 | 0.2367 (0.0444) | 0.2288 (0.0422) | **-0.0079** | 0.383 | 0.605 |

Selection rule, N1 fixed → N1 matched: exp 0.113 → 0.015; one_spike 0.422 → 0.453; two_block 0.383 → 0.412.
Under C's geometry the rule is irrelevant on the reversing spectra: N2 fixed and N2 matched are identical at 0.6500 and 0.6050.

**Interpretation.** The sign of the re-projection shift flips and predicts the reversal. Re-projection is not intrinsically conservative; it moves the null mean toward whatever the sphere imposes on that spectrum. Graded decay: the sphere raises the null silhouette, the null is harder to beat, C is conservative. Single dominant eigendirection: the sphere lowers it, the null is easier to beat, C over-rejects. The shift is 0.008 either way against a null sd of 0.033-0.045, so a quarter-standard-deviation move separates a rejection rate of 0.007 from one of 0.650. Two supporting facts: the k-search correction is inert on the reversing spectra, because the projection concentrates near two antipodal poles and k = 2 wins on every draw; and the observed statistic is what moves across spectra (0.315 and 0.308 against 0.241) while the null means barely differ (0.213-0.241). This does not locate the boundary in spectrum space, three shapes at one PR, one n and one d do not determine a threshold.

---

## E14: resolving the whitening discrepancy

**Question.** E8's ZCA and PCA whitening rows disagree where a rotation argument says they cannot. Is that numerical?

**Design.** 200 datasets at the same geometry. Compare the pairwise-distance matrices the two transforms produce on identical input; report the retained rank and the condition number of the retained spectrum; re-run ZCA with the singular-value cutoff tightened over four orders of magnitude.

| cutoff | retained rank | condition number (median) | ZCA vs PCA distance deviation (median / max) |
|---|---|---|---|
| default | 27.0 | 4.03 × 10³ | 5.34 × 10⁻¹⁴ / 2.45 × 10⁻¹³ |
| 1e-12 … 1e-6 | 27.0 | 4.03 × 10³ | 5.34 × 10⁻¹⁴ / 2.45 × 10⁻¹³ |

**Interpretation.** **The numerical-conditioning hypothesis is wrong.** The two transforms agree to machine precision, the rank is 27 in every dataset, and the condition number is only 4 × 10³. Direct instrumentation gives the real cause: whitening equalizes the spectrum by construction, so 28 points with identity covariance in a 27-dimensional row space are nearly equidistant and the observed silhouette is **0.0000 to machine precision**, while the re-projected null sits at -0.05 to -0.125. The test compares a zero against a small negative number and rejects. The ZCA/PCA difference is which datasets yield any guard-passing k (386 against 352 of 400) plus the sign of a machine-epsilon quantity. Whitening at n < d removes the structure being tested; neither row measures calibration, and both are reported only as a warning.

---

## E6 and E7 re-run at 1,200 datasets

E6 and E7 were re-executed with `n_datasets` tripled so that no arm rests on fewer than ten events. The 400-dataset results are retained as `e6_statistics_n400.json` and `e7_algorithms_n400.json`.

| statistic | M [95% CI] | M events | C [95% CI] | C events | gap | M-only | C-only |
|---|---|---|---|---|---|---|---|
| silhouette | 0.1792 [0.159, 0.202] | 215/1200 | 0.0200 [0.013, 0.030] | 24/1200 | **+0.159** | 193 | 2 |
| Calinski-Harabasz | 0.0250 [0.018, 0.035] | 30/1200 | 0.0200 [0.013, 0.030] | 24/1200 | +0.005 | 23 | 17 |
| -Davies-Bouldin | 0.0450 [0.035, 0.058] | 54/1200 | 0.0500 [0.039, 0.064] | 60/1200 | -0.005 | 21 | 27 |
| -within/between | 0.0267 [0.019, 0.037] | 32/1200 | 0.0525 [0.041, 0.067] | 63/1200 | **-0.026** | 24 | 55 |

Every gap sign and magnitude agrees with the 400-dataset run. One result separates from noise at the larger sample that did not before: under the within/between dispersion ratio the corrected construction is **significantly worse** than the uncorrected one on the exponential spectrum, 0.0525 against 0.0267 with non-overlapping intervals and 55 corrected-only rejections against 24. The reversal is therefore not confined to concentrated spectra, changing the statistic is sufficient to produce it.

E7 at 1,200 datasets:

| algorithm | M [95% CI] | M events | C [95% CI] | C events | gap | fit failures | M-only | C-only |
|---|---|---|---|---|---|---|---|---|
| k-means | 0.1717 [0.151, 0.194] | 206/1200 | 0.0192 [0.013, 0.029] | 23/1200 | +0.153 | 0 | 185 | 2 |
| Ward | 0.1158 [0.099, 0.135] | 139/1200 | 0.0258 [0.018, 0.036] | 31/1200 | +0.090 | 0 | 125 | 17 |
| Gaussian mixture | 0.1742 [0.154, 0.197] | 209/1200 | 0.0183 [0.012, 0.028] | 22/1200 | +0.156 | 0 | 189 | 2 |
| spectral | 0.1292 [0.111, 0.149] | 155/1200 | 0.0175 [0.011, 0.027] | 21/1200 | +0.112 | 0 | 149 | 15 |

Every gap sign and ordering reproduces. No Gaussian-mixture fit failed at `reg_covar` = 1e-4 across approximately 1.2 million fits. The corrected-only counts separate by algorithm at this sample size, 17 (Ward) and 15 (spectral) against 2 and 2, which the 400-dataset run could not resolve; reported, not interpreted.

**Low-event audit, updated.** After the E6 and E7 re-runs, no arm in either experiment rests on fewer than 21 events. The remaining low-event rates are in E1, E2, E3, E4, E5, E8, E9 and E10 as listed above; the pooled E9 estimate of 0.014 [0.009, 0.021] over 1,500 datasets remains the tightest available statement about the corrected arm at the fibromyalgia geometry.

---

## E15: where the re-projection shift changes sign

**Question.** E13 gave the mechanism but used three spectra, so it located no boundary. Where does the sign change, and can it be predicted from something an analyst can compute?

**Design.** A one-parameter family λ₀ = S, λᵢ = exp(-i/τ), with τ bisected per cell on **200** throwaway samples (not the 20 used elsewhere, a 20-sample estimator left a systematic offset) so measured sphere-PR is held fixed while concentration varies. n = 28, d = 119, 12 cells, 400 datasets × 200 draws. Across all cells PR spans 4.27-4.68, mean 4.49, with no trend in S.

| S | τ | PR | PC1 | shift (sd) | M | C | M-only | C-only |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.95 | 4.36 | 0.363 | +0.27 | 0.175 | 0.022 | 62 | 1 |
| 1.5 | 2.48 | 4.66 | 0.361 | +0.15 | 0.117 | 0.025 | 39 | 2 |
| 2 | 2.48 | 4.27 | 0.399 | +0.14 | 0.198 | 0.092 | 48 | 6 |
| 3 | 3.16 | 4.37 | 0.409 | +0.05 | 0.170 | 0.150 | 24 | 16 |
| **4** | **4.02** | **4.68** | **0.402** | **-0.02** | **0.165** | **0.165** | **13** | **13** |
| 6 | 5.11 | 4.64 | 0.416 | -0.05 | 0.223 | 0.305 | 7 | 40 |
| 9 | 6.50 | 4.48 | 0.436 | -0.08 | 0.307 | 0.475 | 8 | 75 |
| 14 | 9.34 | 4.60 | 0.438 | -0.12 | 0.380 | 0.487 | 8 | 51 |
| 22 | 13.40 | 4.51 | 0.451 | -0.14 | 0.472 | 0.613 | 3 | 59 |
| 40 | 21.70 | 4.34 | 0.465 | -0.16 | 0.527 | 0.708 | 1 | 73 |
| 80 | 44.72 | 4.47 | 0.462 | -0.16 | 0.525 | 0.700 | 4 | 74 |
| 160 | 189.91 | 4.52 | 0.460 | -0.18 | 0.495 | 0.690 | 3 | 81 |

**Interpretation.** Three distinct signatures change at the same point. The shift crosses zero between S = 3 and S = 4 (S ≈ 3.7, PC1 ≈ 0.40). The rejection rates cross there: C below M at S = 3, equal at S = 4, above from S = 6. The discordance reverses there: 24 vs 16 at S = 3, exactly 13 vs 13 at S = 4, 7 vs 40 at S = 6. The three are causally linked but separately computed, and their agreement is what turns the mechanism into a usable rule.

Applied to the real embeddings: fibromyalgia trained sits at **PC1 = 0.326** (PR 4.37), conservative side, margin ≈ 0.08; depression trained at 0.176 (PR 8.62), conservative, wide margin; fibromyalgia untrained at 0.784 (PR 1.59), past the crossing, but at a PR where both constructions fail anyway. The paper's corrected p of 0.7009 is therefore a conservative reading, which is the direction that supports its negative conclusion.

Caveat: one PR, one n, one d, one family. PC1 is not monotone in S along the sweep (τ moves to hold PR), so S is the controlled variable and PC1 the observable proxy.

An earlier execution with 20-sample tuning is retained as `e15_boundary_tune20.json`. Its cell PR ran 4.64-5.03 with an upward trend in S; the 200-sample re-tune removed both. The drift ran *against* the effect, higher PR means lower rejection, so it never threatened the conclusion.

---

## E16: verifying the within/between reversal

**Design.** Same protocol as the E2 verification: four seeds, then every discordant dataset re-tested at 2,000 draws. Davies-Bouldin carried as a control.

| statistic | pooled M [95% CI] | pooled C [95% CI] | M-only | C-only | persists at 2,000 draws |
|---|---|---|---|---|---|
| within/between | 0.0275 [0.0205, 0.0367] | **0.0581 [0.0477, 0.0707]** | 29 | **78** | 63 of 78 (**80.8%**) |
| -Davies-Bouldin | 0.0431 [0.0342, 0.0542] | 0.0475 [0.0381, 0.0590] | 33 | 40 | 28 of 40 (70.0%) |

**Interpretation.** The within/between reversal is real and is the most robust in this run, non-overlapping pooled intervals, a 78-to-29 asymmetry, and four fifths of discordances surviving a tenfold draw increase, higher than any concentrated-spectrum cell. The control forces a distinction that had been implicit: **persistence and reversal are different properties.** Davies-Bouldin's discordances also persist (70%), so they are not granularity artifacts, but its counts are 40 vs 33 with overlapping rates, genuine two-way disagreement, no directional failure. Persistence establishes that discordant orderings are real; only the *asymmetry* plus non-overlapping rates establishes that the correction is doing harm.

---

## E17: is consensus stability an artifact of averaging?

**Question.** E11 found the consensus far more stable than the individual runs. RESULTS.md and the paper both asserted that this is what averaging does regardless of structure. That assertion was untested.

**Design.** Permute each run's own label vector, preserving its cluster count and sizes exactly, destroying all cross-run agreement, rebuild the consensus, recompute bootstrap stability. 300 surrogates per protocol, 100 bootstrap resamples each.

| protocol | observed | surrogate mean ± sd | surrogate 95th | surrogate max | excess | p |
|---|---|---|---|---|---|---|
| fixed architecture | 0.6770 | 0.1307 ± 0.0196 | 0.1562 | 0.1819 | **+0.5463** | 0.0033 |
| search-varying | 0.6099 | 0.1307 ± 0.0178 | 0.1570 | 0.1750 | **+0.4792** | 0.0033 |

**Interpretation. The earlier assertion was wrong.** Averaging 20 mutually unrelated partitions gives bootstrap stability of 0.13, not 0.61-0.68. Neither observed value falls inside its surrogate distribution, both exceed the surrogate maximum by more than 0.4, so p = 0.0033 is the floor set by 300 surrogates, not a measured tail. The runs share a reproducible component and consensus recovers it.

What it does not establish: the surrogate destroys cross-run agreement, so the test asks whether the runs agree more than chance, not whether the consensus corresponds to subgroups in the data. No experiment here subjects a consensus partition to a data-level null. The individual runs still agree with the consensus at only 0.37, and its k of 7 and 5 matches no protocol's modal per-run selection.

---

## E18: does the sign change transfer to other geometries?

**Question.** E15 located the re-projection sign change at one participation ratio, one n and one d. Does it sit at the same place elsewhere?

**Design.** 189 cells: target PR {3.0, 4.5, 6.0} x n {28, 60, 119} x d {64, 119, 256} x spike height {1, 2, 3, 4, 6, 9, 14}, 150 datasets x 100 draws each, tau bisected per cell on 100 samples. The shift averages 15,000 silhouettes per arm, so it needs fewer datasets than a rejection rate.

| target PR | n | crossing S (d = 64 / 119 / 256) | PC1 at crossing |
|---|---|---|---|
| 3.0 | 28, 60, 119 | none within S <= 14 | n/a |
| 4.5 | 28 | 4.47 / 3.53 / 4.49 | 0.426 / 0.406 / 0.429 |
| 4.5 | 60 | none within S <= 14 | n/a |
| 4.5 | 119 | none within S <= 14 | n/a |
| 6.0 | 28 | 2.15 / 1.89 / 2.15 | 0.313 / 0.307 / 0.312 |
| 6.0 | 60 | 5.23 / 5.06 / 5.12 | 0.357 / 0.360 / 0.360 |
| 6.0 | 119 | 11.80 / 12.23 / 11.77 | 0.385 / 0.388 / 0.383 |

**Interpretation.** Three results separate.

The crossing is **invariant to dimension**. Across a fourfold range in d the crossing PC1 agrees to within 0.022 at worst and 0.003 at best. This is the one axis along which the failure does not move: rejection rates rise steeply with d at fixed PR (E10), but the point at which the correction changes sign does not.

It **moves with sample size**. At PR 6.0 the crossing runs from S = 2.06 at n = 28 to 5.14 at n = 60 and 11.93 at n = 119. A spectrum concentrated enough to reverse the correction at n = 28 leaves it conservative at n = 119.

It **moves with conditioning**. At n = 28 the crossing falls from S = 4.17 at PR 4.5 to S = 2.06 at PR 6.0, and at PR 3.0 there is no crossing anywhere in the range: the shift stays positive from S = 1 to S = 14. That is not a safe region, since E1 puts both constructions above 0.5 at PR 3.0.

**Consequence.** Across the twelve bracketed cells, PC1 at the crossing spans 0.307 to 0.429. The 0.40 measured in E15 is specific to its geometry and does not transfer. What transfers is the sign test itself: compute the null mean under an ambient draw and under a re-projected one and take the sign of the difference. That costs two null distributions at one k and is valid anywhere.
