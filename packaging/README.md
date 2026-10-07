# nullcal

Covariance-preserving permutation nulls for cluster validation, with the two common procedure
mismatches made switchable so that their effect on a given representation can be measured rather
than assumed.

Reference implementation for Bangsil and Joshi, *Type I error of covariance-preserving cluster
nulls on learned representations*. Section numbers below refer to that manuscript.

## Background

Testing a silhouette score against a Gaussian null matched to the data's mean and covariance is
standard practice in unsupervised subtyping. Two choices in that test are easy to make wrongly,
and both inflate significance.

The first is geometric. Where embeddings are L2-normalized before clustering, the observed data
lies on the unit sphere while draws from the fitted Gaussian fall inside the ball, so the null
occupies a manifold the data cannot. The second is selective. Where the observed statistic is a
maximum over candidate cluster counts, each null draw must be granted the same search; scoring
every draw at the single count the observed data selected denies the null a search the observed
statistic performed.

Measured on 62,789 structureless datasets, the construction carrying both mismatches rejects at
20.5 percent against a nominal 5 percent at one representative geometry, and matching both
brings it to 0.4 percent there.

Matching is not thereby safe. On covariance spectra with a single dominant eigendirection, at
matched participation ratio, sample size and dimension, the matched construction rejects
structureless data roughly twice as often as the unmatched one. The same reversal appears on an
ordinary exponentially decaying spectrum when the silhouette is replaced by a within-between
dispersion ratio. Which regime a given analysis occupies depends on n, d, the eigenvalue
spectrum, the normalization and the statistic, so it has to be measured.

## Installation

```bash
pip install nullcal
```

Requires numpy and scikit-learn.

## Usage

The four constructions are selected by two booleans, `reproject` and `match_selection`. `ladder`
evaluates all four on one embedding and reports the four *p*-values, the four null means and the
null's own selected-*k* distribution side by side. A large spread across the four indicates that
the representation lies in the region where the two mismatches bite.

```python
import nullcal as nc

result = nc.ladder(X, n_draws=20000)   # X is (n, d)
print(result.table())
```

`calibrate_paired` measures the realized Type I error of two constructions on data drawn from
the null the test itself assumes, paired on identical datasets and draw seeds so that the
discordance counts are interpretable in both directions.

```python
cal = nc.calibrate_paired(X, n_datasets=200, n_draws=200)
print(cal.summary)
```

`type1_surface_cell` reproduces any cell of the Type I error surface at a supplied sample size,
dimension and spectrum, which locates a given geometry on that surface without requiring the
imaging stack.

```python
cell = nc.type1_surface_cell(n=28, d=119, tau=3.0, n_datasets=120, n_draws=200)
print(cell["rej_M"], cell["rej_C"], cell["pr_mean"])
```

## Before interpreting a p-value

These five steps condense the recommendations of Section 7.2 to those this module implements.

1. Report the participation ratio of the representation being clustered, computed on the sphere
   where it is normalized, together with n and d. The participation ratio is not sufficient on its
   own: at one matched value the matched construction's rejection rate spans 0.005 to 0.657 across
   spectrum families (Section 4.3).
2. Report the fraction of variance on the leading eigenvector alongside it. At fixed
   participation ratio this is what orders the failure. The point at which matching changes sign is
   invariant to dimension but moves with sample size and conditioning, and the leading fraction at
   that point ranges from 0.307 to 0.429 across the geometries mapped in Section 4.4, so it should
   be re-measured rather than assumed to transfer.
3. Check the sign of the re-projection shift before trusting the matched construction. Compute
   the null mean under an ambient draw and under a re-projected one at the observed *k*, and take
   the sign of the difference. A negative sign means re-projection has made the null easier to
   beat, so the matched construction is the more anti-conservative of the two.
4. Report the effective size of the *k*-search family rather than the nominal range. Under a
   minimum-cluster-size guard the two diverge sharply, and Type I error tracks the effective count
   (Section 4.8).
5. Generate structureless data at the analysis's own spectrum and measure the rejection rate of
   the intended test. Do not assume the matched construction is conservative; run both
   constructions paired on the same datasets and report the discordance in both directions, since a
   matched-only rejection is informative about the spectrum rather than necessarily about the
   implementation.

## API

| function | purpose |
|---|---|
| `ladder(X, ...)` | all four constructions on one embedding |
| `null_test(X, reproject=, match_selection=, ...)` | a single construction |
| `calibrate_paired(X, ...)` | realized Type I error of two constructions, paired |
| `type1_surface_cell(n, d, tau, ...)` | one cell of the Type I error surface |
| `best_silhouette(X, ...)` | maximum silhouette over k under the guard |
| `participation_ratio(X, normalize=)` | effective dimensionality |
| `separation_ratio(X, labels)` | detection floor in transferable units |
| `min_cluster_guard(n)` | the guard used throughout |
| `propagated_null_checklist()` | requirements for a null simulated in raw data space |

The *p*-value estimator is `(c + 1) / (B + 1)` under a `>=` comparison and cannot return zero.
The minimum-cluster-size guard is `max(4, round(0.15 n))`, applied identically to the observed
statistic and to every null draw. The Gaussian is drawn through an SVD factorization, so the
rank-deficient case at n < d is handled without regularization.

## Scope

The fifth construction described in the manuscript, a null simulated in raw data space and
propagated through the entire analysis pipeline, cannot be packaged generically because it
requires the pipeline it is propagated through. `propagated_null_checklist()` states the
conditions such a null must satisfy, including the forward-path validation without which the
null traverses a different pipeline than the observed data.

Every Type I error rate this module measures is defined under a single-component Gaussian null,
which is the hypothesis the covariance-preserving construction implements rather than an
unrestricted notion of absent cluster structure. Calibration under that hypothesis does not
establish calibration under unimodal non-Gaussian alternatives, which are untested.

## License

MIT.
