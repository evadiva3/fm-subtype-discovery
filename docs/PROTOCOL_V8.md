# V8 research extension: protocol fixed before outcome inspection

Date: 2026-09-15. This local analysis plan was written prospectively; it is not a public preregistration.

## Question

Can a shape-aware, fully refitted reference improve the Gaussian-before-normalization
calibration problem, and what does it detect in alternatives and saved embeddings?

## Methods and estimands

Work in fixed, data-independent projections of dimension q=5 or q=10, with n>q.
Assign independent, reproducible seeds to the projection matrices. Normalize after
projection. Do not select projections by their p-values or fit PCA on the
observed data and then treat the projection as fixed.

Fit unregularized Tyler angular shape, trace normalized to q. Whiten by its inverse
square root, normalize each whitened row, and apply the same guarded k-means silhouette
search used in v7. Do not subtract the sample mean: the target is a zero-center angular
central Gaussian (ACG) family, not an unknown-location projected-normal family.

Compare four constructions within each projection:

- C: sample-mean/sample-covariance fitted Gaussian, reprojected and searched.
- T: fitted Tyler shape, zero-mean Gaussian, reprojected and searched; plug-in reference.
- W: Tyler-whitened observed statistic versus isotropic Gaussian directions that are
  themselves Tyler-fitted, whitened, normalized and searched.
- N: same whitened observed statistic versus isotropic directions without refitting.
  This is an explicit mismatch control, not a proposed valid method.

Tyler's affine equivariance implies that the distribution of the ideal W statistic is
independent of positive-definite ACG shape, provided the estimator exists and the
clustering statistic is orthogonally invariant. Verify this algebraically and numerically.
Record every computational failure. Do not silently discard failures or count them as successes.

## Calibration and power

Use the first 200 independent datasets of each of the seven v7 populations, with both
q values: 1,400 reused datasets, 2,800 projected analyses. Each uses 199 independent
reference draws. These analyses extend the previous evidence but do not add 2,800 independent
datasets to the original 2,800. Projection comparisons within a cell are paired.

Add a prospective power study: n=60, d=20, independent standard Gaussian noise, and
a balanced random two-component location mixture with shifts +/- delta along a fixed
axis, delta=0,2,4. Generate 200 independent datasets per delta and evaluate q=5 and 10.
Delta=0 is an additional null; delta>0 is an alternative to the ACG family, not a
'false-positive' population. Record power as departure detection, not biological
subtype identification. Do not choose mixture strength after viewing rejection results.

Report p <= .05, plus-one tails including ties, exact pointwise binomial intervals,
guard and fit failures, all cells and all arms. No multiplicity-adjusted claims are
based on scanning the smallest p-value. Treat changes to the k-means restart count as sensitivity
analysis: replay the first 50 datasets of exponential n=28 and one-spike n=28 at
n_init=20 in q=5, keeping 199 draws. This subset is fixed by index before inspection.

## Applied demonstration

Use saved trained and untrained FM embeddings and the canonical MDD embeddings. Also
include all saved MDD multirun embedding files found in the existing multirun_v2/by_run
directory, without selecting runs based on their results. Hash inputs and record
dimensions and uncentered directional means. Use 999 reference draws, both q values,
fixed projection seed, n_init=5. No subject identifiers or clinical labels are required.

These are conditional analyses of saved representations. Shared training on the same
participants can violate the independent-row ACG model. Rejecting ACG is not evidence
that patients have discrete clinically useful subtypes. This demonstration will not support claims about network training or biological
validation.

## Boundaries

Neither Tyler estimation nor distributional invariance is claimed as new theory.
Compare and cite established angular goodness-of-fit and invariant-testing literature.
The exact affine-invariance argument does not automatically extend to regularized
high-dimensional Tyler variants. Do not substitute them without saying so. Keep v7 results immutable.
No commit, push, external message, deposit or journal submission.
