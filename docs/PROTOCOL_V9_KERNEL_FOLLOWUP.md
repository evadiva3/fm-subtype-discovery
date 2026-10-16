# Outcome-motivated even/odd kernel follow-up

Recorded before running this follow-up, while the primary v9 study was still running.
The completed two-mode n=240 cells showed D4 power .560/.295/.105 at q=5/10/20,
versus K .050/.045/.050. This contrast motivated the following explicitly separate
experiment. It is not part of the original v9 plan and is not publicly preregistered.

On unit directions, exp(-||u-v||^2)=exp(-2) exp(2 u^T v). Split it into the positive
definite even kernel exp(-2) cosh(2 u^T v) and odd kernel exp(-2) sinh(2 u^T v).
The even component targets antipodally symmetric discrepancies; the odd component
retains sensitivity to directional asymmetry. Their sum recovers the original kernel.
After Tyler whitening the second-moment term is fixed, so the leading variable
even terms begin at fourth order, while the odd component includes the empirical mean.
This provides a possible explanation for weak symmetric-mixture sensitivity of K;
it is not a claim that a finite sample always follows that approximation.

For each component use a separately ranked plus-one upper-tail p-value against the
same fully refitted isotropic reference samples. Define KB=min(1,2 min(p_even,p_odd)).
The union bound controls this two-component combination under the ideal pivot without
assuming component independence. No weights, bandwidths or thresholds are fitted to
outcomes. Gamma=1, B=199. Include D4 and original K on the same data for paired checks.
Do not claim a new kernel theory or universal power improvement.

Generate a completely new experiment-14 sample for the full original v9 grid:
six families, n=60/240, d=20, q=5/10/20, 200 datasets per source cell. Retain the same
fixed projections. This adds 2,400 independent source datasets and 7,200 projected
analyses. The new all-version source union becomes 8,800 unique datasets, subject to
hash audit. No source observations from earlier studies are reused.

Report every cell for D4, K, K-even, K-odd and KB, exact pointwise binomial intervals,
and paired changes between K and KB. Report failures explicitly. For use of all three
fixed dimensions, combine each method using min(1,3 min_q p_q); applying this to KB
accounts for both its two components and the three projections. It does not adjust
selection across arbitrary methods. No new applied p-value search is performed.

Audit all observed/projection/source hashes, verify no overlap with earlier studies,
recompute every summary, and fully replay one record per cell. Unit checks establish
the exact kernel decomposition, separate antipodal/sign behavior, affine invariance,
finite-simulation grid and the component Bonferroni rule. All primary v9 and earlier
results remain intact, regardless of the outcome of this follow-up.
