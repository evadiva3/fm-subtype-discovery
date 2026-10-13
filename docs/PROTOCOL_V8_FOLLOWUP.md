# Separate follow-up after seeing the primary power results

Recorded 2026-09-15, after the primary v8 study completed and before this follow-up
was executed. This is an explicitly outcome-motivated follow-up, not part of the
original prospective protocol and not a public preregistration. Preserve all primary
results. The primary W silhouette procedure detected the delta=4 mixture only 4.5%
and 6.5% of the time at q=5 and q=10, respectively.

Question: does replacing silhouette with a fourth-moment discrepancy recover useful
power while using the same fitted angular reference?

For Tyler-whitened and normalized rows w_i, define
D4 = mean_{i,j}[(w_i^T w_j)^4] - 3/[q(q+2)], including diagonal pairs. This is the
squared Frobenius discrepancy between the empirical fourth-moment tensor and the
uniform-sphere fourth-moment tensor. The uniform cross term is constant because
every w_i has unit norm. It is rotation-invariant, so the same shape-invariance
argument applies when the reference samples repeat Tyler fitting. Its upper-tail
test is not claimed to be omnibus: it targets fourth-order angular departures.
No new fourth-moment or spherical goodness-of-fit theory is claimed.

Generate 200 new independent n=60,d=20 datasets at each delta=0,2,4 with the same
equiprobable +/-delta location-mixture law. Use a separate experiment address 11
for observed and reference random streams, so none reuse primary observations.
Keep the original mixture projection matrices fixed for comparability (experiment
8, projection id 100, q=5/10). Evaluate D4 with 199 fully refitted independent
isotropic references. This is 600 new datasets and 1,200 projected analyses.

Report all six cells, pointwise exact binomial intervals, explicit fit failures,
source/data hashes and independently addressed reference draws. Compare with the
primary W rates descriptively, identifying that the two studies use independent
samples. Do not select a favorable delta, q or kernel after seeing these outcomes.
Do not change the primary method, rename this follow-up as prespecified primary
work, or use it to erase the primary low-power finding.

Audit summaries and dataset identities, and fully replay one record per cell. Unit
checks cover the tensor identity, orthogonal/affine invariance and finite-simulation
p-value grid. No applied p-values or clinical claims will be selected from this test.
