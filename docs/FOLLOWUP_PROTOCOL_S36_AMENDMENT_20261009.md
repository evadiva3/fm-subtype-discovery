# Amendment to S36 (fixed-k constructions)

Added on 9 October 2026, after the S36 results for M and G had been seen. The original protocol (`docs/FOLLOWUP_PROTOCOL_S35_S36_20261008.md`) is unchanged and its records are retained and reported.

The S36 protocol specified the call used by `analysis/null_progression.py`, which runs `cluster_evaluate.perm` with its default `guard_fixed=True`; that applies the minimum-cluster-size guard to the fixed-k reference draws of M and G. Table 1 of the manuscript defines M and G as scoring each draw at the observed k without the size guard. The guarded runs therefore do not test the constructions reported in Table S64.

This amendment adds M and G with `guard_fixed=False` for both cohorts, with 20 and 5 restarts, 20,000 draws and the same seeds. Nothing else changes, including the reproduction criterion (every recomputed p-value within three Monte Carlo standard errors of Table S64). Because it was added after seeing results, this comparison is reported as post hoc.
