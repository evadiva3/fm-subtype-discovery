# Superseded source

Only archived files that current code or documentation depends on are kept here.

- `pre_revision_20260923/` preserves the files replaced by the September 23 cleanup. In particular, its shape-aware drivers implement coordinate offsets after covariance rotation. Their historical results remain in `results/`. Active corrected drivers in `analysis/` use covariance eigenvectors and different output filenames. Do not mix the two designs.
- `pre_pink_20260923/paper/figures/` retains only the vector inputs read by `paper/build/pink_theme.py`: Figures 1, 5 and 6.
- `pre_baby_pink_20260923/experiments/ds002748_mdd/propagated_null/e11_consensus.py` is the consensus source before the minimum-cluster-size guard fix described in Section S6.2.