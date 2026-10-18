# Superseded source

This folder keeps the archived files still needed by the current code or documentation.

- `pre_revision_20260923/` preserves the files replaced by the September 23 cleanup. Its shape-aware drivers apply coordinate offsets after covariance rotation. Their historical results remain in `results/`. The corrected drivers now in `analysis/` use covariance eigenvectors and write to different output filenames. Do not mix the two designs.
- `pre_pink_20260923/paper/figures/` keeps only the vector inputs used by `paper/build/pink_theme.py`: Figures 1, 5 and 6.
- `pre_baby_pink_20260923/experiments/ds002748_mdd/propagated_null/e11_consensus.py` contains the consensus source as it stood before the minimum-cluster-size guard fix described in Section S6.2.