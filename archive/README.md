# Superseded source

Archived files that current code or the supplement still uses.

- `recorded_source/` has the exact drivers that wrote the stored records. The current copies differ only in the results paths, which changed when the results folders were renamed. `python reproduce.py verify` checks the stored checksums against these files. Git keeps their bytes unchanged (`.gitattributes`), so the checksums also match on other systems.
- `shape_aware_before_fix/` has the shape-aware drivers replaced on September 23. They apply coordinate offsets after covariance rotation. Their results are in `results/exploratory/controls/`. The corrected drivers in `analysis/` use covariance eigenvectors and write to other file names. Do not mix the two designs.
- `figure_inputs/` has the vector inputs used by `paper/build/pink_theme.py` (Figures 1, 5 and 6).
- `e11_consensus_before_guard_fix.py` is the consensus source from before the minimum-cluster-size guard fix in Section S6.2.
