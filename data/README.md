# Applied inputs

The synthetic studies do not use raw BOLD data. Their only historical spectrum input is in `results/overnight/e2_spectrum_shape.json`.

The saved arrays in `outputs/` and `results/mdd/` let you replay the downstream statistics. Reproducing training from start to finish still requires the raw imaging, all checkpoints and execution logs. The authors should review the subjects and clinical fields before any public redistribution.

Do not use `preprocessing/verify_setup.py` just to inspect the data, because it can move subject folders. `preprocessing/subject_filter.py` and the saved exclusion manifest describe the selection policy.
