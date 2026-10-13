# Applied inputs

The synthetic studies do not use raw BOLD data. Their sole historical spectrum input is in `results/overnight/e2_spectrum_shape.json`.

Saved arrays under `outputs/` and `results/mdd/` support replay of downstream statistics. They do not replace raw imaging, all checkpoints or execution logs needed for end-to-end training reproduction. Subjects and clinical fields should be reviewed by the authors before any public redistribution.

Do not run `preprocessing/verify_setup.py` merely to inspect the data: it can move subject folders. `preprocessing/subject_filter.py` and the saved exclusion manifest describe the selection policy.
