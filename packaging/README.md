# Statistical implementations

Install with `python -m pip install ./packaging` if you need the modules outside the repository. The paper drivers also load this local source directly. Use the pinned environment in `requirements/simulation-lock.txt` for the corrected runs.

| Module | Role |
|---|---|
| `nullcal` | Historical Gaussian cluster references |
| `nullcal_validation` | Diagnostic benchmark and known-generating-law controls |
| `angular_null` | Fixed projections, Tyler shape, whitening and refitted silhouette reference W |
| `angular_gof` | W, fourth-moment D4, kernel K and bootstrap E comparison |
| `angular_fourth`, `angular_balanced` | Earlier fourth-moment and kernel follow-up studies |

The ideal ACG argument assumes independent centered directions and Tyler existence conditions, with n greater than projected dimension. Numerical shape fits fail explicitly. An angular test does not by itself calibrate shared representation training or establish clinical subtypes. Refer to the manuscript for finite-sample evidence and scope.

Historical protocols in `docs/PROTOCOL_*` and source hashes in the result configurations identify earlier executions. Do not restyle or change hash-locked numerical files merely to make formatting consistent. Run `python reproduce.py verify` to check the evidence and `python reproduce.py test` for the tests.
