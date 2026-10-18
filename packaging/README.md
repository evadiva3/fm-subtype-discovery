# Statistical implementations

Install with `python -m pip install ./packaging` if you need the modules outside the repository. The drivers used in the paper can also load this local source directly. Use the pinned environment in `requirements/simulation-lock.txt` for the corrected runs.

| Module | Role |
|---|---|
| `nullcal` | Historical Gaussian cluster references |
| `nullcal_validation` | Diagnostic benchmark and known-generating-law controls |
| `angular_null` | Fixed projections, Tyler shape, whitening and refitted silhouette reference W |
| `angular_gof` | W, fourth-moment D4, kernel K and bootstrap E comparison |
| `angular_fourth`, `angular_balanced` | Earlier fourth-moment and kernel follow-up studies |

The ideal ACG argument assumes independent centered directions and requires the Tyler existence conditions to hold, with n greater than the projected dimension. Numerical shape fits report failures explicitly. An angular test does not by itself calibrate shared representation training or establish clinical subtypes. The manuscript describes the finite-sample evidence and the limits of these claims.

The historical protocols in `docs/PROTOCOL_*` and the source hashes in the result configurations identify earlier runs. Do not reformat or otherwise change numerical files with recorded hashes just to make the formatting consistent. Run `python reproduce.py verify` to check the evidence and `python reproduce.py test` for the tests.
