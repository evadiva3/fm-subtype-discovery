#just a run file.
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

ROOT=Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT/"src", ROOT/"models", ROOT/"analysis", ROOT/"preprocessing"):
    sys.path.insert(0, str(p))

from config import config
import analysis.bootstrap_stability as bs

_orig_run_bootstrap=bs.run_bootstrap
_cap={}
def _capture(runner, emb, ids, *a, **k):
    _cap["n_fm"]=len(ids)
    return _orig_run_bootstrap(runner, emb, ids, *a, **k)
bs.run_bootstrap=_capture

conds=list(config.conditions)
rows, summ, blk=bs.run_real(conds)
print("=" * 70)
if blk is not None:
    print("blocked:", blk)
else:
    print(f"n (FM subjects)= {_cap.get('n_fm')}")
    print(f"n_resamples= {summ['n_resamples']}")
    print(f"ari mean= {summ['ari_mean']:.6f}")
    print(f"ari std= {summ['ari_std']:.6f}")
    print(f"ari min= {summ['ari_min']:.6f}")
    print(f"ari max= {summ['ari_max']:.6f}")
print("=" * 70)
