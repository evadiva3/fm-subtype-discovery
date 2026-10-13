#S36 amendment (post hoc): M and G without the fixed-k size guard; see docs/FOLLOWUP_PROTOCOL_S36_AMENDMENT_20261009.md.
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"
import argparse, hashlib, json, platform, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
import scipy, sklearn
R = Path(__file__).resolve().parents[1]
for p in (R, R / "analysis"):
    sys.path.insert(0, str(p))
from config import config
from evaluate import cluster_evaluate
from cohort_restart_check import COHORTS, STORED, MODES, DRAWS, sha

AMENDMENT = R / "docs/FOLLOWUP_PROTOCOL_S36_AMENDMENT_20261009.md"


def one(t):
    cohort, ninit, arm = t
    config.kmeansNInit = ninit
    emb, lab = COHORTS[cohort]
    x = np.load(emb).astype(float)
    lb = pd.read_csv(lab).sort_values("Subject_Id")["Label"].to_numpy()
    mg, ms = MODES[arm]
    t0 = time.time()
    p, kc = cluster_evaluate().perm(x, lb, n_permutations=DRAWS, match_geometry=mg, match_selection=ms, return_k_counts=True, guard_fixed=False)
    return dict(cohort=cohort, restarts=ninit, arm=arm, guard_fixed=False, p=float(p), null_selected_k_counts={int(a): int(b) for a, b in kc.items()}, seconds=time.time() - t0)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--workers", type=int, default=8)
    a = a.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    cfg = dict(study="S36 amendment (post hoc)", guard_fixed=False, arms=["M", "G"], draws=DRAWS, random_state=config.randomSeed, restarts=[20, 5],
               stored_table_s64=STORED, inputs={k: [sha(e), sha(l)] for k, (e, l) in COHORTS.items()}, amendment_sha256=sha(AMENDMENT),
               source_sha256={p: sha(R / p) for p in ("analysis/cohort_restart_check_amendment.py", "analysis/cohort_restart_check.py", "analysis/evaluate.py", "config.py")},
               versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__))
    (a.output / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    tasks = [(c, n, arm) for n in (20, 5) for c in COHORTS for arm in ("M", "G")]
    jp = a.output / "records.jsonl"
    with Pool(a.workers) as pool, jp.open("w", encoding="utf-8") as j:
        for r in pool.imap_unordered(one, tasks, chunksize=1):
            j.write(json.dumps(r) + "\n"); j.flush()
            print(json.dumps({k: r[k] for k in ("cohort", "restarts", "arm", "p", "seconds")}), flush=True)
    rows = [json.loads(l) for l in jp.read_text().splitlines()]
    out = dict(config=cfg, records_sha256=sha(jp), results={})
    for n in (20, 5):
        res = {}
        for c in COHORTS:
            for arm in ("M", "G"):
                p = next(r["p"] for r in rows if r["cohort"] == c and r["restarts"] == n and r["arm"] == arm)
                s = STORED[c][arm]; se = np.sqrt(s * (1 - s) / DRAWS)
                res[f"{c}_{arm}"] = dict(stored=s, recomputed=p, difference=p - s, mc_se=se, z=(p - s) / se)
        out["results"][f"restarts_{n}"] = dict(cells=res, within_3se=all(abs(v["z"]) <= 3 for v in res.values()))
    (a.output / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v["within_3se"] for k, v in out["results"].items()}))


if __name__ == "__main__":
    main()
