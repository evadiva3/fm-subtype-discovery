#S36: do the Table S64 cohort p-values depend on the k-means restart count? See docs/FOLLOWUP_PROTOCOL_S35_S36_20261008.md.
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

PROTOCOL = R / "docs/FOLLOWUP_PROTOCOL_S35_S36_20261008.md"
DRAWS = 20000
COHORTS = {
    "fibromyalgia": (R / "data/outputs/trained_fm_embeddings.npy", R / "data/outputs/K-Means-Labeling.csv"),
    "depression": (R / "results/mdd/canonical_single/Embeddings.npy", R / "results/mdd/canonical_single/K-Means-Labeling.csv"),
}
# Table S64 values (20,000 draws), and the construction flags used by analysis/null_progression.py
STORED = {"fibromyalgia": {"M": 0.1685, "G": 0.4630, "S": 0.4033, "C": 0.7009},
          "depression": {"M": 0.0508, "G": 0.1540, "S": 0.2782, "C": 0.3843}}
MODES = {"M": (False, False), "G": (True, False), "S": (False, True), "C": (True, True)}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def one(t):
    cohort, ninit, arm = t
    config.kmeansNInit = ninit
    emb, lab = COHORTS[cohort]
    x = np.load(emb).astype(float)
    lb = pd.read_csv(lab).sort_values("Subject_Id")["Label"].to_numpy()
    mg, ms = MODES[arm]
    t0 = time.time()
    p, kc = cluster_evaluate().perm(x, lb, n_permutations=DRAWS, match_geometry=mg, match_selection=ms, return_k_counts=True)
    return dict(cohort=cohort, restarts=ninit, arm=arm, p=float(p), null_selected_k_counts={int(a): int(b) for a, b in kc.items()}, seconds=time.time() - t0)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = a.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    cfg = dict(study="S36", draws=DRAWS, random_state=config.randomSeed, restarts=[20, 5], k_range=list(config.kmeansKRange),
               stored_table_s64=STORED, inputs={k: [sha(e), sha(l)] for k, (e, l) in COHORTS.items()}, protocol_sha256=sha(PROTOCOL),
               source_sha256={p: sha(R / p) for p in ("analysis/cohort_restart_check.py", "analysis/evaluate.py", "config.py")},
               versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__))
    (a.output / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    tasks = [(c, n, arm) for n in (20, 5) for c in COHORTS for arm in ("C", "S", "G", "M")]
    jp = a.output / "records.jsonl"
    done = {(r["cohort"], r["restarts"], r["arm"]) for r in map(json.loads, jp.read_text().splitlines())} if jp.exists() else set()
    with Pool(a.workers) as pool, jp.open("a", encoding="utf-8") as j:
        for r in pool.imap_unordered(one, [t for t in tasks if t not in done], chunksize=1):
            j.write(json.dumps(r) + "\n"); j.flush()
            print(json.dumps({k: r[k] for k in ("cohort", "restarts", "arm", "p", "seconds")}), flush=True)
    rows = [json.loads(l) for l in jp.read_text().splitlines()]
    summary = dict(config=cfg, records_sha256=sha(jp), results={})
    for n in (20, 5):
        res = {}
        for c in COHORTS:
            for arm in MODES:
                p = next(r["p"] for r in rows if r["cohort"] == c and r["restarts"] == n and r["arm"] == arm)
                s = STORED[c][arm]
                se = np.sqrt(s * (1 - s) / DRAWS)
                res[f"{c}_{arm}"] = dict(stored=s, recomputed=p, difference=p - s, mc_se=se, z=(p - s) / se)
        summary["results"][f"restarts_{n}"] = dict(cells=res, reproduces_table_s64=all(abs(v["z"]) <= 3 for v in res.values()))
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v["reproduces_table_s64"] for k, v in summary["results"].items()}))


if __name__ == "__main__":
    main()
