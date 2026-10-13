#S36 post hoc diagnostic: depression S and C p-values as a function of the observed searched statistic.
#Replicates cluster_evaluate.perm (searched modes, 20 restarts, seed config.randomSeed) but keeps every null statistic,
#so the p-value can be read off for the recomputed observed silhouette and for the stored canonical partition
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"
import argparse, hashlib, json, platform, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd, sklearn
R = Path(__file__).resolve().parents[1]
for p in (R, R / "analysis"):
    sys.path.insert(0, str(p))
from config import config
from evaluate import cluster_evaluate
EMB = R / "results/mdd/canonical_single/Embeddings.npy"
LAB = R / "results/mdd/canonical_single/K-Means-Labeling.csv"
DRAWS = 20000


def one(arm):
    config.kmeansNInit = 20
    ev = cluster_evaluate()
    x = np.load(EMB).astype(float)
    xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-08)
    rng = np.random.default_rng(config.randomSeed)
    null = []
    for _ in range(DRAWS):
        d = ev._null_mvn(xn, rng) if arm == "C" else ev._null_mvn_ambient(xn, rng)
        null.append(ev._select_best_silhouette(d, random_state=0, return_k=False))
    return arm, null


def main():
    a = argparse.ArgumentParser(); a.add_argument("--output", type=Path, required=True); a = a.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    config.kmeansNInit = 20
    ev = cluster_evaluate()
    x = np.load(EMB).astype(float); xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-08)
    lb = pd.read_csv(LAB).sort_values("Subject_Id")["Label"].to_numpy()
    obs = {"recomputed_best_silhouette": float(ev._select_best_silhouette(xn, random_state=0)), "stored_canonical_partition": float(ev.silhouette(xn, lb))}
    with Pool(2) as pool:
        res = dict(pool.map(one, ["C", "S"]))
    out = dict(study="S36 post hoc observed-statistic diagnostic", draws=DRAWS, restarts=20, observed=obs, stored_table_s64={"S": 0.2782, "C": 0.3843},
               p={arm: {k: (1 + int(np.sum(np.array(v) >= o))) / (DRAWS + 1) for k, o in obs.items()} for arm, v in res.items()},
               versions=dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__),
               source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    np.savez_compressed(a.output / "null_statistics.npz", **{k: np.array(v) for k, v in res.items()})
    (a.output / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["p"], indent=2), json.dumps(obs))


if __name__ == "__main__":
    main()
