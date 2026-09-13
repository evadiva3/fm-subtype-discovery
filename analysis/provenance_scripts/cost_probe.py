import os
for v in("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[v]="1"
from _paths import RUNS,CAN as CANON
import glob,time,json,warnings
warnings.filterwarnings("ignore")
import numpy as np,pandas as pd
from sklearn.metrics import adjusted_rand_score
from config import config
from analysis.evaluate import cluster_evaluate

ev=cluster_evaluate()

print("unit costs (single-threaded BLAS, this machine)")
emb=np.load((CANON/"Embeddings.npy")).astype(float)
lab=pd.read_csv((CANON/"K-Means-Labeling.csv"))["Label"].to_numpy()
print(f"  canonical embedding {emb.shape}, k={len(np.unique(lab))}, kRange={list(config.kmeansKRange)}")

for ms in(True,False):
    t=time.perf_counter();ev.perm(emb,lab,n_permutations=100,random_state=0,match_selection=ms)
    dt=(time.perf_counter()-t)/100
    tag="selection-matched" if ms else "fixed-k (misspecified)"
    print(f"  {tag:<24} {dt*1000:7.2f} ms/draw  ->  1k={dt*1000/60:5.2f} min   20k={dt*20000/60:6.1f} min")
    if ms:per_draw_sel=dt
    else:per_draw_fix=dt

print()
print("  derived costs for the audit items:")
print(f"    D.1  selected-k dist, 20,000 draws          {per_draw_sel*20000/60:6.1f} min")
print(f"    0.4  misspecified calib, 200 sets x200 draws {per_draw_fix*200*200/60:6.1f} min")
print(f"    2.1  calibration, 1,000 sets x1,000 draws    {per_draw_sel*1000*1000/3600:6.1f} h  (1 core)")
print(f"         ... on 16 cores                          {per_draw_sel*1000*1000/3600/16:6.1f} h")

print()
print("d.2 / todo 2.3: fm cross-run ari, corrected search space")
runs=[]
for d in sorted(glob.glob(str(RUNS/"run*"))):
    p=os.path.join(d,"outputs","K-Means-Labeling.csv")
    if os.path.exists(p):
        df=pd.read_csv(p).sort_values("Subject_Id")
        runs.append((os.path.basename(d),df["Subject_Id"].astype(str).tolist(),df["Label"].to_numpy()))
print(f"  runs with label vectors on disk: {len(runs)}")
ids0=runs[0][1]
assert all(r[1]==ids0 for r in runs),"subject order differs between runs"
print(f"  subject sets identical across all runs: True  (n={len(ids0)})")

ks=[len(np.unique(r[2])) for r in runs]
print(f"  selected k per run: {ks}")
print(f"  k distribution: {dict(sorted(pd.Series(ks).value_counts().items()))}")

pw=[adjusted_rand_score(runs[i][2],runs[j][2]) for i in range(len(runs)) for j in range(i+1,len(runs))]
pw=np.array(pw)
print(f"  pairwise ARI: n={len(pw)} pairs   mean {pw.mean():.4f}   sd {pw.std(ddof=1):.4f}"
      f"   range {pw.min():.4f} to {pw.max():.4f}   median {np.median(pw):.4f}")

rng=np.random.default_rng(0)
null=[]
for _ in range(2000):
    a=rng.permutation(runs[rng.integers(len(runs))][2]);b=rng.permutation(runs[rng.integers(len(runs))][2])
    null.append(adjusted_rand_score(a,b))
null=np.array(null)
print(f"  label-shuffled reference: mean {null.mean():.4f}   97.5th pct {np.percentile(null,97.5):.4f}")
print(f"  -> observed mean ARI is {pw.mean()-null.mean():+.4f} above the no-structure reference")

out=pd.DataFrame({"pairwise_ari":pw})
out.to_csv((RUNS/"crossrun_ari_pairwise.csv"),index=False)
json.dump({"n_runs":len(runs),"n_pairs":int(len(pw)),"mean_ari":float(pw.mean()),
           "sd_ari":float(pw.std(ddof=1)),"min":float(pw.min()),"max":float(pw.max()),
           "median":float(np.median(pw)),"k_per_run":ks,
           "null_mean_ari":float(null.mean()),"null_p975":float(np.percentile(null,97.5))},
          open((RUNS/"crossrun_ari_summary.json"),"w"),indent=2)
print(f"  written: {RUNS/'crossrun_ari_summary.json'}")
