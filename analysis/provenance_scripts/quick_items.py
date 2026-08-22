from __future__ import annotations
import os
for v in("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[v]="1"
from _paths import RP,CAN as CANON,out,detection_curve
import json,time,warnings
warnings.filterwarnings("ignore")
import numpy as np,pandas as pd

OUT=out("audit_gap_fills")
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from analysis.evaluate import cluster_evaluate
from config import config

OUT.mkdir(parents=True,exist_ok=True)
CAN=CANON
emb=np.load(CAN/"Embeddings.npy").astype(float)
lab=pd.read_csv(CAN/"K-Means-Labeling.csv").sort_values("Subject_Id")["Label"].to_numpy()
ev=cluster_evaluate()
NDRAWS=int(os.environ.get("NDRAWS",20000))

print(f"null's selected-k distribution ({NDRAWS:,} draws, canonical checkpoint)")
embN=emb/(np.linalg.norm(emb,axis=1,keepdims=True)+1e-8)
real=silhouette_score(embN,lab)
n=embN.shape[0]
ms=max(config.minClusterSizeFloor,round(config.minClusterSizeFraction*n))
kr=list(config.kmeansKRange)
print(f"  observed silhouette {real:.4f} at k={len(np.unique(lab))}; guard={ms}; k range {kr}")

rng=np.random.default_rng(config.randomSeed)
selK=np.zeros(NDRAWS,dtype=int); selS=np.zeros(NDRAWS)
guardFail=0
t0=time.perf_counter()
for i in range(NDRAWS):
    nl=ev._null_mvn(embN,rng)
    sils,mins=[],[]
    for k in kr:
        lb=KMeans(n_clusters=k,n_init=config.kmeansNInit,random_state=i).fit_predict(nl)
        sils.append(silhouette_score(nl,lb)); mins.append(int(np.bincount(lb).min()))
    ok=[j for j in range(len(kr)) if mins[j]>=ms]
    if not ok: guardFail+=1
    b=max(ok,key=lambda j:sils[j]) if ok else int(np.argmax(sils))
    selK[i]=kr[b]; selS[i]=sils[b]
el=(time.perf_counter()-t0)/60

cnt={int(k):int((selK==k).sum()) for k in kr}
pct={k:100*v/NDRAWS for k,v in cnt.items()}
print(f"  elapsed {el:.1f} min")
print(f"  {'k':>4}{'count':>9}{'percent':>10}")
for k in kr:
    print(f"  {k:>4}{cnt[k]:>9,}{pct[k]:>9.1f}%")
print(f"  draws where no k passed the guard: {guardFail:,} ({100*guardFail/NDRAWS:.2f}%)")
print(f"  null selected-silhouette: mean {selS.mean():.4f}  sd {selS.std(ddof=1):.4f}"
      f"  p95 {np.percentile(selS,95):.4f}")
p_sel=(int((selS>=real).sum())+1)/(NDRAWS+1)
print(f"  permutation p (selection-matched, {NDRAWS:,} draws)={p_sel:.4f}")
np.save(OUT/"null_selected_k.npy",selK); np.save(OUT/"null_selected_sil.npy",selS)

print()
print("detection floor in separation-ratio units")
wit=[]
for u in np.unique(lab):
    g=embN[lab==u]
    wit.append(np.linalg.norm(g-g.mean(axis=0),axis=1))
sd_within=float(np.concatenate(wit).std(ddof=1))
rms_within=float(np.sqrt((np.concatenate(wit)**2).mean()))
cent=np.array([embN[lab==u].mean(axis=0) for u in np.unique(lab)])
pd_=[np.linalg.norm(cent[i]-cent[j]) for i in range(len(cent)) for j in range(i+1,len(cent))]
print(f"  canonical geometry (unit-sphere embedding, n={n}, k={len(cent)}):")
print(f"    mean centroid separation      {np.mean(pd_):.4f}")
print(f"    within-cluster sd of radius   {sd_within:.4f}   (rms radius {rms_within:.4f})")
print(f"    observed separation ratio     {np.mean(pd_)/rms_within:.3f}")
pcp=detection_curve()
if pcp.exists():
    dc=pd.read_csv(pcp)
    dcol=[c for c in dc.columns if c.lower().startswith("delta")][0]
    rcol=[c for c in dc.columns if "reject" in c.lower() or "power" in c.lower()]
    print("\n  planted offset (delta) -> separation ratio, using rms within-cluster radius:")
    print(f"  {'delta':>7}{'sep ratio':>12}"+(f"{'power':>9}" if rcol else ""))
    rows=[]
    for _,r in dc.iterrows():
        d=float(r[dcol]); ratio=d/rms_within
        rows.append({"delta":d,"separation_ratio":ratio,
                     "power":float(r[rcol[0]]) if rcol else None})
        print(f"  {d:>7.1f}{ratio:>12.3f}"+(f"{float(r[rcol[0]]):>9.3f}" if rcol else ""))
    pd.DataFrame(rows).to_csv(OUT/"detection_floor_separation_units.csv",index=False)
else:
    print("  positive_control_detection_curve.csv not found; skipped")

print()
print("required_n_for_80_power for the clinical tests")
cv=pd.read_csv(RP/"results/clinical_validation_results.csv")
print(f"  {len(cv)} tests; columns: {list(cv.columns)}")
ecol=[c for c in cv.columns if "eps" in c.lower() or "effect" in c.lower()]
kcol=[c for c in cv.columns if c.lower() in("k","n_groups","groups")]
res=[]
for _,r in cv.iterrows():
    e2=float(r[ecol[0]]) if ecol else np.nan
    kg=int(r[kcol[0]]) if kcol else len(np.unique(lab))
    if not np.isfinite(e2) or e2<=0:
        res.append({"variable":r.iloc[0],"epsilon_sq":e2,"required_n_80":None,
                    "note":"effect <= 0; no n attains 80% power"})
        continue
    f2=e2/(1-e2)
    from scipy import stats
    lo,hi,need=kg+1,100000,None
    while lo<=hi:
        mid=(lo+hi)//2
        ncp=f2*mid
        crit=stats.f.ppf(0.95,kg-1,mid-kg)
        pw=1-stats.ncf.cdf(crit,kg-1,mid-kg,ncp)
        if pw>=0.80: need,hi=mid,mid-1
        else: lo=mid+1
    res.append({"variable":r.iloc[0],"epsilon_sq":e2,"required_n_80":need,"note":""})
rdf=pd.DataFrame(res)
print(rdf.to_string(index=False))
rdf.to_csv(OUT/"required_n_for_80_power.csv",index=False)

json.dump({"n_draws":NDRAWS,"selected_k_counts":cnt,"selected_k_percent":pct,
           "guard_fail":guardFail,"null_sil_mean":float(selS.mean()),
           "null_sil_sd":float(selS.std(ddof=1)),"observed_sil":float(real),
           "perm_p":p_sel,"elapsed_min":round(el,1),
           "mean_centroid_sep":float(np.mean(pd_)),"within_sd":sd_within,
           "within_rms":rms_within,"observed_separation_ratio":float(np.mean(pd_)/rms_within)},
          open(OUT/"audit_gap_fills_summary.json","w"),indent=2)
print(f"\n  wrote {OUT}")
