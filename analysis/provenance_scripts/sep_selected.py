from __future__ import annotations
import os
for v in("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[v]="1"
from _paths import CAN as CANON,out,detection_curve
import json
import numpy as np,pandas as pd

OUT=out("audit_gap_fills")
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from config import config

KR=list(config.kmeansKRange)
MS=max(config.minClusterSizeFloor,round(config.minClusterSizeFraction*config.subjectAmt))

def norm(x):
    return x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-8)

def ratio(x,lab):
    us=np.unique(lab)
    if len(us)<2:return np.nan
    cent=np.array([x[lab==u].mean(axis=0) for u in us])
    d=[np.linalg.norm(cent[i]-cent[j]) for i in range(len(us)) for j in range(i+1,len(us))]
    rad=np.concatenate([np.linalg.norm(x[lab==u]-cent[i],axis=1) for i,u in enumerate(us)])
    return float(np.mean(d)/np.sqrt((rad**2).mean()))

def select(x,rs=0):
    sils,mins,labs=[],[],[]
    for k in KR:
        lb=KMeans(n_clusters=k,n_init=config.kmeansNInit,random_state=rs).fit_predict(x)
        sils.append(silhouette_score(x,lb));mins.append(int(np.bincount(lb).min()));labs.append(lb)
    ok=[j for j in range(len(KR)) if mins[j]>=MS]
    b=max(ok,key=lambda j:sils[j]) if ok else int(np.argmax(sils))
    return labs[b],KR[b],sils[b]

emb=np.load(config.real2Synth)
rng=np.random.default_rng(config.synthSeedG)
groups=[g[:] for g in config.presetGroups] if config.usePresets else []
while len(groups)<config.numRandGroups:
    c=np.diff(np.insert(np.sort(rng.choice(np.arange(1,config.subjectAmt),
              size=config.clustersPerGroup-1,replace=False)),
              [0,config.clustersPerGroup-1],[0,config.subjectAmt])).tolist()
    if min(c)>=MS:groups.append(c)

rows=[]
for group in groups:
    for run in range(config.runsPerDelta):
        r=np.random.default_rng(config.synthSeedG+run)
        Q,_=np.linalg.qr(r.standard_normal((emb.shape[1],len(group))),mode="reduced")
        pi=r.permutation(len(emb));un=np.argsort(pi)
        ge=np.split(emb[pi],np.cumsum(group)[:-1])
        sig=[np.std(emb@Q[:,i]) for i in range(len(ge))]
        true=np.repeat(np.arange(len(group)),group)[un]
        for delta in config.deltas:
            p=np.concatenate([ge[i]+delta*sig[i]*Q[:,i].reshape(1,-1)
                              for i in range(len(ge))],axis=0)[un]
            pn=norm(p)
            lb,ks,sl=select(pn,rs=0)
            rows.append({"group":"/".join(map(str,group)),"run":run,"delta":delta,
                         "ratio_selected":ratio(pn,lb),"ratio_true":ratio(pn,true),
                         "k_selected":ks,"silhouette":sl})

df=pd.DataFrame(rows)
df.to_csv(OUT/"separation_ratio_selected_full.csv",index=False)
agg=df.groupby("delta").agg(ratio_sel_mean=("ratio_selected","mean"),
                            ratio_sel_sd=("ratio_selected","std"),
                            ratio_true_mean=("ratio_true","mean"),
                            sil_mean=("silhouette","mean")).reset_index()
dc=pd.read_csv(detection_curve())
agg=agg.merge(dc[["delta","detect_rate_p<.05","mean_ARI"]],on="delta",how="left")
agg.to_csv(OUT/"detection_floor_separation_units_selected.csv",index=False)

cemb=np.load((CANON/"Embeddings.npy")).astype(float)
clab=pd.read_csv((CANON/"K-Means-Labeling.csv")).sort_values("Subject_Id")["Label"].to_numpy()
obs=ratio(norm(cemb),clab)
obs_sil=silhouette_score(norm(cemb),clab)

print("\ndetection floor vs observed separation (selected labels)")
print(f"  {'delta':>6}{'ratio (selected)':>22}{'ratio (true)':>14}{'silhouette':>12}"
      f"{'power':>9}{'mean ARI':>10}")
for _,r in agg.iterrows():
    print(f"  {r['delta']:>6.1f}{r['ratio_sel_mean']:>15.3f} +/-{r['ratio_sel_sd']:>5.3f}"
          f"{r['ratio_true_mean']:>14.3f}{r['sil_mean']:>12.4f}"
          f"{r['detect_rate_p<.05']:>9.3f}{r['mean_ARI']:>10.3f}")
print()
print(f"  OBSERVED canonical cohort: separation ratio{obs:.3f} silhouette{obs_sil:.4f}")
print("\n  selected-label ratio isn't comparable to observed (circular at delta=0, dips before climbing)")
print("  true-label ratio is monotone, use that:")
y=agg["detect_rate_p<.05"].to_numpy();rt=agg["ratio_true_mean"].to_numpy()
first=agg[agg["detect_rate_p<.05"]>0].iloc[0]
p80=float(np.interp(0.80,y,rt))
print(f"    random group assignment on these embeddings   ratio{rt[0]:.3f}  (power 0)")
print(f"    first non-zero power (delta{first['delta']:.0f})           "
      f"ratio{first['ratio_true_mean']:.3f}  (power{first['detect_rate_p<.05']:.3f})")
print(f"    80% power                                     ratio{p80:.3f}")
json.dump({"observed_ratio_selected_NOT_COMPARABLE":obs,
           "observed_silhouette":float(obs_sil),
           "no_structure_true_label_ratio":float(rt[0]),
           "ratio_at_first_nonzero_power":float(first["ratio_true_mean"]),
           "ratio_at_80pct_power":p80,
           "by_delta":agg.to_dict(orient="records")},
          open(OUT/"detection_floor_vs_observed.json","w"),indent=2)
print(f"  wrote {OUT}")
