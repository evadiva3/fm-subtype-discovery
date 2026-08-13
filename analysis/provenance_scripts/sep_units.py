from __future__ import annotations
import json
from _paths import CAN as CANON,out,detection_curve
import numpy as np,pandas as pd

OUT=out("audit_gap_fills")
from config import config

OUT.mkdir(parents=True,exist_ok=True)

def geom(x,lab):
    us=np.unique(lab)
    cent=np.array([x[lab==u].mean(axis=0) for u in us])
    d=[np.linalg.norm(cent[i]-cent[j]) for i in range(len(us)) for j in range(i+1,len(us))]
    rad=np.concatenate([np.linalg.norm(x[lab==u]-cent[i],axis=1) for i,u in enumerate(us)])
    ssw=sum(((x[lab==u]-cent[i])**2).sum() for i,u in enumerate(us))
    dof=(len(x)-len(us))*x.shape[1]
    return float(np.mean(d)),float(np.sqrt((rad**2).mean())),float(np.sqrt(ssw/dof))

def norm(x):
    return x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-8)

emb=np.load(config.real2Synth)
print(f"planting source: {config.real2Synth}  shape {emb.shape}")

rng=np.random.default_rng(config.synthSeedG)
minSize=max(config.minClusterSizeFloor,round(config.minClusterSizeFraction*config.subjectAmt))
groups=[g[:] for g in config.presetGroups] if config.usePresets else []
while len(groups)<config.numRandGroups:
    cand=np.diff(np.insert(np.sort(rng.choice(np.arange(1,config.subjectAmt),
                 size=config.clustersPerGroup-1,replace=False)),
                 [0,config.clustersPerGroup-1],[0,config.subjectAmt])).tolist()
    if min(cand)>=minSize:
        groups.append(cand)
print(f"group configurations ({len(groups)}): {['/'.join(map(str,g)) for g in groups]}")

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
            planted=np.concatenate(
                [ge[i]+delta*sig[i]*Q[:,i].reshape(1,-1) for i in range(len(ge))],
                axis=0)[un]
            sa,ra,pa=geom(planted,true)
            sn,rn,pn=geom(norm(planted),true)
            rows.append({"group":"/".join(map(str,group)),"run":run,"delta":delta,
                         "sep_ambient":sa,"rms_within_ambient":ra,"ratio_ambient":sa/ra,
                         "pooled_sd_ambient":pa,"sep_norm":sn,"rms_within_norm":rn,
                         "ratio_norm":sn/rn})
df=pd.DataFrame(rows)
df.to_csv(OUT/"separation_ratio_by_delta_full.csv",index=False)

agg=df.groupby("delta").agg(
    ratio_ambient_mean=("ratio_ambient","mean"),ratio_ambient_sd=("ratio_ambient","std"),
    ratio_norm_mean=("ratio_norm","mean"),ratio_norm_sd=("ratio_norm","std")).reset_index()

pcp=detection_curve()
if pcp.exists():
    dc=pd.read_csv(pcp)
    dcol=[c for c in dc.columns if c.lower().startswith("delta")][0]
    pcol=[c for c in dc.columns if "reject" in c.lower() or "power" in c.lower()]
    agg=agg.merge(dc[[dcol]+pcol].rename(columns={dcol:"delta"}),on="delta",how="left")
agg.to_csv(OUT/"detection_floor_separation_units.csv",index=False)

cemb=np.load((CANON/"Embeddings.npy")).astype(float)
clab=pd.read_csv((CANON/"K-Means-Labeling.csv")).sort_values("Subject_Id")["Label"].to_numpy()
cs,cr,cp=geom(norm(cemb),clab)
print(f"\ncanonical observed (normalized, k=4): centroid sep {cs:.4f}, rms within {cr:.4f}, "
      f"SEPARATION RATIO {cs/cr:.3f}")

print("\ndetection floor in separation-ratio units")
hdr=f"  {'delta':>6}{'ratio (planted, normalized)':>30}{'ratio (ambient)':>18}"
if "power" in "".join(agg.columns).lower() or len(agg.columns)>5:
    extra=[c for c in agg.columns if "reject" in c.lower() or "power" in c.lower()]
else:
    extra=[]
print(hdr+(f"{extra[0]:>12}" if extra else ""))
for _,r in agg.iterrows():
    line=(f"  {r['delta']:>6.1f}{r['ratio_norm_mean']:>22.3f} +/-{r['ratio_norm_sd']:>5.3f}"
          f"{r['ratio_ambient_mean']:>18.3f}")
    if extra:
        line+=f"{r[extra[0]]:>12.3f}"
    print(line)
print(f"\n  canonical observed data sits at ratio {cs/cr:.3f}")
json.dump({"canonical_sep":cs,"canonical_rms_within":cr,"canonical_ratio":cs/cr,
           "by_delta":agg.to_dict(orient="records")},
          open(OUT/"separation_ratio_summary.json","w"),indent=2)
print(f"  wrote {OUT}")
