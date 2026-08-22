import json
import sys
from pathlib import Path
import numpy as np,pandas as pd

_R=Path(__file__).resolve().parents[1]
for _p in (_R,_R/"src",_R/"models",_R/"analysis"):
    if str(_p) not in sys.path:
        sys.path.insert(0,str(_p))

from config import config

o=config.resultsRoot

POWER_SRC=(config.resultsRoot/"handoff_20260802"/"audit_gap_fills"/
           "detection_floor_separation_units_selected.csv")
POWER_COL="detect_rate_p<.05"


def _emb():
    for c in (Path(config.real2Synth),_R/"data"/"outputs"/"trained_fm_embeddings.npy"):
        if Path(c).exists():
            return np.load(c).astype(float)
    raise FileNotFoundError("no embedding background")


def _norm(x):
    return x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-8)


def _rel(p):
    try:
        return str(Path(p).relative_to(_R))
    except ValueError:
        return str(p)


def ratio(x,lab):
    us=np.unique(lab)
    if len(us)<2:
        return float("nan")
    cent=np.array([x[lab==u].mean(axis=0) for u in us])
    d=[np.linalg.norm(cent[i]-cent[j])
       for i in range(len(us)) for j in range(i+1,len(us))]
    rad=np.concatenate([np.linalg.norm(x[lab==u]-cent[i],axis=1)
                        for i,u in enumerate(us)])
    return float(np.mean(d)/np.sqrt((rad**2).mean()))


def groups():
    rng=np.random.default_rng(config.synthSeedG)
    mn=max(config.minClusterSizeFloor,
           round(config.minClusterSizeFraction*config.subjectAmt))
    gl=[g[:] for g in config.presetGroups] if config.usePresets else []
    while len(gl)<config.numRandGroups:
        c=np.diff(np.insert(np.sort(rng.choice(np.arange(1,config.subjectAmt),
                  size=config.clustersPerGroup-1,replace=False)),
                  [0,config.clustersPerGroup-1],[0,config.subjectAmt])).tolist()
        if min(c)>=mn:
            gl.append(c)
    return gl


def by_delta(e):
    rows=[]
    for g in groups():
        for run in range(config.runsPerDelta):
            r=np.random.default_rng(config.synthSeedG+run)
            q,_=np.linalg.qr(r.standard_normal((e.shape[1],len(g))),mode="reduced")
            pi=r.permutation(len(e))
            un=np.argsort(pi)
            ge=np.split(e[pi],np.cumsum(g)[:-1])
            sg=[float(np.std(e@q[:,i])) for i in range(len(ge))]
            true=np.repeat(np.arange(len(g)),g)[un]
            for dl in config.deltas:
                p=np.concatenate([ge[i]+dl*sg[i]*q[:,i].reshape(1,-1)
                                  for i in range(len(ge))],axis=0)[un]
                rows.append({"group":"/".join(map(str,g)),"run":run,"delta":float(dl),
                             "ratio_true":ratio(_norm(p),true)})
    return pd.DataFrame(rows)


def floor_from_power(agg):
    if POWER_COL not in agg or agg[POWER_COL].isna().all():
        return None,("power column unavailable; run the positive-control detection "
                     f"curve and place it at {POWER_SRC}")
    a=agg.dropna(subset=[POWER_COL]).sort_values("delta")
    y=a[POWER_COL].to_numpy()
    rt=a["ratio_true_mean"].to_numpy()
    nz=a[a[POWER_COL]>0]
    if nz.empty:
        return None,"no delta reaches non-zero power in the supplied curve"
    return {
        "no_structure_true_label_ratio":float(rt[0]),
        "ratio_at_first_nonzero_power":float(nz.iloc[0]["ratio_true_mean"]),
        "delta_at_first_nonzero_power":float(nz.iloc[0]["delta"]),
        "ratio_at_80pct_power":float(np.interp(0.80,y,rt)),
    },None


def main():
    e=_emb()
    print(f"background {e.shape[0]}x{e.shape[1]}")
    df=by_delta(e)
    agg=(df.groupby("delta")
           .agg(ratio_true_mean=("ratio_true","mean"),
                ratio_true_sd=("ratio_true","std"))
           .reset_index())
    agg["onaxis"]=agg["delta"]
    agg["pairwise_raw"]=agg["delta"]*np.sqrt(2.0)

    if POWER_SRC.exists():
        pw=pd.read_csv(POWER_SRC)[["delta",POWER_COL]]
        agg=agg.merge(pw,on="delta",how="left")
    else:
        agg[POWER_COL]=np.nan
        print(f"note: {POWER_SRC.name} absent; floor anchors will be unavailable")

    fl,err=floor_from_power(agg)

    o.mkdir(parents=True,exist_ok=True)
    agg.to_csv(o/"separation_ratio_conversion.csv",index=False)
    (o/"separation_ratio_conversion.json").write_text(json.dumps(
        {"table":agg.to_dict(orient="records"),
         "floor":fl,
         "floor_unavailable_reason":err,
         "estimator":"mean pairwise centroid distance / RMS within-cluster radius",
         "labels":"true planted labels, measured after L2 normalization",
         "power_source":(_rel(POWER_SRC) if POWER_SRC.exists() else None)},
        indent=2))

    print(agg.to_string(index=False))
    if fl:
        print(f"\n  structureless (delta 0)          {fl['no_structure_true_label_ratio']:.3f}")
        print(f"  first non-zero power (delta {fl['delta_at_first_nonzero_power']:.0f})   "
              f"{fl['ratio_at_first_nonzero_power']:.3f}")
        print(f"  80% power                        {fl['ratio_at_80pct_power']:.3f}")
    else:
        print(f"\n  floor anchors unavailable: {err}")
    print(f"wrote {o}/separation_ratio_conversion.csv")
    return agg


if __name__=="__main__":
    main()
