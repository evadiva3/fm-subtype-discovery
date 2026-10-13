#Cell 2 direction check, from the per-dataset replay in ambient_control.json.
import json
from pathlib import Path
import numpy as np
_R=Path(__file__).resolve().parents[1];SRC=_R/"results"/"ambient_control.json";OUT=_R/"results"/"cell2_direction_check.json";ALPHA=0.05;FEATURES=["pr","obs","pc1","mean_norm"]
def describe(rows):
    if not rows:return dict(n=0)
    o=dict(n=len(rows))
    for f in FEATURES:
        v=np.array([r[f] for r in rows]);o[f]=dict(mean=float(v.mean()),median=float(np.median(v)),min=float(v.min()),max=float(v.max()))
    o["k_counts"]={str(k):sum((1 for r in rows if r["k"]==k)) for k in sorted({r["k"] for r in rows})}
    o["C_est_exceed"]=dict(mean=float(np.mean([r["exceed_C_est"] for r in rows])),max=int(max((r["exceed_C_est"] for r in rows))))
    o["C_known_exceed"]=dict(mean=float(np.mean([r["exceed_C_known"] for r in rows])),max=int(max((r["exceed_C_known"] for r in rows))));return o
def ranks(rows,key,flag):
    v=np.array([r[key] for r in rows]);o=v.argsort().argsort()+1;f=np.array([flag(r) for r in rows])
    return dict(flagged=int(f.sum()),mean_rank_flagged=float(o[f].mean()) if f.any() else None,mean_rank_all=float(o.mean()))
def main():
    c=json.loads(SRC.read_text())["cells"]["replay__cell2_exp_n200"];rs=c["rows"];rj=lambda r,a:r["p_"+a]<ALPHA
    gs={"C_est_reject_C_known_accept":[r for r in rs if rj(r,"C_est") and(not rj(r,"C_known"))],"C_est_accept_C_known_reject":[r for r in rs if not rj(r,"C_est") and rj(r,"C_known")],"both_reject":[r for r in rs if rj(r,"C_est") and rj(r,"C_known")],"both_accept":[r for r in rs if not rj(r,"C_est") and(not rj(r,"C_known"))]}
    p=float(np.mean([r["pr"] for r in rs]))
    for r in rs:r["pr_dev"]=abs(r["pr"]-p)
    d=lambda r:rj(r,"C_est") and(not rj(r,"C_known"))
    o=dict(source="results/ambient_control.json, replay__cell2_exp_n200",reproduction=c["reproduction"],reproduces_archive=all((v["identical_events"] for v in c["reproduction"].values())),n_datasets=len(rs),pr_mean=p,arms={a:dict(events=v["events"],rate=v["rejection_rate"],wilson95=v["wilson95"]) for a,v in c["arms"].items()},paired=c["paired"],groups={k:describe(v) for k,v in gs.items()},rank_of_downward_datasets={f:ranks(rs,f,d) for f in FEATURES+["pr_dev"]},exceedances_of_downward_datasets=[dict(i=r["i"],pr=r["pr"],obs=r["obs"],k=r["k"],exceed_C_est=r["exceed_C_est"],exceed_C_known=r["exceed_C_known"],exceed_C_ambient=r["exceed_C_ambient"]) for r in gs["C_est_reject_C_known_accept"]])
    OUT.write_text(json.dumps(o,indent=1));print(json.dumps({k:o[k] for k in("reproduces_archive","arms","paired")},indent=1));print(json.dumps(o["rank_of_downward_datasets"],indent=1))
    for k,g in o["groups"].items():print(k,g.get("n"),{f:round(g[f]["mean"],4) for f in FEATURES if f in g},g.get("k_counts"))
if __name__=="__main__":main()
