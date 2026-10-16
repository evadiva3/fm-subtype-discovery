import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[v]="1"
import argparse,hashlib,itertools,json,sys
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/"experiments/ds002748_mdd/propagated_null"))
import nullcal as nc
A=("W","D4","K","E")

def cutoff(p):
    p=np.asarray(p)
    t=float(np.sort(p)[int(np.ceil(.05*len(p)))-1])
    rho=float((.05*len(p)-np.count_nonzero(p<t))/np.count_nonzero(p==t))
    return t,rho

def reject(p,u,t,rho):
    return (p<t)|((p==t)&(u<rho))

def ci(c,n):
    return list(nc._wilson(c,n))

def summarize(d):
    cfg=json.loads((d/"config.json").read_text(encoding="utf-8"))
    arms=tuple(cfg.get("arms",A));na=len(arms);spectra=range(len(cfg.get("spectra",(0,1))))
    rows=[json.loads(l) for l in (d/"records.jsonl").read_text(encoding="utf-8").splitlines()]
    keys={(r["spectrum"],r["q"],r["group"],r["index"]) for r in rows}
    expected={(s,q,g,i) for s in spectra for q in cfg["qs"] for g,n in cfg["groups"].items() for i in range(n)}
    assert len(rows)==len(keys) and keys==expected,"Missing or duplicate records"
    assert not any("error" in r for r in rows),"Numerical errors must be resolved before inference"
    out=dict(config=cfg,records_sha256=hashlib.sha256((d/"records.jsonl").read_bytes()).hexdigest(),summary_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cells={},comparisons={},uncertainty="Conditional Wilson intervals and two-stage paired bootstrap percentile intervals; the latter include calibration-cutoff estimation and evaluation sampling. All intervals are pointwise. No universal method ranking.")
    nb=cfg["bootstrap_resamples"]
    for s,q in itertools.product(spectra,cfg["qs"]):
        cal=sorted([r for r in rows if r["spectrum"]==s and r["q"]==q and r["group"]=="calibration"],key=lambda r:r["index"])
        cp=np.array([[r["p"][a] for a in arms] for r in cal])
        rng=np.random.default_rng(np.random.SeedSequence([20260924,cfg.get("bootstrap_address",73),s,q]))
        boot={}
        ix=rng.integers(0,len(cal),size=(nb,len(cal)))
        cuts=[cutoff(cp[:,a]) for a in range(na)]
        bc=np.empty((nb,na,2))
        for b in range(nb):
            for a in range(na):
                bc[b,a]=cutoff(cp[ix[b],a])
        indices={}
        for n in sorted(set(cfg["groups"].values())):
            indices[n]=rng.integers(0,n,size=(nb,n))
        for g,n in cfg["groups"].items():
            if g=="calibration":
                continue
            rs=sorted([r for r in rows if r["spectrum"]==s and r["q"]==q and r["group"]==g],key=lambda r:r["index"])
            p=np.array([[r["p"][a] for a in arms] for r in rs])
            u=np.array([r["boundary_uniform"] for r in rs])
            br=np.empty((nb,na))
            ix=indices[n]
            for a in range(na):
                br[:,a]=reject(p[ix,a],u[ix],bc[:,a,0,None],bc[:,a,1,None]).mean(axis=1)
            cell=dict(n=n,observed_guard_failures=sum(r["guard_failed"] for r in rs),reference_guard_failures=sum(r["reference_guard_failures"] for r in rs))
            for dk in cfg.get("diagnostics",("projected_squared_half_separation","retained_axis_squared")):
                cell["median_"+dk]=float(np.median([r[dk] for r in rs]))
            cell["arms"]={}
            rr=[]
            for a,name in enumerate(arms):
                t,rho=cuts[a]
                v=reject(p[:,a],u,t,rho)
                rr.append(v)
                c=int(v.sum())
                cn=int((p[:,a]<=.05).sum())
                w=ci(c,n)
                cell["arms"][name]=dict(cutoff=t,boundary_probability=rho,calibration_expected_rate=float(np.mean(cp[:,a]<t)+rho*np.mean(cp[:,a]==t)),nominal_events=cn,nominal_rate=cn/n,nominal_wilson95=ci(cn,n),calibrated_events=c,calibrated_rate=c/n,conditional_wilson95=w,two_stage_bootstrap95=np.quantile(br[:,a],[.025,.975]).tolist(),heldout_tolerance_03_07_supported=bool(w[0]>=.03 and w[1]<=.07) if g=="null" else None)
            key=f"{s}_q{q}_{g}"
            out["cells"][key]=cell
            out["comparisons"][key]={}
            for a,b in itertools.combinations(range(na),2):
                dv=rr[a].astype(float)-rr[b].astype(float)
                out["comparisons"][key][arms[a]+"-"+arms[b]]=dict(rate_difference=float(dv.mean()),a_only=int(np.count_nonzero(dv==1)),b_only=int(np.count_nonzero(dv==-1)),two_stage_bootstrap95=np.quantile(br[:,a]-br[:,b],[.025,.975]).tolist())
            boot[g]=(rr,br)
        # Paired contrasts between alternatives share evaluation indices and bootstrap resamples.
        for g1,g2 in cfg.get("contrasts",()):
            assert cfg["groups"][g1]==cfg["groups"][g2]
            key=f"{s}_q{q}_{g1}-{g2}"
            out.setdefault("alternative_contrasts",{})[key]={}
            for a,name in enumerate(arms):
                dv=boot[g1][0][a].astype(float)-boot[g2][0][a].astype(float)
                out["alternative_contrasts"][key][name]=dict(rate_difference=float(dv.mean()),first_only=int(np.count_nonzero(dv==1)),second_only=int(np.count_nonzero(dv==-1)),two_stage_bootstrap95=np.quantile(boot[g1][1][:,a]-boot[g2][1][:,a],[.025,.975]).tolist())
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--directory",type=Path,required=True)
    a=p.parse_args()
    out=summarize(a.directory)
    (a.directory/"summary.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps({k:{a:round(v["calibrated_rate"],3) for a,v in c["arms"].items()} for k,c in out["cells"].items()},indent=2))

if __name__=="__main__":
    main()
