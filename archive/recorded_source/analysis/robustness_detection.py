#S29: non-Gaussian laws and non-eigenvector alternatives; see docs/FOLLOWUP_PROTOCOL_S29_20260925.md
import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[v]="1"
import argparse,hashlib,json,platform,sys,time,traceback
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import scipy,sklearn
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/"reproducibility/s16/modules"))
sys.path.insert(0,str(R/"analysis"))
import angular_gof as gof
import nullcal_validation as nv
from asymmetric_detection import PROJ,spectrum,sha
import mc_calibrated_sensitivity as s24
from summarize_detection import cutoff,reject,ci,summarize
nc,ov=s24.nc,s24.ov
PROTOCOL=R/"docs/FOLLOWUP_PROTOCOL_S29_20260925.md"
GA=("calibration","null","rand34","hexa4","skew","heavy")
GB=("skew","heavy")
SID={"calibration":0,"null":1,"rand34":1,"hexa4":1,"skew":2,"heavy":3}

def sd(root,exp,*a):
    return int.from_bytes(hashlib.sha256("|".join(map(str,(root,exp,*a))).encode()).digest()[:16],"big")

def noise(law,rng,shape):
    if law=="skew":
        return rng.exponential(size=shape)-1.
    if law=="heavy":
        return rng.standard_t(4,size=shape)/np.sqrt(2.)
    return rng.normal(size=shape)

def sizes(g,n):
    if g=="rand34":
        return [n-2*(n//3),n//3,n//3]
    return [n-5*(n//6)]+[n//6]*5

def means(g,rot,lam,perm,dirs):
    if g not in ("rand34","hexa4"):
        return None,None
    n=len(perm)
    if g=="rand34":
        cov=(rot*lam)@rot.T
        a=np.stack([np.sqrt(u@cov@u)*u for u in dirs.T])
        th=(0.,2*np.pi/3,4*np.pi/3)
    else:
        a=(np.sqrt(lam[:2])*rot[:,:2]).T
        th=[j*np.pi/3 for j in range(6)]
    M=4.*np.array([[np.cos(t),np.sin(t)] for t in th])@a
    sz=sizes(g,n)
    lab=np.repeat(np.arange(len(sz)),sz)[np.argsort(np.argsort(perm))]
    w=np.bincount(lab,minlength=len(M))/n
    return M-w@M,lab

def one_a(t):
    root,n,s,q,g,i,nd=t
    name=GA[g]
    ds=sd(root,82,n,0,s,SID[name],i)
    r=dict(n=n,spectrum=s,q=q,group=name,index=i,data_seed=str(ds))
    try:
        rng=np.random.default_rng(ds)
        z=noise(name,rng,(n,119))
        rot=np.linalg.qr(rng.normal(size=(119,119)))[0]
        perm=rng.permutation(n)
        dirs=np.linalg.qr(np.random.default_rng(sd(root,82,n,5,s,i)).normal(size=(119,2)))[0]
        lam=spectrum(s)
        x=(z*np.sqrt(lam))@rot.T
        M,lab=means(name,rot,lam,perm,dirs)
        p=PROJ[:,:q]
        cov=(p.T@rot*lam)@(rot.T@p)
        between=shift=0.
        if M is not None:
            x=x+M[lab]
            w=np.bincount(lab,minlength=len(M))/n
            pm=M@p
            mbar=w@pm
            dm=pm-mbar
            between=float(np.trace(np.linalg.solve(cov,(dm.T*w)@dm)))
            shift=float(mbar@np.linalg.solve(cov,mbar))
        xp=nv.normalize(nv.normalize(x)@p)
        rs=sd(root,82,n,2,s,q,g,i)
        bs=sd(root,82,n,3,s,q,g,i)
        u=float(np.random.default_rng(sd(root,82,n,4,s,q,g,i)).uniform())
        r.update(group_sizes=None if lab is None else np.bincount(lab).tolist(),source_sha256=hashlib.sha256(x.astype("<f8").tobytes()).hexdigest(),projected_sha256=hashlib.sha256(xp.astype("<f8").tobytes()).hexdigest(),projected_between_scatter=between,projected_mean_shift=shift,reference_seed=str(rs),bootstrap_seed=str(bs),boundary_uniform=u)
        r.update(gof.evaluate(xp,draws=nd,seed=rs,bootstrap_seed=bs,auxiliary=200))
    except Exception:
        r["error"]=traceback.format_exc()
    return r

def one_b(t):
    root,g,i,nd=t
    name=GB[g]
    ds=sd(root,84,0,g,i)
    r=dict(group=name,index=i,data_seed=str(ds))
    try:
        rng=np.random.default_rng(ds)
        x=nc.l2_normalize(s24.MEAN+noise(name,rng,(s24.N,s24.FACTOR.shape[0]))@s24.FACTOR)
        rs=sd(root,84,2,g,i)
        u=float(np.random.default_rng(sd(root,84,4,g,i)).uniform())
        s,k,l=nc.best_silhouette(x,k_grid=ov.K_GRID,min_size=nc.min_cluster_guard(s24.N),n_init=5,random_state=i)
        r.update(source_sha256=hashlib.sha256(x.astype("<f8").tobytes()).hexdigest(),observed=s,selected_k=k,guard_failed=k is None,reference_seed=str(rs),boundary_uniform=u)
        if k is None:
            r.update(p={"M":1.,"C":1.},reference_guard_failures=0)
        else:
            kw=dict(k_grid=ov.K_GRID,n_draws=nd,min_size=nc.min_cluster_guard(s24.N),n_init=5,seed=rs,observed=(s,k))
            res={a:nc.null_test(x,reproject=z,match_selection=z,**kw) for a,z in (("M",False),("C",True))}
            r.update(p={a:float(v.p) for a,v in res.items()},reference_guard_failures=int(res["C"].n_degenerate))
    except Exception:
        r["error"]=traceback.format_exc()
    return r

def sources():
    fs=[Path(__file__),R/"analysis/asymmetric_detection.py",R/"analysis/calibrated_detection.py",R/"analysis/mc_calibrated_sensitivity.py",R/"analysis/summarize_detection.py",Path(nc.__file__),Path(ov.__file__),*(R/"reproducibility/s16/modules").glob("*.py")]
    return {str(f.resolve().relative_to(R)).replace("\\","/"):sha(f) for f in fs}

def versions():
    return dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__)

def config_a(root,n,ns):
    return dict(study="S29a",root_seed=root,experiment=82,n=n,d=119,qs=[5,10],spectra=["two_block_1_0.01","exponential_tau10"],groups=dict(zip(GA,ns)),draws=199,auxiliary=200,bootstrap_resamples=2000,bootstrap_address=83,diagnostics=["projected_between_scatter","projected_mean_shift"],projection="S19 fixed nested projection (experiment 72 address)",protocol_sha256=sha(PROTOCOL),source_sha256=sources(),versions=versions())

def config_b(root,ns):
    return dict(study="S29b",root_seed=root,experiment=84,n=s24.N,d=int(s24._X.shape[1]),arms=["M","C"],groups=dict(zip(GB,ns)),draws=499,bootstrap_resamples=2000,bootstrap_address=85,calibration="results/revision_20260925/mc_calibrated_sensitivity",input_sha256=sha(s24.EMB),protocol_sha256=sha(PROTOCOL),source_sha256=sources(),versions=versions())

def summarize_b(d):
    cfg=json.loads((d/"config.json").read_text(encoding="utf-8"))
    cal=[json.loads(l) for l in (R/cfg["calibration"]/"records.jsonl").read_text(encoding="utf-8").splitlines()]
    cal=sorted([r for r in cal if r["group"]=="calibration"],key=lambda r:r["index"])
    assert len(cal)==1000 and not any("error" in r for r in cal)
    rows=[json.loads(l) for l in (d/"records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {(r["group"],r["index"]) for r in rows}=={(g,i) for g,n in cfg["groups"].items() for i in range(n)} and len(rows)==sum(cfg["groups"].values()),"Missing or duplicate records"
    assert not any("error" in r for r in rows),"Numerical errors must be resolved before inference"
    out=dict(config=cfg,records_sha256=sha(d/"records.jsonl"),summary_source_sha256=sha(Path(__file__)),cells={})
    nb=cfg["bootstrap_resamples"]
    for gi,g in enumerate(cfg["groups"]):
        rs=sorted([r for r in rows if r["group"]==g],key=lambda r:r["index"])
        cell=dict(n=len(rs),observed_guard_failures=sum(r["guard_failed"] for r in rs),reference_guard_failures=sum(r["reference_guard_failures"] for r in rs),arms={})
        rng=np.random.default_rng(np.random.SeedSequence([20260925,cfg["bootstrap_address"],gi]))
        ic=rng.integers(0,len(cal),size=(nb,len(cal)))
        ie=rng.integers(0,len(rs),size=(nb,len(rs)))
        u=np.array([r["boundary_uniform"] for r in rs])
        for a in cfg["arms"]:
            cp=np.array([r["p"][a] for r in cal])
            p=np.array([r["p"][a] for r in rs])
            t,rho=cutoff(cp)
            v=reject(p,u,t,rho)
            bt=np.array([reject(p[ie[b]],u[ie[b]],*cutoff(cp[ic[b]])).mean() for b in range(nb)])
            c=int(v.sum());cn=int((p<=.05).sum())
            cell["arms"][a]=dict(cutoff=t,boundary_probability=rho,calibrated_events=c,calibrated_rate=c/len(rs),conditional_wilson95=ci(c,len(rs)),two_stage_bootstrap95=np.quantile(bt,[.025,.975]).tolist(),nominal_events=cn,nominal_rate=cn/len(rs),nominal_wilson95=ci(cn,len(rs)))
        out["cells"][g]=cell
    return out

def run(out,cfg,tasks,fn,key,total,workers):
    out.mkdir(parents=True,exist_ok=True)
    cp=out/"config.json"
    if cp.exists():
        assert json.loads(cp.read_text(encoding="utf-8"))==cfg,"Configuration mismatch"
    else:
        cp.write_text(json.dumps(cfg,indent=2),encoding="utf-8")
    jp=out/"records.jsonl"
    rows=[json.loads(l) for l in jp.read_text(encoding="utf-8").splitlines()] if jp.exists() else []
    done={key(r) for r in rows}
    ts=[t for t in tasks if t[-1] not in done]
    t0=time.time()
    with Pool(workers) as pool,jp.open("a",encoding="utf-8") as j:
        for r in pool.imap_unordered(fn,[t[0] for t in ts],chunksize=1):
            j.write(json.dumps(r)+"\n");j.flush();rows.append(r)
            if len(rows)%250==0:
                print(f"{len(rows)}/{total}; {time.time()-t0:.0f}s",flush=True)
    end=dict(n=len(rows),errors=sum("error" in r for r in rows),records_sha256=sha(jp),execution_wall_s=time.time()-t0)
    (out/"execution.json").write_text(json.dumps(end,indent=2),encoding="utf-8")
    print(json.dumps(end),flush=True)
    if end["errors"]:
        raise RuntimeError("Errors retained; do not summarize as completed")

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--part",required=True,choices=("a","b"))
    a.add_argument("--output",type=Path,required=True)
    a.add_argument("--n",type=int,choices=(28,100))
    a.add_argument("--workers",type=int,default=16)
    a.add_argument("--pilot",action="store_true")
    a.add_argument("--summarize",action="store_true")
    a=a.parse_args()
    root=2026092505 if a.pilot else 20260925
    if a.part=="a":
        if a.summarize:
            (a.output/"summary.json").write_text(json.dumps(summarize(a.output),indent=2),encoding="utf-8");return
        ns=[3]*len(GA) if a.pilot else [1000,1000,250,250,1000,1000]
        tasks=[((root,a.n,s,q,g,i,199),(s,q,GA[g],i)) for s in range(2) for q in (5,10) for g in range(len(GA)) for i in range(ns[g])]
        run(a.output,config_a(root,a.n,ns),tasks,one_a,lambda r:(r["spectrum"],r["q"],r["group"],r["index"]),len(tasks),a.workers)
    else:
        if a.summarize:
            (a.output/"summary.json").write_text(json.dumps(summarize_b(a.output),indent=2),encoding="utf-8");return
        ns=[3]*len(GB) if a.pilot else [1000,1000]
        tasks=[((root,g,i,499),(GB[g],i)) for g in range(len(GB)) for i in range(ns[g])]
        run(a.output,config_b(root,ns),tasks,one_b,lambda r:(r["group"],r["index"]),len(tasks),a.workers)

if __name__=="__main__":
    main()
