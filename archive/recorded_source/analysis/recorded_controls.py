#S31 mechanism control and S32 direction reversal; see docs/FOLLOWUP_PROTOCOL_S31_S32_20260926.md
import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[v]="1"
import argparse,hashlib,json,platform,sys,time,traceback
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import scipy,sklearn
from scipy.stats import binomtest
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/"experiments/ds002748_mdd/propagated_null"))
sys.path.insert(0,str(R/"analysis"))
import nullcal as nc
import ov_common as ov
from popcov_control import sphere_spectrum
PROTOCOL=R/"docs/FOLLOWUP_PROTOCOL_S31_S32_20260926.md"
D=ov.D
CELLS={"cell1":("exp",28,"popcov_control.json","cell1_exp_n28"),"cell3":("two_block",28,"popcov_control.json","cell3_two_block_n28"),"cell6":("two_block",180,"popcov_control_cells567.json","cell6_two_block_n180")}
REVERSAL=("two_block",28,"overnight/e2_spectrum_shape.json","two_block_pr4.5")
ARMS=("M_fit","C_fit","M_H","C_H","C_genlaw")
BLOCKS=4
_G={}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def sd(root,exp,*a):
    return int.from_bytes(hashlib.sha256("|".join(map(str,(root,exp,*a))).encode()).digest()[:16],"big")

def stored(src,key):
    return json.loads((R/"results"/src).read_text(encoding="utf-8"))["cells"][key]["param_value"]

def spectrum(fam,param):
    if fam=="exp":
        return ov.spec_exp(D,float(param))
    m,e=str(param).split("/")
    return ov.spec_two_block(D,int(m),float(e))

def init(spectra):
    _G.update(spectra)

def draw(lam,n,seed):
    rng=np.random.default_rng(seed)
    z=rng.standard_normal((n,D))
    q=np.linalg.qr(rng.standard_normal((D,D)))[0]
    return nc.l2_normalize(z@(np.sqrt(lam)[:,None]*q.T)),q

def observe(x,n,i):
    g=nc.min_cluster_guard(n)
    s,k,_=nc.best_silhouette(x,k_grid=ov.K_GRID,min_size=g,n_init=ov.N_INIT,random_state=i)
    return g,s,k

def base(study,x,ds,rs,s,k):
    return dict(study=study,data_seed=str(ds),reference_seed=str(rs),source_sha256=hashlib.sha256(x.astype("<f8").tobytes()).hexdigest(),observed=s,selected_k=k,guard_failed=k is None,participation_ratio=float(nc.participation_ratio(x)))

def tests(x,arms,g,s,k,seed,nd):
    kw=dict(k_grid=ov.K_GRID,n_draws=nd,min_size=g,n_init=ov.N_INIT,seed=seed,observed=(s,k))
    res={a:nc.null_test(x,**v,**kw) for a,v in arms.items()}
    return {a:float(v.p) for a,v in res.items()},{a:int(v.n_exceed) for a,v in res.items()},{a:int(v.n_degenerate) for a,v in res.items()}

def one_a(t):
    root,cell,i,nd=t
    n=CELLS[cell][1];lam,c=_G[cell]
    ds=sd(root,87,cell,i,0);rs=sd(root,87,cell,i,1)
    r=dict(cell=cell,index=i)
    try:
        x,q=draw(lam,n,ds)
        g,s,k=observe(x,n,i)
        r.update(base("S31",x,ds,rs,s,k))
        if k is None:
            r.update(p={a:1. for a in ARMS},exceedances=None,reference_guard_failures=None)
            return r
        h=q*c@q.T;amb=q*lam@q.T
        arms={"M_fit":dict(reproject=False,match_selection=False),"C_fit":dict(reproject=True,match_selection=True),"M_H":dict(reproject=False,match_selection=False,cov=h),"C_H":dict(reproject=True,match_selection=True,cov=h),"C_genlaw":dict(reproject=True,match_selection=True,cov=amb,mean=np.zeros(D))}
        p,e,dg=tests(x,arms,g,s,k,rs,nd)
        r.update(p=p,exceedances=e,reference_guard_failures=dg)
    except Exception:
        r["error"]=traceback.format_exc()
    return r

def one_b(t):
    root,block,i,nd,deep=t
    lam=_G["reversal"]
    ds=sd(root,88,block,i,0);rs=sd(root,88,block,i,1)
    r=dict(block=block,index=i)
    try:
        x,_=draw(lam,28,ds)
        g,s,k=observe(x,28,i)
        r.update(base("S32",x,ds,rs,s,k))
        if k is None:
            r.update(p={"M":1.,"C":1.},exceedances=None,reference_guard_failures=None,retest=None)
            return r
        arms={"M":dict(reproject=False,match_selection=False),"C":dict(reproject=True,match_selection=True)}
        p,e,dg=tests(x,arms,g,s,k,rs,nd)
        r.update(p=p,exceedances=e,reference_guard_failures=dg,retest=None)
        if (p["M"]<ov.ALPHA)!=(p["C"]<ov.ALPHA):
            rs2=sd(root,88,block,i,2)
            p2,e2,_=tests(x,arms,g,s,k,rs2,deep)
            r["retest"]=dict(reference_seed=str(rs2),draws=deep,p=p2,exceedances=e2)
    except Exception:
        r["error"]=traceback.format_exc()
    return r

def spectra(root):
    out={cell:None for cell in CELLS}
    for cell,(fam,n,src,key) in CELLS.items():
        lam=spectrum(fam,stored(src,key));c=sphere_spectrum(lam)
        assert abs(float(c.sum())-1)<1e-10
        out[cell]=(lam,c)
    out["reversal"]=spectrum(REVERSAL[0],stored(REVERSAL[2],REVERSAL[3]))
    return out

def config(study,root,groups):
    src=[Path(__file__),Path(nc.__file__),Path(ov.__file__),R/"analysis/popcov_control.py"]
    inputs=["popcov_control.json","popcov_control_cells567.json"] if study=="S31" else [REVERSAL[2]]
    cells={c:dict(spectrum=v[0],n=v[1],source=f"results/{v[2]}",key=v[3],param=stored(v[2],v[3])) for c,v in CELLS.items()} if study=="S31" else {"reversal":dict(spectrum=REVERSAL[0],n=REVERSAL[1],source=f"results/{REVERSAL[2]}",key=REVERSAL[3],param=stored(REVERSAL[2],REVERSAL[3]))}
    return dict(study=study,root_seed=root,experiment=87 if study=="S31" else 88,d=D,cells=cells,groups=groups,draws=200,retest_draws=2000 if study=="S32" else None,arms=list(ARMS) if study=="S31" else ["M","C"],alpha=ov.ALPHA,rule="p < 0.05",k_grid=list(ov.K_GRID),n_init=ov.N_INIT,
                protocol_sha256=sha(PROTOCOL),source_sha256={str(f.resolve().relative_to(R)).replace("\\","/"):sha(f) for f in src},input_sha256={f"results/{f}":sha(R/"results"/f) for f in inputs},
                versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__))

def wilson(c,n):
    return list(nc._wilson(c,n))

def rate(v):
    c=int(sum(v));n=len(v)
    return dict(events=c,n=n,rate=c/n,wilson95=wilson(c,n))

def summarize_a(d):
    cfg=json.loads((d/"config.json").read_text(encoding="utf-8"))
    rows=[json.loads(l) for l in (d/"records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {(r["cell"],r["index"]) for r in rows}=={(c,i) for c,n in cfg["groups"].items() for i in range(n)} and len(rows)==sum(cfg["groups"].values()),"Missing or duplicate records"
    assert not any("error" in r for r in rows),"Numerical errors must be resolved before inference"
    out=dict(config=cfg,records_sha256=sha(d/"records.jsonl"),summary_source_sha256=sha(Path(__file__)),cells={})
    for cell in cfg["groups"]:
        rs=sorted([r for r in rows if r["cell"]==cell],key=lambda r:r["index"])
        rej={a:[r["p"][a]<cfg["alpha"] for r in rs] for a in cfg["arms"]}
        arms={a:rate(v) for a,v in rej.items()}
        g=arms["C_genlaw"]
        out["cells"][cell]=dict(n=len(rs),observed_guard_failures=sum(r["guard_failed"] for r in rs),participation_ratio_mean=float(np.mean([r["participation_ratio"] for r in rs])),arms=arms,
            C_fit_to_C_H=dict(C_H_only=sum(h and not f for f,h in zip(rej["C_fit"],rej["C_H"])),C_fit_only=sum(f and not h for f,h in zip(rej["C_fit"],rej["C_H"]))),
            generating_law_within_3_7=bool(.03<=g["rate"]<=.07),generating_law_wilson_contains_5=bool(g["wilson95"][0]<=.05<=g["wilson95"][1]))
    c=out["cells"]
    out["checks"]=dict(generating_law_within_3_7_all_cells=all(v["generating_law_within_3_7"] for v in c.values()),
        C_H_above_C_fit_cells_1_and_6=all(c[k]["arms"]["C_H"]["rate"]>c[k]["arms"]["C_fit"]["rate"] for k in ("cell1","cell6")))
    return out

def summarize_b(d):
    cfg=json.loads((d/"config.json").read_text(encoding="utf-8"))
    rows=[json.loads(l) for l in (d/"records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {(r["block"],r["index"]) for r in rows}=={(b,i) for b in range(BLOCKS) for i in range(cfg["groups"]["per_block"])} and len(rows)==BLOCKS*cfg["groups"]["per_block"],"Missing or duplicate records"
    assert not any("error" in r for r in rows),"Numerical errors must be resolved before inference"
    rej={a:[r["p"][a]<cfg["alpha"] for r in rows] for a in ("M","C")}
    m_only=sum(m and not c for m,c in zip(rej["M"],rej["C"]));c_only=sum(c and not m for m,c in zip(rej["M"],rej["C"]))
    disc=[r for r in rows if r["retest"] is not None]
    assert len(disc)==m_only+c_only
    keep=lambda r,a,b:r["retest"]["p"][a]<cfg["alpha"]<=r["retest"]["p"][b]
    persist=dict(C_only=sum(keep(r,"C","M") for r in disc if r["p"]["C"]<cfg["alpha"]),M_only=sum(keep(r,"M","C") for r in disc if r["p"]["M"]<cfg["alpha"]))
    blocks={b:dict(M=rate([r["p"]["M"]<cfg["alpha"] for r in rows if r["block"]==b]),C=rate([r["p"]["C"]<cfg["alpha"] for r in rows if r["block"]==b])) for b in range(BLOCKS)}
    out=dict(config=cfg,records_sha256=sha(d/"records.jsonl"),summary_source_sha256=sha(Path(__file__)),n=len(rows),observed_guard_failures=sum(r["guard_failed"] for r in rows),
        participation_ratio_mean=float(np.mean([r["participation_ratio"] for r in rows])),M=rate(rej["M"]),C=rate(rej["C"]),M_only=m_only,C_only=c_only,
        mcnemar_exact_p=float(binomtest(c_only,m_only+c_only,.5).pvalue) if m_only+c_only else 1.,persisted_at_retest=persist,blocks=blocks)
    out["checks"]=dict(C_above_M=out["C"]["rate"]>out["M"]["rate"])
    return out

def run(out,cfg,tasks,fn,key,workers,init_args):
    out.mkdir(parents=True,exist_ok=True)
    cp=out/"config.json"
    if cp.exists():
        assert json.loads(cp.read_text(encoding="utf-8"))==cfg,"Configuration mismatch"
    else:
        cp.write_text(json.dumps(cfg,indent=2),encoding="utf-8")
    jp=out/"records.jsonl"
    rows=[json.loads(l) for l in jp.read_text(encoding="utf-8").splitlines()] if jp.exists() else []
    done={key(r) for r in rows}
    ts=[t for t,k in tasks if k not in done]
    t0=time.time()
    with Pool(workers,initializer=init,initargs=(init_args,)) as pool,jp.open("a",encoding="utf-8") as j:
        for r in pool.imap_unordered(fn,ts,chunksize=1):
            j.write(json.dumps(r)+"\n");j.flush();rows.append(r)
            if len(rows)%100==0:
                print(f"{len(rows)}/{len(tasks)}; {time.time()-t0:.0f}s",flush=True)
    end=dict(n=len(rows),errors=sum("error" in r for r in rows),records_sha256=sha(jp),execution_wall_s=time.time()-t0)
    (out/"execution.json").write_text(json.dumps(end,indent=2),encoding="utf-8")
    print(json.dumps(end),flush=True)
    if end["errors"]:
        raise RuntimeError("Errors retained; do not summarize as completed")

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--study",required=True,choices=("S31","S32"))
    a.add_argument("--output",type=Path,required=True)
    a.add_argument("--workers",type=int,default=16)
    a.add_argument("--pilot",action="store_true")
    a.add_argument("--summarize",action="store_true")
    a=a.parse_args()
    root=2026092601 if a.pilot else 20260926
    if a.summarize:
        s=summarize_a(a.output) if a.study=="S31" else summarize_b(a.output)
        (a.output/"summary.json").write_text(json.dumps(s,indent=2),encoding="utf-8")
        print(json.dumps(s["checks"]))
        return
    sp=spectra(root)
    if a.study=="S31":
        groups={c:(3 if a.pilot else 400) for c in CELLS}
        tasks=[((root,c,i,200),(c,i)) for c,n in groups.items() for i in range(n)]
        run(a.output,config("S31",root,groups),tasks,one_a,lambda r:(r["cell"],r["index"]),a.workers,sp)
    else:
        per=3 if a.pilot else 400
        tasks=[((root,b,i,200,2000),(b,i)) for b in range(BLOCKS) for i in range(per)]
        run(a.output,config("S32",root,dict(per_block=per,blocks=BLOCKS)),tasks,one_b,lambda r:(r["block"],r["index"]),a.workers,sp)

if __name__=="__main__":
    main()
