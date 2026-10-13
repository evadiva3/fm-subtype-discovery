#Phase 1B: does the M against C ordering survive departures from Gaussianity?
import json
import os
import platform
import sys
import time
from datetime import datetime,timezone
from multiprocessing import Pool
from pathlib import Path
for _v in("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ.setdefault(_v,"1")
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import nullcal as nc
import ov_common as ov
D=119;PR_TARGETS=[3.0,4.5,9.0];NS=[28,119];FAMILIES=["gaussian","mvt3","mvt5","lognormal","gmm_R0.3"];NDS=400;N_DRAWS=200;N_INIT=5;K_GRID=(2,3,4,5,6);ALPHA=0.05;LOGNORMAL_S=0.5;GMM_R=0.3
SEED_ROOT=4200000;_CFG=None
def _init(cfg):
    global _CFG;_CFG=cfg
def draw_raw(family,n,d,A,delta,rng):
    Z=rng.standard_normal((n,d))@A
    if family=="gaussian":return(Z,None)
    if family in("mvt3","mvt5"):
        nu=3 if family=="mvt3" else 5;Zs=Z*np.sqrt((nu-2.0)/nu);w=rng.chisquare(nu,size=(n,1))/nu;return(Zs/np.sqrt(w),None)
    if family=="lognormal":
        sd=np.sqrt((A**2).sum(axis=0));sd[sd==0]=1.0;Wn=Z/sd;s=LOGNORMAL_S;m=np.exp(s*s/2.0);v=np.sqrt(np.exp(s*s)*(np.exp(s*s)-1.0));return((np.exp(s*Wn)-m)/v*sd,None)
    if family=="gmm_R0.3":
        g=nc.min_cluster_guard(n);n1=max(g,n//2);l=np.zeros(n,dtype=int);l[n1:]=1;u=rng.standard_normal(d);u/=np.linalg.norm(u);o=np.where(l[:,None]==0,-0.5,0.5)*delta*u[None,:];return(Z+o,l)
    raise ValueError(family)
def one(args):
    ky,f,n,i,sd=args;d,A,da=(_CFG[ky]["d"],_CFG[ky]["A"],_CFG[ky].get("delta",{}).get(f))
    try:
        rg=np.random.default_rng(sd);rw,l=draw_raw(f,n,d,A,da,rg);X=nc.l2_normalize(rw);g=nc.min_cluster_guard(n);s,k,_=nc.best_silhouette(X,k_grid=K_GRID,min_size=g,n_init=N_INIT,random_state=i)
        if k is None:return{"key":ky,"family":f,"skipped":True}
        kw=dict(k_grid=K_GRID,n_draws=N_DRAWS,min_size=g,n_init=N_INIT,seed=sd+7,observed=(s,k))
        o={"key":ky,"family":f,"pr":float(nc.participation_ratio(X)),"pr_raw":float(nc.participation_ratio(rw)),"obs":float(s),"k":int(k)}
        if l is not None:o["sep_ratio"]=float(nc.separation_ratio(X,l))
        for ne,rp,ms in(("M",False,False),("G",True,False),("S",False,True),("C",True,True)):
            r=nc.null_test(X,reproject=rp,match_selection=ms,**kw);o["p"+ne]=float(r.p);o["null"+ne]=float(r.null_mean);o["sd"+ne]=float(r.null_sd)
        return o
    except Exception as e:return{"key":ky,"family":f,"error":f"{type(e).__name__}: {e}"}
def tune_delta(n,d,A,target,seed=987654,m=60):
    def meas(delta):
        v=[]
        for j in range(m):
            rg=np.random.default_rng(seed+31*j);r,l=draw_raw("gmm_R0.3",n,d,A,delta,rg);v.append(nc.separation_ratio(nc.l2_normalize(r),l))
        return float(np.mean(v))
    lo,hi=(0.0,2.0*float(np.sqrt((A**2).sum())))
    for _ in range(40):
        md=0.5*(lo+hi)
        if meas(md)<target:lo=md
        else:hi=md
        if hi-lo<1e-06*max(1.0,hi):break
    dl=0.5*(lo+hi);return(dl,meas(dl))
def summarize(res):
    g=[r for r in res if "pM" in r]
    if not g:return{"n_datasets":0,"n_errors":sum((1 for r in res if "error" in r)),"error_sample":next((r["error"] for r in res if "error" in r),None)}
    ps=np.array([r["pr"] for r in g])
    cl={"n_datasets":len(g),"n_skipped":sum((1 for r in res if r.get("skipped"))),"n_errors":sum((1 for r in res if "error" in r)),"pr_sphere":{"mean":float(ps.mean()),"sd":float(ps.std(ddof=1)),"min":float(ps.min()),"max":float(ps.max())},"pr_raw_mean":float(np.mean([r["pr_raw"] for r in g])),"obs_mean":float(np.mean([r["obs"] for r in g])),"selected_k_counts":{str(k):int(sum((1 for r in g if r["k"]==k))) for k in K_GRID}}
    if "sep_ratio" in g[0]:
        sr=np.array([r["sep_ratio"] for r in g]);cl["separation_ratio"]={"mean":float(sr.mean()),"sd":float(sr.std(ddof=1))}
    for a in("M","G","S","C"):
        p=np.array([r["p"+a] for r in g]);c=int(np.sum(p<ALPHA))
        cl[a]={"rej":float(c/p.size),"events":c,"ci_wilson":list(nc._wilson(c,p.size)),"mean_p":float(p.mean()),"ks_D":float(nc._ks_uniform(p)),"null_mean":float(np.mean([r["null"+a] for r in g])),"null_sd":float(np.mean([r["sd"+a] for r in g]))}
    ab,sh=(cl["M"]["null_mean"],cl["G"]["null_mean"]);s=cl["M"]["null_sd"]
    cl["shift"]={"ambient_null_mean":ab,"sphere_null_mean":sh,"raw":float(sh-ab),"in_ambient_sd":float((sh-ab)/s) if s else None,"sign":int(np.sign(sh-ab))};pM=np.array([r["pM"] for r in g])
    pC=np.array([r["pC"] for r in g]);cl["M_only"]=int(np.sum((pM<ALPHA)&(pC>=ALPHA)));cl["C_only"]=int(np.sum((pC<ALPHA)&(pM>=ALPHA)));cl["C_minus_M"]=float(cl["C"]["rej"]-cl["M"]["rej"]);return cl
def main():
    s=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ");o1=ov.ROOT/"results"/"phase1"/s;o1.mkdir(parents=True,exist_ok=True);o=o1/"r1b_nongaussian.json";t=time.time();cg,g=({},{})
    for pr in PR_TARGETS:
        for n in NS:
            k=f"pr{pr:g}_n{n}";tu,p,ok,ne=ov.bisect_param(n,D,lambda t:ov.spec_exp(D,t),0.2,200.0,pr,tol=0.05,m=100);rg=np.random.default_rng(SEED_ROOT+hash(k)%100000)
            A=ov.factor(D,ov.spec_exp(D,tu),rg);cg[k]={"d":D,"A":A,"delta":{}};d,r=tune_delta(n,D,A,GMM_R);cg[k]["delta"]["gmm_R0.3"]=d
            g[k]={"pr_target":pr,"n":n,"tau":float(tu),"tuned_pr_gaussian":float(p),"tune_ok":bool(ok),"tune_note":ne,"gmm_delta":float(d),"gmm_separation_ratio_measured":float(r)}
            print(f"[tune] {k:12s} tau={tu:8.4f} gaussian PR={p:6.3f}  gmm delta={d:.4f} R={r:.4f}",flush=True)
    import scipy
    import sklearn
    h={"experiment":"R1B non-Gaussian robustness","purpose":"measure whether the M against C ordering and the shift sign survive departures from Gaussianity","written_utc":s,"d":D,"n_values":NS,"pr_targets":PR_TARGETS,"families":FAMILIES,"n_datasets_per_cell":NDS,"n_draws":N_DRAWS,"n_init":N_INIT,"k_grid":list(K_GRID),"alpha":ALPHA,"lognormal_latent_scale":LOGNORMAL_S,"gmm_target_separation_ratio":GMM_R,"gmm_note":"0.3 is far below the article's detection floor of 1.17, so no partition is recoverable and a rejection remains a false positive","covariance_matching":"tau bisected once per (PR target, n) on the Gaussian; all five families in that cell share the same Sigma. Realized sphere PR is measured per cell and reported","seed_root":SEED_ROOT,"seed_rule":"dataset i of cell (key, family) uses seed_root + cell_offset + family_offset + 1000*i; its draws use that + 7","geometries":g,"versions":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"scikit_learn":sklearn.__version__,"platform":platform.platform(),"n_cores_used":47}}
    se={"header":h,"cells":{}}
    with Pool(47,initializer=_init,initargs=(cg,))as po:
        for pr in PR_TARGETS:
            for n in NS:
                k=f"pr{pr:g}_n{n}";cf=int(pr*1000)*7919+n*104729
                for fi,f in enumerate(FAMILIES):
                    t0=time.time();a=[(k,f,n,i,SEED_ROOT+cf+fi*1000003+1000*i) for i in range(NDS)];rs=list(po.imap_unordered(one,a,chunksize=2));c=summarize(rs)
                    c.update(pr_target=pr,n=n,family=f,wall_s=round(time.time()-t0,1));se["cells"][f"{k}_{f}"]=c
                    if c.get("n_datasets"):
                        print(f"{k:12s} {f:10s} PR {c["pr_sphere"]["mean"]:6.2f}  M {c["M"]["rej"]:.3f}  C {c["C"]["rej"]:.3f}  shift {c["shift"]["in_ambient_sd"]:+.3f} sd  {c["wall_s"]:.0f}s",flush=True)
                    else:print(f"{k:12s} {f:10s} FAILED {c.get("error_sample")}",flush=True)
                    json.dump(se,open(o,"w"),indent=1)
    se["header"]["total_wall_s"]=round(time.time()-t,1);json.dump(se,open(o,"w"),indent=1);print(f"\ntotal wall {se["header"]["total_wall_s"] / 60:.1f} min");print(f"wrote {o}")
if __name__=="__main__":main()
