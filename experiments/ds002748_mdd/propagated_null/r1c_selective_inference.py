#Phase 1C: benchmark against the existing selective-inference methods.
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import nullcal as nc
import ov_common as ov
RSCRIPT = os.environ.get("RSCRIPT") or shutil.which("Rscript")
RCMD =[RSCRIPT] if RSCRIPT else []
R_DRIVER=Path(__file__).resolve().parent/"r1c_selective_inference.R";D=119;PR_TARGETS=[3.0,4.5,9.0];NS=[28,119];NDS=int(os.environ.get("R1C_NDS",200));K=2;ALPHA=0.05;SEED_ROOT=4300000
def write_cell(path,arrays,meta):
    path.mkdir(parents=True,exist_ok=True);a=np.ascontiguousarray(np.stack(arrays),dtype=np.float64);a.tofile(path/"data.bin");meta=dict(meta)
    meta.update(n_datasets=int(a.shape[0]),n=int(a.shape[1]),d=int(a.shape[2]),K=K);json.dump(meta,open(path/"manifest.json","w"),indent=1);return meta
def run_r(path):
    t0=time.time();r=subprocess.run(RCMD+[str(R_DRIVER),str(path)],capture_output=True,text=True,timeout=7200)
    return{"returncode":r.returncode,"stdout":r.stdout[-4000:],"stderr":r.stderr[-4000:],"wall_s":round(time.time()-t0,1)}
def read_results(path):
    import csv
    f=path/"results.csv"
    if not f.exists():return None
    rs=list(csv.DictReader(open(f,newline="")));o={}
    for cl in("p_hier_iso","p_hier_gencov","p_kmeans_iso","p_kmeans_gencov"):
        v=[float(r[cl]) for r in rs if r[cl] not in("NA","","NaN")];e=[r["err_"+cl.replace("p_","")] for r in rs if r.get("err_"+cl.replace("p_",""),"")];c=int(sum((1 for v in v if v<ALPHA)))
        o[cl]={"n_ran":len(v),"n_attempted":len(rs),"n_failed":len(rs)-len(v),"rejection_rate":float(c/len(v)) if v else None,"events":c,"ci_wilson":list(nc._wilson(c,len(v))) if v else None,"mean_p":float(np.mean(v)) if v else None,"ks_D":float(nc._ks_uniform(np.array(v))) if len(v)>1 else None,"error_sample":e[0] if e else None,"n_distinct_errors":len(set(e))}
    o["sig_hat_mean"]=float(np.mean([float(r["sig_hat"]) for r in rs if r["sig_hat"] not in("NA","")]));return o
def gen_cells(work):
    c=[]
    def add(name,arrays,meta):
        m=write_cell(work/name,arrays,meta);c.append({"name":name,**m});print(f"  built {name}: {m["n_datasets"]} x {m["n"]} x {m["d"]}",flush=True)
    for ne,n,d in(("control_iso",200,20),("control_iso_nltd",28,D)):
        a1=[]
        for i in range(NDS):
            rg=np.random.default_rng(SEED_ROOT+11*1000003+1000*i+d);a1.append(rg.standard_normal((n,d)))
        add(ne,a1,{"kind":"positive control, Sigma = I, unnormalized","normalized":False,"spectrum":"isotropic","defensible":True,"pr_sphere_mean":float(np.mean([nc.participation_ratio(nc.l2_normalize(a)) for a in a1[:50]]))})
    for pr in PR_TARGETS:
        for n in NS:
            t,p,ok,n1=ov.bisect_param(n,D,lambda t:ov.spec_exp(D,t),0.2,200.0,pr,tol=0.05,m=100);l=ov.spec_exp(D,t);r=[]
            for i in range(NDS):
                rg=np.random.default_rng(SEED_ROOT+int(pr*1000)*7919+n*104729+1000*i);r.append(rg.standard_normal((n,D))@ov.factor(D,l,rg))
            b={"pr_target":pr,"n":n,"tau":float(t),"tuned_pr_gaussian":float(p),"tune_ok":bool(ok),"spectrum":"exp(-i/tau)"}
            add(f"pr{pr:g}_n{n}_raw",r,{**b,"kind":"surface cell, unnormalized","normalized":False,"defensible":True,"caveat":"anisotropic, so the isotropy assumption is violated; the general-covariance path needs an invertible covariance","pr_sphere_mean":float(np.mean([nc.participation_ratio(nc.l2_normalize(a)) for a in r[:50]]))})
            add(f"pr{pr:g}_n{n}_l2",[nc.l2_normalize(a) for a in r],{**b,"kind":"surface cell, L2-normalized","normalized":True,"defensible":False,"caveat":"rows lie on the unit sphere, a measure-zero set under the assumed Gaussian model; reported as inapplicable","pr_sphere_mean":float(np.mean([nc.participation_ratio(nc.l2_normalize(a)) for a in r[:50]]))})
    return c
def real_embeddings():
    f={}
    for n,r in(("fibromyalgia","data/outputs/trained_fm_embeddings.npy"),("depression","results/mdd/Embeddings.npy")):
        p=ov.ROOT/r
        if p.exists():f[n]=np.load(p)
    return f
def main():
    if not RSCRIPT:
        print("Rscript was not found. Add it to PATH or set RSCRIPT to its executable path.");sys.exit(2)
    s=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ");o1=ov.ROOT/"results"/"phase1"/s;o1.mkdir(parents=True,exist_ok=True);o=o1/"r1c_selective_inference.json";w=o1/"_cells";t=time.time()
    v=R_DRIVER.parent/"_r1c_versions.R"
    v.write_text('cat(R.version.string, "|"); for (p in c("clusterpval","KmeansInference","jsonlite")) cat(p, as.character(packageVersion(p)), "|")\n',encoding="utf-8")
    rv=subprocess.run(RCMD+[str(v)],capture_output=True,text=True);print("R stack:",rv.stdout.strip(),flush=True)
    import scipy
    import sklearn
    h={"experiment":"R1C benchmark against existing selective inference","purpose":"run Chen and Witten and Gao et al. where defensible, record inapplicability where not","audit":"docs/selective_inference_assumption_audit.md, written before this ran","written_utc":s,"n_datasets_per_cell":NDS,"K_tested":K,"alpha":ALPHA,"d":D,"pr_targets":PR_TARGETS,"n_values":NS,"seed_root":SEED_ROOT,"seed_rule":"dataset i of a cell uses seed_root + cell_offset + 1000*i","sigma_supplied":"pooled isotropic estimate sqrt(sum(scale(X)^2)/(n*q - q)), the packages' own default","r_stack":rv.stdout.strip(),"versions":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"scikit_learn":sklearn.__version__,"platform":platform.platform()}}
    se={"header":h,"cells":{}};print("building cells ...",flush=True);cs=gen_cells(w)
    for ne,X in real_embeddings().items():
        n,d=X.shape;aw=[np.asarray(X,dtype=np.float64)]
        m=write_cell(w/f"real_{ne}_raw",aw,{"kind":"real embedding, as stored","normalized":False,"defensible":False,"cohort":ne,"caveat":"single realization, not a Type I cell; run to record whether the methods execute","pr_sphere_mean":float(nc.participation_ratio(nc.l2_normalize(X)))})
        cs.append({"name":f"real_{ne}_raw",**m});a=[nc.l2_normalize(np.asarray(X,dtype=np.float64))]
        m=write_cell(w/f"real_{ne}_l2",a,{"kind":"real embedding, L2-normalized as clustered","normalized":True,"defensible":False,"cohort":ne,"caveat":"rows on the unit sphere; reported as inapplicable","pr_sphere_mean":float(nc.participation_ratio(nc.l2_normalize(X)))})
        cs.append({"name":f"real_{ne}_l2",**m});print(f"  built real_{ne}: {n} x {d}",flush=True)
    for c in cs:
        p=w/c["name"];r=run_r(p);rs=read_results(p);se["cells"][c["name"]]={"manifest":c,"r_run":r,"results":rs}
        if rs:
            l="  ".join((f"{k.replace("p_", "")}: {rs[k]["n_ran"]}/{rs[k]["n_attempted"]} ran"+(f", rej {rs[k]["rejection_rate"]:.3f}"if rs[k]["rejection_rate"] is not None else "") for k in("p_hier_iso","p_hier_gencov","p_kmeans_iso","p_kmeans_gencov")))
            print(f"{c["name"]:22s} {l}  ({r["wall_s"]:.0f}s)",flush=True)
        else:print(f"{c["name"]:22s} NO RESULTS rc={r["returncode"]} {r["stderr"][-300:]}",flush=True)
        json.dump(se,open(o,"w"),indent=1)
    se["header"]["total_wall_s"]=round(time.time()-t,1);json.dump(se,open(o,"w"),indent=1);shutil.rmtree(w,ignore_errors=True);print(f"\ntotal wall {se["header"]["total_wall_s"] / 60:.1f} min")
    print(f"wrote {o}")
if __name__=="__main__":main()
