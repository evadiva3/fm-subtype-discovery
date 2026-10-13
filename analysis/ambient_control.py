#Ambient-covariance control
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime,timezone
from multiprocessing import Pool
from pathlib import Path
for _v in("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ.setdefault(_v,"1")
import numpy as np
_R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(_R/"analysis"));sys.path.insert(0,str(_R/"experiments"/"ds002748_mdd"/"propagated_null"))
import nullcal as nc
import ov_common as ov
import popcov_control as pcc
D=ov.D;NDS=400;N_DRAWS=200;N_INIT=ov.N_INIT;K_GRID=ov.K_GRID;ALPHA=ov.ALPHA;N_WORKERS=47;ROOT_SEED=20260916;EXPERIMENT_ID=31;ARCHIVE_BASE=6000000;DRAW_OFFSET=7919
OUT=_R/"results"/"ambient_control.json"
CELLS={"cell1_exp_n28":("exp",28,4.5,0,"popcov_control.json","cell1_exp_n28"),"cell2_exp_n200":("exp",200,4.5,1,"popcov_control.json","cell2_exp_n200"),"cell6_two_block_n180":("two_block",180,30.0,21,"popcov_control_cells567.json","cell6_two_block_n180"),"cell3_two_block_n28":("two_block",28,4.5,10,"popcov_control.json","cell3_two_block_n28"),"cell5_exp_n180":("exp",180,9.0,20,"popcov_control_cells567.json","cell5_exp_n180"),"cell7_exp_n119":("exp",119,6.0,22,"popcov_control_cells567.json","cell7_exp_n119")}
FRESH=["cell1_exp_n28","cell6_two_block_n180","cell3_two_block_n28","cell5_exp_n180","cell7_exp_n119"]
REPLAY=["cell1_exp_n28","cell2_exp_n200","cell6_two_block_n180","cell3_two_block_n28","cell5_exp_n180","cell7_exp_n119"];ARMS=["M_est","C_est","M_known","C_known","C_ambient","C_ambient_meanest"]
_G={}
def prepare(key):
    s,n,t,o,at,a=CELLS[key];r=json.loads((_R/"results"/at).read_text())["cells"][a];pcc.PR_TARGET=t;l,p,pe,tr,ok,_=pcc.tune(s,n)
    if pe!=r["param_value"] or tr!=r["tuned_pr"]:raise AssertionError("%s: tuning does not reproduce the archive (%r %r vs %r %r)"%(key,pe,tr,r["param_value"],r["tuned_pr"]))
    c=pcc.sphere_spectrum(l);assert round(float(c.sum()),10)==1.0;return(l,c,r,dict(param_name=p,param_value=pe,tuned_pr=tr,tuned_ok=bool(ok),archive=at,archive_key=a))
def _init(lam,c):_G["lam"],_G["c"]=(lam,c)
def generate(n,seed):
    l,c=(_G["lam"],_G["c"]);r=np.random.default_rng(seed);Z=r.standard_normal((n,D));Q,_=np.linalg.qr(r.standard_normal((D,D)));Xn=nc.l2_normalize(Z@(np.sqrt(l)[:,None]*Q.T));a=Q*l@Q.T
    return(Xn,Q*c@Q.T,a)
def worker(args):
    n,i,d,dd,o=args
    try:
        Xn,st,at=generate(n,d);g=nc.min_cluster_guard(n);s,k,_=nc.best_silhouette(Xn,k_grid=K_GRID,min_size=g,n_init=N_INIT,random_state=o)
        if k is None:return{"skip":True,"i":i}
        kw=dict(k_grid=K_GRID,n_draws=N_DRAWS,min_size=g,n_init=N_INIT,seed=dd,observed=(s,k));ev=np.linalg.eigvalsh(np.cov(Xn,rowvar=False))[::-1]
        ot={"i":i,"data_seed":str(d),"draw_seed":str(dd),"k":int(k),"obs":float(s),"pr":float(nc.participation_ratio(Xn)),"pc1":float(ev[0]/ev.sum()),"mean_norm":float(np.linalg.norm(Xn.mean(axis=0)))}
        rs={"M_est":dict(reproject=False,match_selection=False),"C_est":dict(reproject=True,match_selection=True),"M_known":dict(reproject=False,match_selection=False,cov=st),"C_known":dict(reproject=True,match_selection=True,cov=st),"C_ambient":dict(reproject=True,match_selection=True,cov=at,mean=np.zeros(D)),"C_ambient_meanest":dict(reproject=True,match_selection=True,cov=at/np.trace(at))}
        for am,a in rs.items():
            r=nc.null_test(Xn,**a,**kw);ot["p_"+am]=float(r.p);ot["nullmean_"+am]=float(r.null_mean);ot["nullsd_"+am]=float(r.null_sd);ot["exceed_"+am]=int(r.n_exceed)
            ot["degenerate_"+am]=int(r.n_degenerate)
        return ot
    except Exception:return{"error":traceback.format_exc(),"i":i}
def paired(p_a,p_b):
    a,b=(p_a<ALPHA,p_b<ALPHA);return dict(a_only=int((a&~b).sum()),b_only=int((b&~a).sum()))
def summarize(rows):
    p={a:np.array([r["p_"+a] for r in rows]) for a in ARMS}
    o=dict(n_datasets=len(rows),pr_mean=float(np.mean([r["pr"] for r in rows])),pr_sd=float(np.std([r["pr"] for r in rows],ddof=1)),arms={a:pcc.arm(p[a],a) for a in ARMS},null_mean={a:float(np.mean([r["nullmean_"+a] for r in rows])) for a in ARMS},degenerate_draws={a:int(sum((r["degenerate_"+a] for r in rows))) for a in ARMS})
    o["paired"]={"C_est_vs_C_ambient":paired(p["C_est"],p["C_ambient"]),"C_known_vs_C_ambient":paired(p["C_known"],p["C_ambient"]),"C_ambient_vs_C_ambient_meanest":paired(p["C_ambient"],p["C_ambient_meanest"]),"M_est_vs_C_est":paired(p["M_est"],p["C_est"]),"M_known_vs_C_known":paired(p["M_known"],p["C_known"])}
    return o
def reproduction(summary,rec):
    n={"M_est":"M_est","C_est":"C_est","M_known":"M_known","C_known":"C_known"};r={}
    for m,t in n.items():
        r[m]=dict(archived_events=rec["arms"][t]["events"],replay_events=summary["arms"][m]["events"],archived_mean_p=rec["arms"][t]["mean_p"],replay_mean_p=summary["arms"][m]["mean_p"])
        r[m]["identical_events"]=r[m]["archived_events"]==r[m]["replay_events"]
    return r
def run_cell(key,mode):
    sm,n,_,o,_,_=CELLS[key];l,c,rc,te=prepare(key)
    if mode=="fresh":
        a1=[(n,i,ov.hashed_seed(ROOT_SEED,EXPERIMENT_ID,o,0,i,0),ov.hashed_seed(ROOT_SEED,EXPERIMENT_ID,o,0,i,1),i) for i in range(NDS)];s=[a[2] for a in a1]+[a[3] for a in a1];assert len(set(s))==len(s)
    else:a1=[(n,i,ARCHIVE_BASE+1000*i+o,ARCHIVE_BASE+1000*i+o+DRAW_OFFSET,i) for i in range(NDS)]
    t=time.time()
    with Pool(N_WORKERS,initializer=_init,initargs=(l,c))as p:rs=list(p.imap_unordered(worker,a1,chunksize=1))
    e=[r for r in rs if "error" in r]
    if e:raise RuntimeError(e[0]["error"])
    r1=sorted((r for r in rs if "p_C_ambient" in r),key=lambda r:r["i"])
    cl=dict(spectrum=sm,n=n,d=D,cell_offset=o,mode=mode,n_skipped=sum((1 for r in rs if r.get("skip"))),wall_s=round(time.time()-t,1),**te,**summarize(r1))
    if mode=="replay":cl["reproduction"]=reproduction(cl,rc)
    cl["rows"]=r1;a=cl["arms"];print("[%s %s] PR %.2f  "%(mode,key,cl["pr_mean"])+"  ".join(("%s %.4f(%d)"%(x,a[x]["rejection_rate"],a[x]["events"]) for x in ARMS))+"  [%.0fs]"%cl["wall_s"],flush=True)
    return cl
def main():
    w=sys.argv[1] if len(sys.argv)>1 else "all"
    import scipy
    import sklearn
    s=json.loads(OUT.read_text()) if OUT.exists() else dict(experiment="ambient covariance control",cells={})
    s["config"]=dict(written_utc=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),d=D,n_datasets=NDS,n_draws=N_DRAWS,n_init=N_INIT,k_grid=list(K_GRID),alpha=ALPHA,arms=ARMS,arm_definitions={"M_est":"fitted mean and covariance, ambient draws, fixed observed k","C_est":"fitted mean and covariance, re-projected draws, guarded search","M_known":"fitted mean, population second moment of normalized rows, fixed k","C_known":"fitted mean, population second moment of normalized rows, re-projected, search","C_ambient":"mean zero, ambient generating covariance Q diag(lambda) Q.T, re-projected, search: the generating law of the normalized rows","C_ambient_meanest":"fitted mean, ambient covariance scaled to trace one, re-projected, search"},fresh_seed_rule="data ov_common.hashed_seed(%d, %d, cell_offset, 0, i, 0); draws the same address with purpose 1; observed search random_state=i"%(ROOT_SEED,EXPERIMENT_ID),replay_seed_rule="data %d + 1000*i + cell_offset; draws that + %d; observed search random_state=i, as archived"%(ARCHIVE_BASE,DRAW_OFFSET),verification="C-only verification protocol not run; see module docstring",versions={"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"scikit_learn":sklearn.__version__,"platform":platform.platform(),"n_workers":N_WORKERS})
    j=[]
    if w in("fresh","all"):j+=[(k,"fresh") for k in FRESH]
    if w in("replay","all"):j+=[(k,"replay") for k in REPLAY]
    for ky,m in j:
        n="%s__%s"%(m,ky)
        if n in s["cells"]:
            print("skip %s"%n,flush=True);continue
        s["cells"][n]=run_cell(ky,m);OUT.write_text(json.dumps(s,indent=1))
    print("wrote %s"%OUT)
if __name__=="__main__":main()
