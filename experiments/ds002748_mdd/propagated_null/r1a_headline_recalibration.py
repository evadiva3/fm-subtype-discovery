#Phase 1A: recalibrate the headline cell at 10,000 datasets by 1,000 draws.
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
N=28;D=119;PR_TARGET=4.36;PR_TOL=0.02;TUNE_M=200;N_BLOCKS=5;NDS_PER_BLOCK=2000;N_DRAWS=1000;N_INIT=5;K_GRID=(2,3,4,5,6);ALPHA=0.05;SEED_ROOT=4100000;BLOCK_STRIDE=10000000;_LAM=None
def _init(lam):
    global _LAM;_LAM=lam
def one(args):
    b,i,sd=args
    try:
        X=ov.gen(N,D,_LAM,sd);g=nc.min_cluster_guard(N);s,k,_=nc.best_silhouette(X,k_grid=K_GRID,min_size=g,n_init=N_INIT,random_state=i)
        if k is None:return{"block":b,"i":i,"skipped":True}
        kw=dict(k_grid=K_GRID,n_draws=N_DRAWS,min_size=g,n_init=N_INIT,seed=sd+7,observed=(s,k));rM=nc.null_test(X,reproject=False,match_selection=False,**kw)
        rC=nc.null_test(X,reproject=True,match_selection=True,**kw)
        return{"block":b,"i":i,"pr":float(nc.participation_ratio(X)),"obs":float(s),"k":int(k),"pM":float(rM.p),"pC":float(rC.p),"nullM":float(rM.null_mean),"nullC":float(rC.null_mean),"sdM":float(rM.null_sd),"sdC":float(rC.null_sd)}
    except Exception as e:return{"block":b,"i":i,"error":f"{type(e).__name__}: {e}"}
def clopper_pearson(c,n,alpha=0.05):
    from scipy.stats import beta
    lo=0.0 if c==0 else float(beta.ppf(alpha/2,c,n-c+1));hi=1.0 if c==n else float(beta.ppf(1-alpha/2,c+1,n-c));return(lo,hi)
def arm_stats(p,label):
    from scipy.stats import kstest
    c=int(np.sum(p<ALPHA));ks=kstest(p,"uniform")
    return{"label":label,"n_datasets":int(p.size),"events":c,"rejection_rate":float(c/p.size),"wilson95":list(nc._wilson(c,p.size)),"clopper_pearson95":list(clopper_pearson(c,p.size)),"mean_p":float(p.mean()),"median_p":float(np.median(p)),"ks_D_vs_uniform":float(ks.statistic),"ks_p":float(ks.pvalue),"ks_D_nullcal":float(nc._ks_uniform(p)),"p_deciles":[float(x) for x in np.quantile(p,np.arange(0,11)/10)],"p_histogram_20bins":[int(x) for x in np.histogram(p,bins=20,range=(0,1))[0]]}
def main():
    sp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ");o1=ov.ROOT/"results"/"phase1"/sp;o1.mkdir(parents=True,exist_ok=True);o=o1/"r1a_headline_recalibration.json";t=time.time()
    print("tuning tau ...",flush=True);tu,p,ok,n=ov.bisect_param(N,D,lambda t:ov.spec_exp(D,t),0.2,60.0,PR_TARGET,tol=PR_TOL,m=TUNE_M);d=abs(p-PR_TARGET)/PR_TARGET
    print(f"tau={tu:.6f}  tuned sphere PR={p:.4f}  deviation={d * 100:.2f}%  ok={ok} {n}",flush=True);l=ov.spec_exp(D,tu)
    import scipy
    import sklearn
    h={"experiment":"R1A headline recalibration","purpose":"replace the four-event 0.4 percent point estimate with an interval","written_utc":sp,"n":N,"d":D,"n_datasets":N_BLOCKS*NDS_PER_BLOCK,"n_blocks":N_BLOCKS,"n_datasets_per_block":NDS_PER_BLOCK,"n_draws":N_DRAWS,"n_init":N_INIT,"k_grid":list(K_GRID),"alpha":ALPHA,"spectrum":"exp(-i/tau) in a random orthonormal basis, redrawn per dataset","tau":float(tu),"pr_target":PR_TARGET,"pr_tolerance_fraction":PR_TOL,"pr_tuned_on_throwaway_samples":float(p),"pr_tuning_deviation_fraction":float(d),"pr_tuning_samples":TUNE_M,"tune_ok":bool(ok),"tune_note":n,"seed_root":SEED_ROOT,"block_stride":BLOCK_STRIDE,"seed_rule":"dataset i of block b uses seed_root + b*block_stride + 1000*i; its draws use that + 7","pairing":"M and C evaluated on identical datasets against nulls fitted to that dataset, sharing the draw seed","versions":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"scikit_learn":sklearn.__version__,"platform":platform.platform(),"n_cores_used":47}}
    a1=[(b,i,SEED_ROOT+b*BLOCK_STRIDE+1000*i) for b in range(N_BLOCKS) for i in range(NDS_PER_BLOCK)];print(f"running {len(a1)} datasets at {N_DRAWS} draws, both arms ...",flush=True);rs=[]
    t0=time.time()
    with Pool(47,initializer=_init,initargs=(l,))as po:
        for j,r in enumerate(po.imap_unordered(one,a1,chunksize=4),1):
            rs.append(r)
            if j%500==0:
                el=time.time()-t0;print(f"  {j}/{len(a1)}  {el / 60:.1f} min elapsed, {el / j * (len(a1) - j) / 60:.1f} min remaining",flush=True)
    g=[r for r in rs if "pM" in r];pM=np.array([r["pM"] for r in g]);pC=np.array([r["pC"] for r in g]);np.save(o1/"r1a_p_misspecified.npy",pM);np.save(o1/"r1a_p_corrected.npy",pC)
    from scipy.stats import binomtest
    m=int(np.sum((pM<ALPHA)&(pC>=ALPHA)));c=int(np.sum((pC<ALPHA)&(pM>=ALPHA)));nd=m+c;mc={"M_only":m,"C_only":c,"n_discordant":nd,"exact_binomial_p":float(binomtest(m,nd,0.5).pvalue) if nd else None}
    if nd:
        c2=(abs(m-c)-1)**2/nd
        from scipy.stats import chi2 as chi2dist
        mc["mcnemar_chi2_cc"]=float(c2);mc["mcnemar_chi2_cc_p"]=float(chi2dist.sf(c2,1))
    bs=[]
    for b in range(N_BLOCKS):
        sl=[r for r in g if r["block"]==b];bM=np.array([r["pM"] for r in sl]);bC=np.array([r["pC"] for r in sl])
        bs.append({"block":b,"seed_base":SEED_ROOT+b*BLOCK_STRIDE,"n_datasets":int(bM.size),"M_rate":float(np.mean(bM<ALPHA)),"M_events":int(np.sum(bM<ALPHA)),"C_rate":float(np.mean(bC<ALPHA)),"C_events":int(np.sum(bC<ALPHA)),"C_wilson95":list(nc._wilson(int(np.sum(bC<ALPHA)),bC.size))})
    sr={}
    for a in("M","C"):
        v=np.array([b[a+"_rate"] for b in bs])
        sr[a]={"mean":float(v.mean()),"sd":float(v.std(ddof=1)),"min":float(v.min()),"max":float(v.max()),"range_ratio":float(v.max()/v.min()) if v.min()>0 else None}
    ps=np.array([r["pr"] for r in g])
    se={"header":h,"realized_pr":{"mean":float(ps.mean()),"sd":float(ps.std(ddof=1)),"min":float(ps.min()),"max":float(ps.max()),"deviation_from_target_fraction":float(abs(ps.mean()-PR_TARGET)/PR_TARGET)},"n_skipped":sum((1 for r in rs if r.get("skipped"))),"n_errors":sum((1 for r in rs if "error" in r)),"observed_statistic":{"mean":float(np.mean([r["obs"] for r in g])),"sd":float(np.std([r["obs"] for r in g],ddof=1))},"null_means":{"M":float(np.mean([r["nullM"] for r in g])),"C":float(np.mean([r["nullC"] for r in g])),"M_sd":float(np.mean([r["sdM"] for r in g])),"C_sd":float(np.mean([r["sdC"] for r in g]))},"M":arm_stats(pM,"M, unmatched"),"C":arm_stats(pC,"C, fully matched"),"mcnemar":mc,"blocks":bs,"seed_level_variance":sr,"selected_k_counts":{str(k):int(sum((1 for r in g if r["k"]==k))) for k in K_GRID}}
    se["header"]["total_wall_s"]=round(time.time()-t,1);json.dump(se,open(o,"w"),indent=1);print('\n'+"="*62)
    print(f"realized sphere PR {ps.mean():.4f} (target {PR_TARGET}, {abs(ps.mean() - PR_TARGET) / PR_TARGET * 100:.2f}% off)")
    for a in("M","C"):
        s=se[a]
        print(f"{s["label"]:20s} {s["rejection_rate"]:.4f}  Wilson [{s["wilson95"][0]:.4f}, {s["wilson95"][1]:.4f}]  CP [{s["clopper_pearson95"][0]:.4f}, {s["clopper_pearson95"][1]:.4f}]  {s["events"]} events  KS D {s["ks_D_vs_uniform"]:.4f}")
    print(f"McNemar: M-only {m}, C-only {c}, exact p {mc["exact_binomial_p"]}")
    for a in("M","C"):
        v=sr[a];print(f"seed-level {a}: mean {v["mean"]:.4f} sd {v["sd"]:.4f} range {v["min"]:.4f} to {v["max"]:.4f}")
    print(f"total wall {se["header"]["total_wall_s"] / 60:.1f} min");print(f"wrote {o}")
if __name__=="__main__":main()
