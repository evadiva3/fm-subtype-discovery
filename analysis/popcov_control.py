#Population-covariance control: does C's behaviour survive a known Sigma?
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
_R=Path(__file__).resolve().parents[1];_NC=_R/"experiments"/"ds002748_mdd"/"propagated_null"
if str(_NC) not in sys.path:sys.path.insert(0,str(_NC))
import nullcal as nc
import ov_common as ov
D=ov.D;PR_TARGET=4.5;NDS=400;N_DRAWS=200;DEEP_DRAWS=2000;N_INIT=ov.N_INIT;K_GRID=ov.K_GRID;ALPHA=ov.ALPHA;TOL=ov.TOL;N_WORKERS=47;SEED_BASE=6000000;EXTRA_BASES=[16000000,26000000,36000000]
DRAW_OFFSET=7919;CELLS=[("cell1_exp_n28","exp",28,0),("cell2_exp_n200","exp",200,1),("cell3_two_block_n28","two_block",28,10),("cell4_two_block_n200","two_block",200,11)]
OUT=_R/"results"/"popcov_control_independent_v7.json";CSV=_R/"results"/"popcov_control_independent_v7.csv"
CSV_COLS=["cell","spectrum","n","d","regime","pr_target","pr_mean","pr_sd","param_name","param_value","tuned_pr","tuned_ok","n_datasets","n_draws","rej_M","count_M","ci_M_lo","ci_M_hi","rej_C","count_C","ci_C_lo","ci_C_hi","rej_M_known","count_M_known","ci_M_known_lo","ci_M_known_hi","rej_C_known","count_C_known","ci_C_known_lo","ci_C_known_hi","mean_pM","mean_pC","mean_pM_known","mean_pC_known","ks_M","ks_C","ks_M_known","ks_C_known","M_only","C_only","M_only_known","C_only_known","shift_est","shift_sign_est","shift_known","shift_sign_known","verified","total_C_only","total_datasets","deep_persist_frac","wall_s"]
_LAM=None;_C=None
def sphere_spectrum(lam,ulo=-50.0,uhi=90.0,npts=40001):
    from scipy.integrate import simpson
    lam=np.asarray(lam,dtype=np.float64);u=np.linspace(ulo,uhi,npts);t=np.exp(u);tl=np.outer(t,lam);g=np.exp(-0.5*np.log1p(2.0*tl).sum(axis=1));i=lam[None,:]/(1.0+2.0*tl)*(g*t)[:,None]
    return simpson(i,x=u,axis=0)
def gen_and_cov(n,d,lam,c,seed):
    r=np.random.default_rng(seed);Z=r.standard_normal((n,d));Q,_=np.linalg.qr(r.standard_normal((d,d)));Xn=nc.l2_normalize(Z@(np.sqrt(lam)[:,None]*Q.T));return(Xn,Q*c@Q.T)
def tune(spectrum,n):
    if spectrum=="exp":
        p,pr,ok,ne=ov.bisect_param(n,D,lambda p:ov.spec_exp(D,p),0.02,100000.0,PR_TARGET,tol=TOL);return(ov.spec_exp(D,p),"tau",float(p),pr,ok,ne)
    if spectrum=="two_block":
        p,pr,ok,ne=ov.tune_two_block(n,D,PR_TARGET);l=ov.spec_two_block(D,p["m"],p["eps"]);return(l,"m/eps","%d/%.6g"%(p["m"],p["eps"]),pr,ok,ne)
    raise ValueError(spectrum)
def _init(lam,c):
    global _LAM,_C;_LAM,_C=(lam,c)
def screen(args):
    n,i,sd,ns=args
    try:
        Xn,sa=gen_and_cov(n,D,_LAM,_C,sd);g=nc.min_cluster_guard(n);s,k,_=nc.best_silhouette(Xn,k_grid=K_GRID,min_size=g,n_init=N_INIT,random_state=i)
        if k is None:return{"skip":True,"i":i,"seed":sd}
        kw=dict(k_grid=K_GRID,n_draws=ns,min_size=g,n_init=N_INIT,seed=sd+DRAW_OFFSET,observed=(s,k));o={"i":i,"seed":sd,"k":int(k),"obs":float(s),"pr":float(nc.participation_ratio(Xn))}
        for t,c in(("est",None),("known",sa)):
            rM=nc.null_test(Xn,reproject=False,match_selection=False,cov=c,**kw);rG=nc.null_test(Xn,reproject=True,match_selection=False,cov=c,**kw)
            rC=nc.null_test(Xn,reproject=True,match_selection=True,cov=c,**kw);o["pM_"+t]=float(rM.p);o["pC_"+t]=float(rC.p);o["ambient_"+t]=float(rM.null_mean);o["sphere_"+t]=float(rG.null_mean)
            o["ambient_sd_"+t]=float(rM.null_sd);o["nullC_"+t]=float(rC.null_mean);o["nullC_sd_"+t]=float(rC.null_sd);o["degC_"+t]=int(rC.n_degenerate)
        return o
    except Exception:return{"error":traceback.format_exc()}
def deep(args):
    n,i,sd,ns=args
    try:
        Xn,sa=gen_and_cov(n,D,_LAM,_C,sd);g=nc.min_cluster_guard(n);s,k,_=nc.best_silhouette(Xn,k_grid=K_GRID,min_size=g,n_init=N_INIT,random_state=i)
        if k is None:return{"skip":True,"seed":sd}
        kw=dict(k_grid=K_GRID,n_draws=ns,min_size=g,n_init=N_INIT,seed=sd+DRAW_OFFSET,observed=(s,k));o={"i":i,"seed":sd}
        for t,c in(("est",None),("known",sa)):
            o["pM_"+t]=float(nc.null_test(Xn,reproject=False,match_selection=False,cov=c,**kw).p);o["pC_"+t]=float(nc.null_test(Xn,reproject=True,match_selection=True,cov=c,**kw).p)
        return o
    except Exception:return{"error":traceback.format_exc()}
def arm(p,label):
    c=int((p<ALPHA).sum())
    return dict(label=label,n_datasets=int(p.size),events=c,rejection_rate=float(c/p.size),wilson95=list(nc._wilson(c,p.size)),mean_p=float(p.mean()),ks_D_vs_uniform=float(nc._ks_uniform(p)))
def discordance(pm,pc,a="M",b="C"):return{a+"_only":int(((pm<ALPHA)&(pc>=ALPHA)).sum()),b+"_only":int(((pc<ALPHA)&(pm>=ALPHA)).sum()),"n_discordant":int(((pm<ALPHA)!=(pc<ALPHA)).sum())}
def discordant_rows(good):
    o=[]
    for r in good:
        ts=[]
        for t in("est","known"):
            if(r["pM_"+t]<ALPHA)!=(r["pC_"+t]<ALPHA):
                ts.append(t)
        if ts:o.append(dict(i=r["i"],seed=r["seed"],modes=ts,pM_est=r["pM_est"],pC_est=r["pC_est"],pM_known=r["pM_known"],pC_known=r["pC_known"]))
    return o
def run_pass(pool,n,base,off,nds,n_draws):
    a=[(n,i,base+1000*i+off,n_draws) for i in range(nds)];rs=list(pool.imap_unordered(screen,a,chunksize=1));g=[r for r in rs if "pM_est" in r];e=[r["error"] for r in rs if "error" in r]
    return(g,sum((1 for r in rs if r.get("skip"))),e)
def replication_record(good,base):
    p={t:np.array([r[t] for r in good]) for t in("pM_est","pC_est","pM_known","pC_known")};rc=dict(base=base,n_datasets=len(good))
    for tg in("est","known"):
        d=discordance(p["pM_"+tg],p["pC_"+tg]);rc["C_only_"+tg]=d["C_only"];rc["M_only_"+tg]=d["M_only"];rc["rej_M_"+tg]=float((p["pM_"+tg]<ALPHA).mean())
        rc["rej_C_"+tg]=float((p["pC_"+tg]<ALPHA).mean())
    rc["C_only_mode_events"]=rc["C_only_est"]+rc["C_only_known"];rc["C_only_either"]=int(np.count_nonzero((p["pC_est"]<ALPHA)&(p["pM_est"]>=ALPHA)|(p["pC_known"]<ALPHA)&(p["pM_known"]>=ALPHA)))
    return rc
def summarize(good,n_skipped,errs,wall,**extra):
    p={t:np.array([r[t] for r in good]) for t in("pM_est","pC_est","pM_known","pC_known")};pr=np.array([r["pr"] for r in good])
    a1={"M_est":arm(p["pM_est"],"M, covariance estimated"),"C_est":arm(p["pC_est"],"C, covariance estimated"),"M_known":arm(p["pM_known"],"M, known population Sigma"),"C_known":arm(p["pC_known"],"C, known population Sigma")}
    s={}
    for tg in("est","known"):
        a=float(np.mean([r["ambient_"+tg] for r in good]));sh=float(np.mean([r["sphere_"+tg] for r in good]));sd=float(np.mean([r["ambient_sd_"+tg] for r in good]))
        ps=int(sum((1 for r in good if r["sphere_"+tg]>r["ambient_"+tg])))
        s[tg]=dict(null_mean_ambient=a,null_mean_sphere=sh,null_sd_ambient=sd,shift=sh-a,shift_in_sd=(sh-a)/sd if sd>0 else None,sign=int(np.sign(sh-a)),datasets_with_positive_shift=ps,n_datasets=len(good))
    o=dict(pr_mean=float(pr.mean()),pr_sd=float(pr.std(ddof=1)),n_datasets=len(good),n_skipped=int(n_skipped),n_errors=len(errs),error_sample=errs[0] if errs else None,n_draws=N_DRAWS,rej_M=a1["M_est"]["rejection_rate"],count_M=a1["M_est"]["events"],ci_M=a1["M_est"]["wilson95"],rej_C=a1["C_est"]["rejection_rate"],count_C=a1["C_est"]["events"],ci_C=a1["C_est"]["wilson95"],mean_pM=a1["M_est"]["mean_p"],mean_pC=a1["C_est"]["mean_p"],ks_M=a1["M_est"]["ks_D_vs_uniform"],ks_C=a1["C_est"]["ks_D_vs_uniform"],null_mean_M=s["est"]["null_mean_ambient"],null_sd_M=s["est"]["null_sd_ambient"],null_mean_C=float(np.mean([r["nullC_est"] for r in good])),null_sd_C=float(np.mean([r["nullC_sd_est"] for r in good])),obs_mean=float(np.mean([r["obs"] for r in good])),M_only=discordance(p["pM_est"],p["pC_est"])["M_only"],C_only=discordance(p["pM_est"],p["pC_est"])["C_only"],degenerate_C=int(sum((r["degC_est"] for r in good))),k_counts={int(k):int(sum((1 for r in good if r["k"]==k))) for k in sorted({r["k"] for r in good})},arms=a1,discordance=dict(estimated=discordance(p["pM_est"],p["pC_est"]),known=discordance(p["pM_known"],p["pC_known"])),shift=s,wall_s=round(wall,1),failed=False)
    o.update(extra);return o
def verify(pool,n,off,cell,screen0,wall_start):
    ov.assert_seed_blocks([SEED_BASE]+EXTRA_BASES,NDS);rs=[replication_record(screen0,SEED_BASE)];rs[0]["note"]="reproduces the original cell"
    dc=[dict(replication=0,**g) for g in discordant_rows(screen0)]
    for j,b in enumerate(EXTRA_BASES,start=1):
        gd,sk,e=run_pass(pool,n,b,off,NDS,N_DRAWS);r=replication_record(gd,b);r["replication"]=j;r["n_skipped"]=sk;r["n_errors"]=len(e);rs.append(r)
        dc+=[dict(replication=j,**g) for g in discordant_rows(gd)]
        print("      rep%d base=%d M/C est %.3f/%.3f known %.3f/%.3f C-only est %d known %d  (%.0fs)"%(j,b,r["rej_M_est"],r["rej_C_est"],r["rej_M_known"],r["rej_C_known"],r["C_only_est"],r["C_only_known"],time.time()-wall_start),flush=True)
    d=None
    if dc:
        a=[(n,g["i"],g["seed"],DEEP_DRAWS) for g in dc];r1=list(pool.imap_unordered(deep,a,chunksize=1));by={(r["seed"],r["i"]):r for r in r1 if "pM_est" in r};p={"est":0,"known":0}
        for g in dc:
            r=by.get((g["seed"],g["i"]))
            if r is None:
                g["deep"]=None;continue
            g["deep"]={}
            for tg in("est","known"):
                pm,pc=(r["pM_"+tg],r["pC_"+tg]);w=(g["pM_"+tg]<ALPHA)!=(g["pC_"+tg]<ALPHA);nw=(pm<ALPHA)!=(pc<ALPHA);s=nw and w and((pm<ALPHA)==(g["pM_"+tg]<ALPHA))
                g["deep"][tg]=dict(pM=pm,pC=pc,persists=bool(s));p[tg]+=int(s and w)
        td={t:sum((1 for g in dc if t in g["modes"])) for t in("est","known")}
        d=dict(draws=DEEP_DRAWS,n_retested=len(dc),tested_estimated=td["est"],tested_known=td["known"],persisting_estimated=p["est"],persisting_known=p["known"],surviving_fraction_estimated=p["est"]/td["est"] if td["est"] else None,surviving_fraction_known=p["known"]/td["known"] if td["known"] else None)
        print("      deep: %d discordant re-tested at %d draws; survive est %d/%d known %d/%d"%(len(dc),DEEP_DRAWS,p["est"],td["est"],p["known"],td["known"]),flush=True)
    tc=sum((r["C_only_est"]+r["C_only_known"] for r in rs));tn=sum((r["n_datasets"] for r in rs))
    cell["verification"]=dict(triggered=True,n_replications=len(rs),seed_bases=[r["base"] for r in rs],replications=rs,total_C_only=tc,total_datasets=tn,C_only_by_mode={tg:dict(count=sum((r["C_only_"+tg] for r in rs)),rate=sum((r["C_only_"+tg] for r in rs))/tn,ci=list(nc._wilson(sum((r["C_only_"+tg] for r in rs)),tn))) for tg in("est","known")},n_discordant=len(dc),discordant=dc,discordant_truncated=False,deep=d)
def csv_row(key,c):
    v=c.get("verification") or{};d=v.get("deep") or{}
    r={"cell":key,"spectrum":c["spectrum"],"n":c["n"],"d":c["d"],"regime":c["regime"],"pr_target":c["pr_target"],"pr_mean":c["pr_mean"],"pr_sd":c["pr_sd"],"param_name":c["param_name"],"param_value":c["param_value"],"tuned_pr":c["tuned_pr"],"tuned_ok":c["tuned_ok"],"n_datasets":c["n_datasets"],"n_draws":c["n_draws"],"mean_pM":c["arms"]["M_est"]["mean_p"],"mean_pC":c["arms"]["C_est"]["mean_p"],"mean_pM_known":c["arms"]["M_known"]["mean_p"],"mean_pC_known":c["arms"]["C_known"]["mean_p"],"ks_M":c["arms"]["M_est"]["ks_D_vs_uniform"],"ks_C":c["arms"]["C_est"]["ks_D_vs_uniform"],"ks_M_known":c["arms"]["M_known"]["ks_D_vs_uniform"],"ks_C_known":c["arms"]["C_known"]["ks_D_vs_uniform"],"M_only":c["discordance"]["estimated"]["M_only"],"C_only":c["discordance"]["estimated"]["C_only"],"M_only_known":c["discordance"]["known"]["M_only"],"C_only_known":c["discordance"]["known"]["C_only"],"shift_est":c["shift"]["est"]["shift"],"shift_sign_est":c["shift"]["est"]["sign"],"shift_known":c["shift"]["known"]["shift"],"shift_sign_known":c["shift"]["known"]["sign"],"verified":bool(v.get("triggered")),"total_C_only":v.get("total_C_only",""),"total_datasets":v.get("total_datasets",""),"deep_persist_frac":d.get("surviving_fraction_estimated",""),"wall_s":c["wall_s"]}
    for a,p in(("M_est","M"),("C_est","C"),("M_known","M_known"),("C_known","C_known")):
        r["rej_"+p]=c["arms"][a]["rejection_rate"];r["count_"+p]=c["arms"][a]["events"];r["ci_"+p+"_lo"]=c["arms"][a]["wilson95"][0];r["ci_"+p+"_hi"]=c["arms"][a]["wilson95"][1]
    return r
def write_csv(state):
    rs=[",".join(CSV_COLS)]
    for k,c in state["cells"].items():
        if c.get("failed"):continue
        r=csv_row(k,c);rs.append(",".join((str(round(r.get(col,""),6) if isinstance(r.get(col),float) else r.get(col,"")) for col in CSV_COLS)))
    CSV.write_text('\n'.join(rs)+'\n')
def main():
    t0=time.time()
    import scipy
    import sklearn
    cg=dict(experiment="population covariance control",purpose="separate C's behaviour from the sample covariance estimate",written_utc=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),d=D,pr_target=PR_TARGET,pr_tolerance=TOL,n_datasets=NDS,n_draws=N_DRAWS,deep_draws=DEEP_DRAWS,n_init=N_INIT,k_grid=list(K_GRID),alpha=ALPHA,arms=["M_est","C_est","M_known","C_known"],seed_base=SEED_BASE,extra_bases=EXTRA_BASES,seed_rule="dataset i of a cell uses seed_base + 1000*i + cell_offset; its draws use that + %d, shared by all six tests"%DRAW_OFFSET,pairing="the four arms see identical datasets and identical draw seeds",population_covariance="Q diag(c) Q.T, c_j = E[lambda_j z_j^2 / sum_k lambda_k z_k^2] by quadrature; sum(c) = 1 exactly",verification="a cell with any C-only rejection is re-run at three further seed bases and every discordance in either mode re-tested at %d draws"%DEEP_DRAWS,versions={"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"scikit_learn":sklearn.__version__,"platform":platform.platform(),"n_workers":N_WORKERS})
    se=ov.load_state(OUT,"popcov_control",cg);se["config"]=cg
    for k,s,n,o in CELLS:
        if k in se["cells"] and(not se["cells"][k].get("failed")):
            print("skip %s"%k,flush=True);continue
        tc=time.time();l,p,p1,tr,tk,te=tune(s,n);c=sphere_spectrum(l);cm=float(c.sum());assert abs(cm-1.0)<1e-06,"quadrature failed: sum(c) = %r"%cm;pe,_=gen_and_cov(n,D,l,c,SEED_BASE+o)
        assert np.array_equal(pe,ov.gen(n,D,l,SEED_BASE+o)),"generation diverges from ov_common.gen"
        print("[%s] %s n=%d  %s=%s  tuned PR %.3f (target %.1f, ok=%s)  sum(c)=%.10f"%(k,s,n,p,p1,tr,PR_TARGET,tk,cm),flush=True)
        with Pool(N_WORKERS,initializer=_init,initargs=(l,c))as pl:
            g,sk,e=run_pass(pl,n,SEED_BASE,o,NDS,N_DRAWS)
            if not g:
                se["cells"][k]=dict(failed=True,n_errors=len(e),error_sample=e[0] if e else None,spectrum=s,n=n,d=D);ov.save_state(OUT,se);continue
            cl=summarize(g,sk,e,time.time()-tc,spectrum=s,n=n,d=D,regime="n < d" if n<D else "n > d",pr_target=PR_TARGET,param_name=p,param_value=p1,tuned_pr=tr,tuned_ok=bool(tk),tune_note=te,sphere_spectrum_sum=cm,sphere_spectrum_top10=[float(x) for x in np.sort(c)[::-1][:10]],cell_offset=o)
            a=cl["arms"]
            print("      M est %.4f (%d)  C est %.4f (%d)  |  M known %.4f (%d)  C known %.4f (%d)  |  PR %.2f"%(a["M_est"]["rejection_rate"],a["M_est"]["events"],a["C_est"]["rejection_rate"],a["C_est"]["events"],a["M_known"]["rejection_rate"],a["M_known"]["events"],a["C_known"]["rejection_rate"],a["C_known"]["events"],cl["pr_mean"]),flush=True)
            print("      shift est %+.5f (%+d)  known %+.5f (%+d)  |  C-only est %d known %d"%(cl["shift"]["est"]["shift"],cl["shift"]["est"]["sign"],cl["shift"]["known"]["shift"],cl["shift"]["known"]["sign"],cl["discordance"]["estimated"]["C_only"],cl["discordance"]["known"]["C_only"]),flush=True)
            t=cl["discordance"]["estimated"]["C_only"]>0 or cl["discordance"]["known"]["C_only"]>0
            if t:
                print("      C-only present: running the verification protocol",flush=True);verify(pl,n,o,cl,g,tc)
            else:cl["verification"]=dict(triggered=False,reason="no C-only rejection in either covariance mode")
        cl["wall_s"]=round(time.time()-tc,1);se["cells"][k]=cl;ov.save_state(OUT,se);write_csv(se);print('      cell done in %.0fs, total %.0fs\n'%(cl["wall_s"],time.time()-t0),flush=True)
    se["config"]["total_wall_s"]=round(time.time()-t0,1);ov.save_state(OUT,se);write_csv(se);print("popcov control done in %.0fs"%(time.time()-t0));print("wrote %s"%OUT);print("wrote %s"%CSV)
if __name__=="__main__":main()
