#Population-covariance control at participation ratio 9: cells 5 and 6.
import argparse
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
_HERE=Path(__file__).resolve().parent
if str(_HERE) not in sys.path:sys.path.insert(0,str(_HERE))
import popcov_control as pcc
import nullcal as nc
import ov_common as ov
_R=_HERE.parents[0];D=ov.D;PR_TARGET=9.0;N=180;NDS=400;SCREEN_NDS=100;N_DRAWS=200;DEEP_DRAWS=pcc.DEEP_DRAWS;ALPHA=ov.ALPHA;N_INIT=ov.N_INIT;K_GRID=ov.K_GRID;TOL=ov.TOL;N_WORKERS=pcc.N_WORKERS
DRAW_OFFSET=pcc.DRAW_OFFSET;SEED_BASE=pcc.SEED_BASE;EXTRA_BASES=pcc.EXTRA_BASES;SCREEN_LO,SCREEN_HI=(0.05,0.95);SHIFT_BAND=0.03
CELLS=[("cell5_exp_n180","exp",N,20),("cell6_two_block_n180","two_block",N,21)];OUT=_R/"results"/"popcov_control_pr9_independent_v7.json";CSV=_R/"results"/"popcov_control_pr9_independent_v7.csv"
SCREEN_OUT=_R/"results"/"popcov_control_pr9_screen_independent_v7.json";LADDER_OUT=_R/"results"/"popcov_control_pr9_screen_ladder.json";PRIOR=_R/"results"/"popcov_control_independent_v7.json"
pcc.PR_TARGET=PR_TARGET;pcc.NDS=NDS;pcc.N_DRAWS=N_DRAWS;pcc.CELLS=CELLS;pcc.OUT=OUT;pcc.CSV=CSV
def screen_est(args):
    n,i,sd,ns=args
    try:
        Xn,_=pcc.gen_and_cov(n,D,pcc._LAM,pcc._C,sd);g=nc.min_cluster_guard(n);s,k,_=nc.best_silhouette(Xn,k_grid=K_GRID,min_size=g,n_init=N_INIT,random_state=i)
        if k is None:return{"skip":True,"i":i,"seed":sd}
        kw=dict(k_grid=K_GRID,n_draws=ns,min_size=g,n_init=N_INIT,seed=sd+DRAW_OFFSET,observed=(s,k));rM=nc.null_test(Xn,reproject=False,match_selection=False,cov=None,**kw)
        rC=nc.null_test(Xn,reproject=True,match_selection=True,cov=None,**kw)
        return{"i":i,"seed":sd,"k":int(k),"obs":float(s),"pr":float(nc.participation_ratio(Xn)),"pM_est":float(rM.p),"pC_est":float(rC.p)}
    except Exception:return{"error":traceback.format_exc()}
def classify_shift(shift_in_sd,band=SHIFT_BAND):
    if shift_in_sd is None:return "indeterminate"
    if abs(shift_in_sd)<band:return "indeterminate"
    return "positive" if shift_in_sd>0 else "negative"
def annotate_shift(cell):
    for t in("est","known"):
        s=cell.get("shift",{}).get(t)
        if s is None:continue
        s["shift_band_null_sd"]=SHIFT_BAND;s["shift_class"]=classify_shift(s.get("shift_in_sd"))
    return cell
def prior_shift_classification():
    if not PRIOR.exists():return None
    p=json.loads(PRIOR.read_text());r=[]
    for k,c in p["cells"].items():
        for t in("est","known"):
            s=c["shift"][t]
            r.append(dict(cell=k,spectrum=c["spectrum"],n=c["n"],d=c["d"],mode="estimated" if t=="est" else "known",null_mean_ambient=s["null_mean_ambient"],null_mean_sphere=s["null_mean_sphere"],null_sd_ambient=s["null_sd_ambient"],shift=s["shift"],shift_in_sd=s["shift_in_sd"],sign_bit=s["sign"],shift_class=classify_shift(s["shift_in_sd"]),datasets_with_positive_shift=s["datasets_with_positive_shift"],n_datasets=s["n_datasets"]))
    return dict(source="results/popcov_control_independent_v7.json",note="derived here from the archived artifact; that artifact is not modified by this run",band_null_sd=SHIFT_BAND,rows=r)
def prepare(spectrum,n,off,key,pr_target=None):
    pr_target=PR_TARGET if pr_target is None else pr_target;l,p,p1,tr,t,ne=pcc.tune(spectrum,n);c=pcc.sphere_spectrum(l);cm=float(c.sum());assert abs(cm-1.0)<1e-06,"quadrature failed: sum(c) = %r"%cm
    assert round(cm,10)==1.0,"trace identity fails at ten decimals: %r"%cm;pe,_=pcc.gen_and_cov(n,D,l,c,SEED_BASE+off)
    assert np.array_equal(pe,ov.gen(n,D,l,SEED_BASE+off)),"generation diverges from ov_common.gen"
    print("[%s] %s n=%d  %s=%s  tuned PR %.3f (target %.1f, ok=%s)  sum(c)=%.10f"%(key,spectrum,n,p,p1,tr,pr_target,t,cm),flush=True)
    return(l,c,dict(param_name=p,param_value=p1,tuned_pr=tr,tuned_ok=bool(t),tune_note=ne,sphere_spectrum_sum=cm,sphere_spectrum_top10=[float(x) for x in np.sort(c)[::-1][:10]]))
def _versions():
    import scipy
    import sklearn
    return{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"scikit_learn":sklearn.__version__,"platform":platform.platform(),"n_workers":N_WORKERS}
def screen_one(key,spectrum,n,off,pr_target):
    tc=time.time();pcc.PR_TARGET=pr_target
    try:l,c,t=prepare(spectrum,n,off,key,pr_target)
    finally:pcc.PR_TARGET=PR_TARGET
    with Pool(N_WORKERS,initializer=pcc._init,initargs=(l,c))as p:
        a1=[(n,i,SEED_BASE+1000*i+off,N_DRAWS) for i in range(SCREEN_NDS)];rs=list(p.imap_unordered(screen_est,a1,chunksize=1))
    g=[r for r in rs if "pM_est" in r];e=[r["error"] for r in rs if "error" in r];n1=sum((1 for r in rs if r.get("skip")))
    if not g:return dict(failed=True,spectrum=spectrum,n=n,d=D,pr_target=pr_target,n_errors=len(e),error_sample=e[0] if e else None)
    pM=np.array([r["pM_est"] for r in g]);pC=np.array([r["pC_est"] for r in g]);pr=np.array([r["pr"] for r in g])
    cl=dict(spectrum=spectrum,n=n,d=D,cell_offset=off,regime="n < d" if n<D else "n > d",pr_target=pr_target,pr_mean=float(pr.mean()),pr_sd=float(pr.std(ddof=1)),n_datasets=len(g),n_skipped=int(n1),n_errors=len(e),error_sample=e[0] if e else None,n_draws=N_DRAWS,arms=dict(M_est=pcc.arm(pM,"M, covariance estimated"),C_est=pcc.arm(pC,"C, covariance estimated")),discordance=pcc.discordance(pM,pC),k_counts={int(k):int(sum((1 for r in g if r["k"]==k))) for k in sorted({r["k"] for r in g})},wall_s=round(time.time()-tc,1))
    cl.update(t)
    for a in("M_est","C_est"):
        r=cl["arms"][a]["rejection_rate"];cl["arms"][a]["in_accept_band"]=bool(SCREEN_LO<r<SCREEN_HI)
    cl["in_accept_band"]=all((cl["arms"][a]["in_accept_band"] for a in("M_est","C_est")))
    print("      M est %.4f (%d)  C est %.4f (%d)  |  PR %.2f  |  in band: %s  (%.0fs)"%(cl["arms"]["M_est"]["rejection_rate"],cl["arms"]["M_est"]["events"],cl["arms"]["C_est"]["rejection_rate"],cl["arms"]["C_est"]["events"],cl["pr_mean"],cl["in_accept_band"],time.time()-tc),flush=True)
    return cl
def run_ladder(spectrum,prs,off,label):
    t0=time.time()
    s=dict(experiment="population covariance control at PR 9, screening ladder",purpose="locate where the %s family leaves the saturation ceiling at n = %d, d = %d"%(spectrum,N,D),written_utc=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),config=dict(d=D,n=N,spectrum=spectrum,pr_targets=list(prs),pr_tolerance=TOL,n_datasets=SCREEN_NDS,n_draws=N_DRAWS,n_init=N_INIT,k_grid=list(K_GRID),alpha=ALPHA,arms=["M_est","C_est"],covariance="estimated only",seed_base=SEED_BASE,cell_offset=off,seed_rule="dataset i uses seed_base + 1000*i + %d; its draws use that + %d"%(off,DRAW_OFFSET),prefix_note="every rung uses the cell offset of the candidate cell, so the adopted rung's screen is a prefix of the full run at that conditioning",accept_band=[SCREEN_LO,SCREEN_HI],versions=_versions()),rungs={})
    if LADDER_OUT.exists():
        pv=json.loads(LADDER_OUT.read_text())
        if pv.get("config",{}).get("spectrum")==spectrum:
            s["rungs"]=pv.get("rungs",{});s["config"]["pr_targets"]=sorted(set(pv["config"].get("pr_targets",[]))|set(prs))
    for pt in prs:
        ky="%s_pr%g"%(label,pt);s["rungs"][ky]=screen_one(ky,spectrum,N,off,pt);LADDER_OUT.write_text(json.dumps(s,indent=1))
    s["rungs"]=dict(sorted(s["rungs"].items(),key=lambda kv:kv[1].get("pr_target",0.0)));p=[k for k,c in s["rungs"].items() if c.get("in_accept_band")];s["rungs_in_band"]=p
    s["wall_s_this_invocation"]=round(time.time()-t0,1);s["total_wall_s"]=round(sum((c.get("wall_s",0.0) for c in s["rungs"].values())),1);LADDER_OUT.write_text(json.dumps(s,indent=1))
    print('\nrungs inside the band: %s'%(p or "none"));print("wrote %s"%LADDER_OUT);return p
def run_screen():
    t0=time.time()
    se=dict(experiment="population covariance control at PR 9, screening pass",purpose="check that cells 5 and 6 are off the ceiling before spending the full run",written_utc=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),config=dict(d=D,n=N,pr_target=PR_TARGET,pr_tolerance=TOL,n_datasets=SCREEN_NDS,n_draws=N_DRAWS,n_init=N_INIT,k_grid=list(K_GRID),alpha=ALPHA,arms=["M_est","C_est"],covariance="estimated only",seed_base=SEED_BASE,seed_rule="dataset i uses seed_base + 1000*i + cell_offset; its draws use that + %d"%DRAW_OFFSET,prefix_note="datasets i = 0 .. %d of the full run's seed rule, so the screen is a prefix of the full run and not an extra set of datasets"%(SCREEN_NDS-1),accept_band=[SCREEN_LO,SCREEN_HI],versions=_versions()),cells={})
    for k,s,n,o in CELLS:
        se["cells"][k]=screen_one(k,s,n,o,PR_TARGET);SCREEN_OUT.write_text(json.dumps(se,indent=1))
    ok=all((c.get("in_accept_band") for c in se["cells"].values()))
    se["decision"]=dict(all_rates_in_band=bool(ok),band=[SCREEN_LO,SCREEN_HI],rule="proceed to the full run only if every screened rate falls strictly between %.2f and %.2f"%(SCREEN_LO,SCREEN_HI))
    se["total_wall_s"]=round(time.time()-t0,1);SCREEN_OUT.write_text(json.dumps(se,indent=1));print('\nscreen decision: proceed=%s'%ok);print("wrote %s"%SCREEN_OUT);return ok
def run_full():
    t0=time.time()
    cg=dict(experiment="population covariance control at PR 9",purpose="the n > d direction question, off the saturation ceiling of cells 2 and 4",written_utc=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),d=D,n=N,pr_target=PR_TARGET,pr_tolerance=TOL,n_datasets=NDS,n_draws=N_DRAWS,deep_draws=DEEP_DRAWS,n_init=N_INIT,k_grid=list(K_GRID),alpha=ALPHA,arms=["M_est","C_est","M_known","C_known"],seed_base=SEED_BASE,extra_bases=EXTRA_BASES,cell_offsets={k:o for k,_,_,o in CELLS},seed_rule="dataset i of a cell uses seed_base + 1000*i + cell_offset; its draws use that + %d, shared by all six tests"%DRAW_OFFSET,pairing="the four arms see identical datasets and identical draw seeds",population_covariance="Q diag(c) Q.T, c_j = E[lambda_j z_j^2 / sum_k lambda_k z_k^2] by quadrature; sum(c) = 1 exactly, asserted to ten decimals",verification="a cell with any C-only rejection is re-run at three further seed bases and every discordance in either mode re-tested at %d draws"%DEEP_DRAWS,shift_classification="positive, negative, or indeterminate where |shift| < %.2f ambient null sd"%SHIFT_BAND,protocol_source="analysis/popcov_control.py, imported; only the conditioning constants differ",supersedes="cells 2 and 4 of results/popcov_control_independent_v7.json, which saturate; that artifact is not modified",versions=_versions())
    se=ov.load_state(OUT,"popcov_control_pr9",cg);se["config"]=cg
    for ky,s,n,of in CELLS:
        if ky in se["cells"] and(not se["cells"][ky].get("failed")):
            print("skip %s"%ky,flush=True);continue
        tc=time.time();l,c,to=prepare(s,n,of,ky)
        with Pool(N_WORKERS,initializer=pcc._init,initargs=(l,c))as p:
            g,sk,e=pcc.run_pass(p,n,SEED_BASE,of,NDS,N_DRAWS)
            if not g:
                se["cells"][ky]=dict(failed=True,n_errors=len(e),error_sample=e[0] if e else None,spectrum=s,n=n,d=D);ov.save_state(OUT,se);continue
            cl=pcc.summarize(g,sk,e,time.time()-tc,spectrum=s,n=n,d=D,regime="n < d" if n<D else "n > d",pr_target=PR_TARGET,cell_offset=of,**to);annotate_shift(cl);a=cl["arms"]
            print("      M est %.4f (%d)  C est %.4f (%d)  |  M known %.4f (%d)  C known %.4f (%d)  |  PR %.2f"%(a["M_est"]["rejection_rate"],a["M_est"]["events"],a["C_est"]["rejection_rate"],a["C_est"]["events"],a["M_known"]["rejection_rate"],a["M_known"]["events"],a["C_known"]["rejection_rate"],a["C_known"]["events"],cl["pr_mean"]),flush=True)
            print("      shift est %+.6f (%+.3f sd, %s)  known %+.6f (%+.3f sd, %s)  |  C-only est %d known %d"%(cl["shift"]["est"]["shift"],cl["shift"]["est"]["shift_in_sd"],cl["shift"]["est"]["shift_class"],cl["shift"]["known"]["shift"],cl["shift"]["known"]["shift_in_sd"],cl["shift"]["known"]["shift_class"],cl["discordance"]["estimated"]["C_only"],cl["discordance"]["known"]["C_only"]),flush=True)
            t=cl["discordance"]["estimated"]["C_only"]>0 or cl["discordance"]["known"]["C_only"]>0
            if t:
                print("      C-only present: running the verification protocol",flush=True);pcc.verify(p,n,of,cl,g,tc)
            else:cl["verification"]=dict(triggered=False,reason="no C-only rejection in either covariance mode")
        cl["wall_s"]=round(time.time()-tc,1);se["cells"][ky]=cl;ov.save_state(OUT,se);pcc.write_csv(se);print('      cell done in %.0fs, total %.0fs\n'%(cl["wall_s"],time.time()-t0),flush=True)
    se["prior_cells_shift_classification"]=prior_shift_classification();se["config"]["total_wall_s"]=round(time.time()-t0,1);ov.save_state(OUT,se);pcc.write_csv(se)
    print("popcov control at PR 9 done in %.0fs"%(time.time()-t0));print("wrote %s"%OUT);print("wrote %s"%CSV)
def main():
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument("--screen",action="store_true");g.add_argument("--ladder",action="store_true")
    g.add_argument("--full",action="store_true");ap.add_argument("--spectrum",default="two_block",choices=["exp","two_block"],help="ladder only: which spectrum family to walk")
    ap.add_argument("--prs",default="12,15,20,25,30",help="ladder only: comma-separated participation-ratio targets");a=ap.parse_args()
    if a.screen:run_screen()
    elif a.ladder:
        o=dict(((s,o) for _,s,_,o in CELLS))[a.spectrum];l=dict(((s,k) for k,s,_,_ in CELLS))[a.spectrum];run_ladder(a.spectrum,[float(x) for x in a.prs.split(",")],o,l)
    else:run_full()
if __name__=="__main__":main()
