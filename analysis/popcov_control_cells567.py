#Population-covariance control, cells 5, 6 and 7: the n/d gradient.
import argparse
import json
import os
import sys
import time
from datetime import datetime,timezone
from multiprocessing import Pool
from pathlib import Path
for _v in("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ.setdefault(_v,"1")
import numpy as np
_HERE=Path(__file__).resolve().parent
if str(_HERE) not in sys.path:sys.path.insert(0,str(_HERE))
import popcov_control as pcc
import popcov_control_pr9 as p9
import nullcal as nc
import ov_common as ov
_R=_HERE.parents[0];D=ov.D;NDS=400;SCREEN_NDS=100;N_DRAWS=200;DEEP_DRAWS=pcc.DEEP_DRAWS;ALPHA=ov.ALPHA;N_INIT=ov.N_INIT;K_GRID=ov.K_GRID;TOL=ov.TOL;N_WORKERS=pcc.N_WORKERS;DRAW_OFFSET=pcc.DRAW_OFFSET
SEED_BASE=pcc.SEED_BASE;EXTRA_BASES=pcc.EXTRA_BASES;SCREEN_LO,SCREEN_HI=(p9.SCREEN_LO,p9.SCREEN_HI);SHIFT_BAND=p9.SHIFT_BAND
CELLS=[("cell5_exp_n180","exp",180,9.0,20),("cell6_two_block_n180","two_block",180,30.0,21),("cell7_exp_n119","exp",119,6.0,22)];SCREEN_DEFAULT=["cell7_exp_n119"]
OUT=_R/"results"/"popcov_control_cells567_independent_v7.json";CSV=_R/"results"/"popcov_control_cells567_independent_v7.csv"
SCREEN_OUT=_R/"results"/"popcov_control_cells567_screen_independent_v7.json";PRIOR=_R/"results"/"popcov_control_independent_v7.json";pcc.NDS=NDS;pcc.N_DRAWS=N_DRAWS
pcc.CELLS=[(k,s,n,off) for k,s,n,_,off in CELLS];pcc.OUT=OUT;pcc.CSV=CSV
def prepare_cell(key,spectrum,n,off,pr_target):
    pcc.PR_TARGET=pr_target
    try:return p9.prepare(spectrum,n,off,key,pr_target)
    finally:pcc.PR_TARGET=pr_target
def _cells(keys=None):
    if keys is None:return CELLS
    by={c[0]:c for c in CELLS};m=[k for k in keys if k not in by]
    if m:raise SystemExit("unknown cell key(s): %s"%", ".join(m))
    return[by[k] for k in keys]
def run_screen(keys):
    t0=time.time();t=_cells(keys)
    se=dict(experiment="population covariance control, cells 5 to 7, screening pass",purpose="confirm the new cell is off the ceiling before spending the full run",written_utc=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),config=dict(d=D,pr_tolerance=TOL,n_datasets=SCREEN_NDS,n_draws=N_DRAWS,n_init=N_INIT,k_grid=list(K_GRID),alpha=ALPHA,arms=["M_est","C_est"],covariance="estimated only",seed_base=SEED_BASE,cells_screened=[c[0] for c in t],seed_rule="dataset i uses seed_base + 1000*i + cell_offset; its draws use that + %d"%DRAW_OFFSET,prefix_note="datasets i = 0 .. %d of the full run's seed rule, so the screen is a prefix of the full run and not an extra set of datasets"%(SCREEN_NDS-1),accept_band=[SCREEN_LO,SCREEN_HI],versions=p9._versions()),cells={})
    for k,s,n,p,o in t:
        pcc.PR_TARGET=p
        try:se["cells"][k]=p9.screen_one(k,s,n,o,p)
        finally:pcc.PR_TARGET=p
        SCREEN_OUT.write_text(json.dumps(se,indent=1))
    ok=all((c.get("in_accept_band") for c in se["cells"].values()))
    se["decision"]=dict(all_rates_in_band=bool(ok),band=[SCREEN_LO,SCREEN_HI],rule="proceed to the full run only if every screened rate falls strictly between %.2f and %.2f"%(SCREEN_LO,SCREEN_HI))
    se["total_wall_s"]=round(time.time()-t0,1);SCREEN_OUT.write_text(json.dumps(se,indent=1));print('\nscreen decision: proceed=%s'%ok);print("wrote %s"%SCREEN_OUT);return ok
def run_full():
    t0=time.time()
    cg=dict(experiment="population covariance control, cells 5, 6 and 7",purpose="locate the n/d transition in the direction of the covariance effect, at n/d = 1.51 and n/d = 1.00",written_utc=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),d=D,n_datasets=NDS,n_draws=N_DRAWS,deep_draws=DEEP_DRAWS,pr_tolerance=TOL,n_init=N_INIT,k_grid=list(K_GRID),alpha=ALPHA,arms=["M_est","C_est","M_known","C_known"],cells={k:dict(spectrum=s,n=n,d=D,n_over_d=round(n/D,4),pr_target=pr,cell_offset=o) for k,s,n,pr,o in CELLS},seed_base=SEED_BASE,extra_bases=EXTRA_BASES,seed_rule="dataset i of a cell uses seed_base + 1000*i + cell_offset; its draws use that + %d, shared by all six tests"%DRAW_OFFSET,pairing="the four arms see identical datasets and identical draw seeds",population_covariance="Q diag(c) Q.T, c_j = E[lambda_j z_j^2 / sum_k lambda_k z_k^2] by quadrature; sum(c) = 1 exactly, asserted to ten decimals",verification="a cell with any C-only rejection is re-run at three further seed bases and every discordance in either mode re-tested at %d draws; a cell with none records triggered=false, which is the protocol behaving correctly rather than a missing result"%DEEP_DRAWS,shift_classification="positive, negative, or indeterminate where |shift| < %.2f ambient null sd"%SHIFT_BAND,protocol_source="analysis/popcov_control.py and analysis/popcov_control_pr9.py, imported; only the per-cell conditioning differs",screening_provenance="cells 5 and 6 were screened in results/popcov_control_pr9_screen_independent_v7.json and its ladder; cell 7 in results/popcov_control_cells567_screen_independent_v7.json. Every screen is a prefix of the cell it screens, on the same offset and seed rule",versions=p9._versions())
    se=ov.load_state(OUT,"popcov_control_cells567",cg);se["config"]=cg
    for ky,sm,n,pt,o in CELLS:
        if ky in se["cells"] and(not se["cells"][ky].get("failed")):
            print("skip %s"%ky,flush=True);continue
        tc=time.time();l,c,to=prepare_cell(ky,sm,n,o,pt)
        with Pool(N_WORKERS,initializer=pcc._init,initargs=(l,c))as p:
            g,sk,e=pcc.run_pass(p,n,SEED_BASE,o,NDS,N_DRAWS)
            if not g:
                se["cells"][ky]=dict(failed=True,n_errors=len(e),error_sample=e[0] if e else None,spectrum=sm,n=n,d=D);ov.save_state(OUT,se);continue
            cl=pcc.summarize(g,sk,e,time.time()-tc,spectrum=sm,n=n,d=D,n_over_d=round(n/D,4),regime="n < d" if n<D else "n = d" if n==D else "n > d",pr_target=pt,cell_offset=o,**to)
            p9.annotate_shift(cl);a=cl["arms"]
            print("      M est %.4f (%d)  C est %.4f (%d)  |  M known %.4f (%d)  C known %.4f (%d)  |  PR %.2f"%(a["M_est"]["rejection_rate"],a["M_est"]["events"],a["C_est"]["rejection_rate"],a["C_est"]["events"],a["M_known"]["rejection_rate"],a["M_known"]["events"],a["C_known"]["rejection_rate"],a["C_known"]["events"],cl["pr_mean"]),flush=True)
            print("      shift est %+.6f (%+.3f sd, %s)  known %+.6f (%+.3f sd, %s)  |  C-only est %d known %d"%(cl["shift"]["est"]["shift"],cl["shift"]["est"]["shift_in_sd"],cl["shift"]["est"]["shift_class"],cl["shift"]["known"]["shift"],cl["shift"]["known"]["shift_in_sd"],cl["shift"]["known"]["shift_class"],cl["discordance"]["estimated"]["C_only"],cl["discordance"]["known"]["C_only"]),flush=True)
            t=cl["discordance"]["estimated"]["C_only"]>0 or cl["discordance"]["known"]["C_only"]>0
            if t:
                print("      C-only present: running the verification protocol",flush=True);pcc.verify(p,n,o,cl,g,tc)
            else:cl["verification"]=dict(triggered=False,reason="no C-only rejection in either covariance mode; the protocol does not call for a replication and none was forced")
        cl["wall_s"]=round(time.time()-tc,1);se["cells"][ky]=cl;ov.save_state(OUT,se);pcc.write_csv(se);print('      cell done in %.0fs, total %.0fs\n'%(cl["wall_s"],time.time()-t0),flush=True)
    se["prior_cells_shift_classification"]=p9.prior_shift_classification();se["config"]["total_wall_s"]=round(time.time()-t0,1);ov.save_state(OUT,se);pcc.write_csv(se)
    print("popcov control cells 5, 6 and 7 done in %.0fs"%(time.time()-t0));print("wrote %s"%OUT);print("wrote %s"%CSV)
def main():
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument("--screen",action="store_true");g.add_argument("--full",action="store_true")
    ap.add_argument("--cells",default=",".join(SCREEN_DEFAULT),help="screen only: comma-separated cell keys");a=ap.parse_args()
    if a.screen:run_screen([k for k in a.cells.split(",") if k])
    else:run_full()
if __name__=="__main__":main()
