#Post hoc check of held-out size across S19, S20 and S22 (not part of any prespecified protocol).
#1.Simulates how often an exactly calibrated statistic meets the prespecified 3-7% interval criterion when its
#cutoff is estimated from 1,000 calibration datasets and evaluated on 1,000 others, using the 199-draw p-value grid.
#2.Pools each statistic's held-out null rejections over all independently calibrated cells, with a Wilson interval.
#3.Compares calibration and evaluation null p-values in every cell with a two-sample Kolmogorov-Smirnov test.
import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[v]="1"
import json,sys
from pathlib import Path
import numpy as np
from scipy import stats
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/"analysis"))
from summarize_detection import cutoff,reject,ci
D=R/"results/revision_20260924"
D25=R/"results/revision_20260925"
STUDIES={"S19":D/"calibrated_detection","S20":D/"asymmetric_detection","S22 n=28":D/"scope_detection_n28","S22 n=100":D/"scope_detection_n100","S25 n=50":D25/"sample_size_n50","S25 n=200":D25/"sample_size_n200"}
A=("W","D4","K","E")

def exact_pass_probability(sims=20000,n=1000,draws=199,seed=20260925):
    rng=np.random.default_rng(seed)
    grid=np.arange(1,draws+2)/(draws+1)
    ok=0;rates=np.empty(sims)
    for b in range(sims):
        cal=rng.choice(grid,size=n);ev=rng.choice(grid,size=n);u=rng.uniform(size=n)
        t,rho=cutoff(cal)
        c=int(reject(ev,u,t,rho).sum());rates[b]=c/n
        lo,hi=ci(c,n);ok+=lo>=.03 and hi<=.07
    return dict(simulations=sims,pass_probability=ok/sims,heldout_rate_sd=float(rates.std()),heldout_rate_mean=float(rates.mean()))

def main():
    out=dict(note="Post hoc; not prespecified. Pools independently calibrated cells that use different cutoffs, so the pooled rate estimates average held-out size.",exact_calibration=exact_pass_probability(),studies={},pooled={},ks={})
    counts={a:[0,0] for a in A};rates={a:[] for a in A};passes=[0,0];ksp=[]
    for name,d in STUDIES.items():
        sp=d/"summary.json"
        if not sp.exists():
            continue
        s=json.loads(sp.read_text(encoding="utf-8"))
        rows=[json.loads(l) for l in (d/"records.jsonl").read_text(encoding="utf-8").splitlines()]
        cells=[k for k in s["cells"] if k.endswith("_null")]
        out["studies"][name]=dict(cells=len(cells)*4,meeting_criterion=sum(s["cells"][k]["arms"][a]["heldout_tolerance_03_07_supported"] for k in cells for a in A))
        passes[0]+=out["studies"][name]["meeting_criterion"];passes[1]+=len(cells)*4
        for k in cells:
            for a in A:
                v=s["cells"][k]["arms"][a];counts[a][0]+=v["calibrated_events"];counts[a][1]+=s["cells"][k]["n"];rates[a].append(v["calibrated_rate"])
            sp_,q=int(k.split("_")[0]),int(k.split("_")[1][1:])
            for a in A:
                cp=[r["p"][a] for r in rows if r["spectrum"]==sp_ and r["q"]==q and r["group"]=="calibration"]
                ep=[r["p"][a] for r in rows if r["spectrum"]==sp_ and r["q"]==q and r["group"]=="null"]
                p=float(stats.ks_2samp(cp,ep).pvalue)
                out["ks"][f"{name} {k} {a}"]=p;ksp.append(p)
    # Each cell's held-out rate varies by about sd even under exact calibration (cutoff estimation plus evaluation).
    # The q = 5 and q = 10 cells share source datasets, so report intervals under independence and under full
    # dependence within those pairs.
    sd=out["exact_calibration"]["heldout_rate_sd"]
    for a in A:
        r=np.array(rates[a]);k=len(r)
        m=float(r.mean())
        out["pooled"][a]=dict(cells=k,mean_heldout_rate=m,interval95_independent=[m-1.96*sd/np.sqrt(k),m+1.96*sd/np.sqrt(k)],interval95_paired_projections=[m-1.96*sd/np.sqrt(k/2),m+1.96*sd/np.sqrt(k/2)],events=counts[a][0],n=counts[a][1])
    e=out["exact_calibration"]["pass_probability"]
    out["criterion"]=dict(cells=passes[1],meeting=passes[0],expected_under_exact_calibration=e*passes[1],binomial_two_sided_p=float(stats.binomtest(passes[0],passes[1],e).pvalue))
    out["ks_summary"]=dict(tests=len(ksp),min_p=min(ksp),below_0_05=sum(p<.05 for p in ksp),expected_below_0_05=.05*len(ksp))
    p=R/"reproducibility/audits/pooled_null_check.json"
    p.write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps({k:out[k] for k in ("exact_calibration","studies","pooled","criterion","ks_summary")},indent=1))

if __name__=="__main__":
    main()
