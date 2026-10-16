from __future__ import annotations
import os
for v in("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[v]="1"
from _paths import RUNS,out
import json
import numpy as np,pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

OUT=out("audit_gap_fills")

print("\nf.6: re-render figures 1-3")
from analysis.figures import figure_gen
fg=figure_gen()
print(f"  output dir: {fg.dire}")
for nm,fn in(("fig1_null_silhouette","plot_null_silh"),
             ("fig2_ablation","plot_ablation"),
             ("fig3_gap_statistic","plot_gap")):
    try:
        getattr(fg,fn)()
        p=Path(fg.dire)/(nm+".png")
        print(f"  {nm:<24} ok   {p.stat().st_size/1024:7.1f} KB")
    except Exception as e:
        print(f"  {nm:<24} failed  {type(e).__name__}: {e}")

print("\n0.4: calibration-adjusted p-values")
pn=np.load((RUNS/"calibration_paired"/"p_corrected.npy"))
print(f"  null p-distribution: {len(pn)} datasets under H0, "
      f"mean {pn.mean():.4f}, median {np.median(pn):.4f}")

def adj(p):
    return float((np.sum(pn<=p)+1)/(len(pn)+1))

obs=[("canonical FM (1,000 draws)",0.6773226773226774),
     ("canonical FM (20,000 draws)",0.7009)]
sm=pd.read_csv((RUNS/"stability_summary.csv")).sort_values("run")
for _,r in sm.iterrows():
    obs.append((f"stability run{int(r['run']):02d}",float(r["perm_p"])))

rows=[]
print(f"\n  {'result':<32}{'raw p':>9}{'adjusted p':>13}{'sig@.05?':>11}")
for nm,p in obs:
    a=adj(p)
    rows.append({"result":nm,"raw_p":p,"calibrated_p":a,"sig_raw":p<0.05,
                 "sig_calibrated":a<0.05})
    print(f"  {nm:<32}{p:>9.4f}{a:>13.4f}{'yes' if a<0.05 else 'no':>11}")

df=pd.DataFrame(rows)
df.to_csv(OUT/"calibration_adjusted_pvalues.csv",index=False)
print(f"\n  raw significant: {int(df.sig_raw.sum())}/{len(df)}")
print(f"  adjusted significant: {int(df.sig_calibrated.sum())}/{len(df)}")
print(f"  smallest raw p {df.raw_p.min():.4f} -> adjusted {df.loc[df.raw_p.idxmin(),'calibrated_p']:.4f}")
json.dump({"n_null_datasets":int(len(pn)),"null_p_mean":float(pn.mean()),
           "results":rows},open(OUT/"calibration_adjusted_pvalues.json","w"),indent=2)
print(f"  wrote {OUT/'calibration_adjusted_pvalues.csv'}")
