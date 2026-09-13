from __future__ import annotations
from _paths import CAN as CANON,out
import json,glob
import numpy as np,pandas as pd

OUT=out("propagated_null")
rows=[json.load(open(f)) for f in sorted(glob.glob(str(OUT/"draws"/"*.json")))]
df=pd.DataFrame(rows).sort_values("draw")
df.to_csv(OUT/"propagated_null_draws.csv",index=False)

can=pd.read_csv((CANON/"silhouette-scores.csv"))
kSel=int(can["k_selected_silhouette"].iloc[0])
obs=float(can.loc[can["k"]==kSel,"silhouette_score"].iloc[0])

s=df["silhouette"].to_numpy()
n=len(s)
p=(int((s>=obs).sum())+1)/(n+1)

print("fully propagated null simulate raw timeseries+propagate every transform")
print(f"  draws                       {n}")
print(f"  observed (canonical)        silhouette {obs:.4f} at k={kSel}")
print(f"  null silhouette             mean {s.mean():.4f}  sd {s.std(ddof=1):.4f}")
print(f"                              min {s.min():.4f}  median {np.median(s):.4f}  max {s.max():.4f}")
print(f"  percentiles                 5th {np.percentile(s,5):.4f}   50th {np.percentile(s,50):.4f}"
      f"   95th {np.percentile(s,95):.4f}")
print(f"  draws >= observed           {int((s>=obs).sum())}/{n}")
print(f"  permutation p               {p:.4f}")
print()
kd=df["k_selected"].value_counts().sort_index()
print(f"  null's selected k: {{{', '.join(f'{k}: {v}' for k,v in kd.items())}}}")
print(f"  observed selected k={kSel}")
print()
print("  comparison of the three nulls on the same canonical result:")
print(f"    {'null construction':<44}{'p':>9}{'null mean sil':>16}")
for nm,pv,ms in [("misspecified (no sphere projection)",0.1685,0.2160),
                 ("geometry corrected only",0.4630,0.2420),
                 ("fully corrected (embedding space)",0.7009,0.2591),
                 ("FULLY PROPAGATED (raw data space)",p,float(s.mean()))]:
    print(f"    {nm:<44}{pv:>9.4f}{ms:>16.4f}")

json.dump({"n_draws":n,"observed_silhouette":obs,"observed_k":kSel,
           "null_mean":float(s.mean()),"null_sd":float(s.std(ddof=1)),
           "null_min":float(s.min()),"null_max":float(s.max()),
           "null_median":float(np.median(s)),
           "pct5":float(np.percentile(s,5)),"pct95":float(np.percentile(s,95)),
           "n_ge_observed":int((s>=obs).sum()),"p":p,
           "selected_k_counts":{int(k):int(v) for k,v in kd.items()}},
          open(OUT/"propagated_null_summary.json","w"),indent=2)
print(f"\n  wrote {OUT/'propagated_null_summary.json'}")
