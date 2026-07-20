import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from driver_utils import (config, fm_subs, fm_ids, load_enc, emb_attention, severity_y, cluster_eval, loo_predict, score_r, _ridge)
def sil_ci_noreplace(Xraw, k, frac=0.8, seed=None):
    seed=config.randomSeed if seed is None else seed
    Xn=Xraw/(np.linalg.norm(Xraw, axis=1, keepdims=True)+1e-8)
    n=len(Xn); m=int(round(frac*n))
    rng=np.random.default_rng(seed)
    vals=np.empty(config.bootstrapNResamples)
    for b in range(config.bootstrapNResamples):
        idx=rng.choice(n, size=m, replace=False)
        sub=Xn[idx]
        lab=KMeans(n_clusters=k, n_init=config.kmeansNInit, random_state=config.randomSeed).fit_predict(sub)
        vals[b]=silhouette_score(sub, lab)
    return vals
def r_ci(X, y, seed=None):
    seed=config.randomSeed if seed is None else seed
    n=len(y)
    rng=np.random.default_rng(seed)
    vals=np.empty(config.bootstrapNResamples)
    for b in range(config.bootstrapNResamples):
        idx=rng.integers(0, n, n)
        preds=loo_predict(_ridge, X[idx], y[idx])
        vals[b]=np.corrcoef(preds, y[idx])[0, 1]
    return vals

def main():
    fm=fm_subs(); ids=fm_ids(fm)
    y=severity_y(ids)
    enc, pool=load_enc()
    X=emb_attention(enc, pool, fm)
    ce=cluster_eval(X)
    k=ce["k_sel"]; sil_pt=ce["silhouette"]
    sv=sil_ci_noreplace(X, k)
    r_pt, _=score_r(loo_predict(_ridge, X, y), y)
    rv=r_ci(X, y)
    rows=[
        {"metric": f"silhouette_k{k}", "point": sil_pt, "ci_lo": float(np.percentile(sv, 2.5)),
         "ci_hi": float(np.percentile(sv, 97.5)), "resample": "subsample_without_replacement", "frac": 0.8, "n_boot": config.bootstrapNResamples},
        {"metric": "ridge_r", "point": r_pt, "ci_lo": float(np.percentile(rv, 2.5)),
         "ci_hi": float(np.percentile(rv, 97.5)), "resample": "bootstrap_with_replacement", "frac": 1.0, "n_boot": config.bootstrapNResamples},
    ]
    df=pd.DataFrame(rows)
    out=config.resultsRoot/"bootstrap_cis.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}")
    print(df.to_string(index=False))
    return df

if __name__ == "__main__":
    main()
