import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from driver_utils import config, fm_subs, fm_ids, load_enc, rand_enc, emb_attention, severity_y, loo_predict, score_r, perm_r

# no primary k and dont add one
K_SWEEP=(3,5,7)

def knn_fp(kk):
    return lambda Xtr, ytr, Xte: KNeighborsRegressor(n_neighbors=kk).fit(Xtr, ytr).predict(Xte)
def main():
    fm=fm_subs(); ids=fm_ids(fm)
    y=severity_y(ids)
    enc, pool=load_enc()
    renc, rpool=rand_enc()
    T=emb_attention(enc, pool, fm)
    U=emb_attention(renc, rpool, fm)
    rows=[]
    for nm, X in (("trained", T), ("random-encoder", U)):
        for kk in K_SWEEP:
            fp=knn_fp(kk)
            preds=loo_predict(fp, X, y)
            r, r2=score_r(preds, y)
            pp=perm_r(fp, X, y, r)
            rows.append({"encoder": nm, "k_neighbors": kk, "r": r, "R2": r2, "perm_p": pp,
                         "n": len(y)})
            print(f"{nm} k={kk}: r={r:+.4f} R2={r2:+.4f} perm_p={pp:.4f}")
    df=pd.DataFrame(rows)
    out=config.resultsRoot/"knn_probe.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}")
    print(df.to_string(index=False))
    w=df.pivot(index="k_neighbors", columns="encoder", values="r")
    winner={kk: ("trained" if w.loc[kk,"trained"]>=w.loc[kk,"random-encoder"] else "random-encoder")
            for kk in K_SWEEP}
    print(f"\nwinner by k: {winner}")
    if len(set(winner.values()))>1:
        print("warning: trained-vs-random verdict changes across the neighbour sweep, report all three")
    return df

if __name__ == "__main__":
    main()
