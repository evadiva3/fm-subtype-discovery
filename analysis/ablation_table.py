import pandas as pd
from driver_utils import config, fm_subs, load_enc, rand_enc, emb_attention, emb_mean, cluster_eval

def main():
    fm=fm_subs()
    enc, pool=load_enc()
    renc, rpool=rand_enc()
    conds=[
        ("Full model", emb_attention(enc, pool, fm)),
        ("Mean pooling", emb_mean(enc, fm)),
        ("Untrained encoder", emb_attention(renc, rpool, fm)),
    ]
    rows=[]
    for nm, X in conds:
        r=cluster_eval(X)
        r.pop("label")
        r["condition"]=nm
        rows.append(r)
        print(f"{nm}: sil={r['silhouette']:.6f} k={r['k_sel']} perm_p={r['perm_p']:.4f} guard={r['sizeOk']} eff_rank={r['eff_rank']:.4f} pc1%={r['pc1_pct']:.2f}")
    cols=["condition", "silhouette", "k_sel", "perm_p", "sizeOk", "k_gap", "eff_rank", "pc1_pct", "rank_ceiling"]
    df=pd.DataFrame(rows)[cols]
    out=config.resultsRoot/"ablation_table.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}")
    print(df.to_string(index=False))
    return df

if __name__ == "__main__":
    main()
