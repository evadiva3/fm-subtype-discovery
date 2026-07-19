import numpy as np
import pandas as pd
from driver_utils import config, fm_subs, load_enc, rand_enc, emb_attention, eff_rank

def main():
    fm=fm_subs()
    enc, pool=load_enc()
    renc, rpool=rand_enc()
    T=emb_attention(enc, pool, fm)
    U=emb_attention(renc, rpool, fm)
    np.save(config.clusterOutput/"trained_fm_embeddings.npy", T)
    np.save(config.clusterOutput/"untrained_fm_embeddings.npy", U)
    ceil=min(U.shape)-1
    rows=[]
    for nm, X in (("trained", T), ("untrained", U)):
        er, pc1=eff_rank(X)
        rows.append({"encoder": nm, "dim": X.shape[1], "eff_rank": er, "rank_ceiling": ceil, "pc1_pct": pc1})
        print(f"{nm}: eff_rank={er:.4f}/{ceil} pc1%={pc1:.2f}")
    df=pd.DataFrame(rows)
    out=config.resultsRoot/"effective_rank.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}")
    print(df.to_string(index=False))
    return df

if __name__ == "__main__":
    main()
