import os
import numpy as np
import pandas as pd
from driver_utils import config, fm_subs, load_enc, rand_enc, emb_attention, eff_rank

EMB_ATOL=1e-5

def _save_guarded(path,arr):
    if path.exists():
        old=np.load(path)
        if old.shape==arr.shape:
            dev=float(np.abs(old-arr).max())
            if dev<=EMB_ATOL:
                print(f"  {path.name}: unchanged within {EMB_ATOL:g} (max dev {dev:.2e}); keeping the canonical file")
                return
        else:
            dev=float("inf")
        if os.environ.get("FM_ALLOW_EMBEDDING_OVERWRITE")!="1":
            raise RuntimeError(
                f"refusing to overwrite {path}: the regenerated embedding differs from "
                f"the canonical one by {dev:.2e}, beyond the {EMB_ATOL:g} tolerance. "
                f"Four null drivers read this file, so changing it invalidates every "
                f"downstream p-value. If this change is intended, re-run with "
                f"FM_ALLOW_EMBEDDING_OVERWRITE=1 and regenerate the nulls, PROVENANCE, "
                f"and the affected paper numbers.")
        print(f"  {path.name}: OVERWRITTEN (forced, max dev {dev:.2e})")
    np.save(path,arr)

def main():
    fm=fm_subs()
    enc,pool=load_enc()
    renc,rpool=rand_enc()
    T=emb_attention(enc,pool,fm)
    U=emb_attention(renc,rpool,fm)
    _save_guarded(config.clusterOutput/"trained_fm_embeddings.npy",T)
    _save_guarded(config.clusterOutput/"untrained_fm_embeddings.npy",U)
    ceil=min(U.shape)-1
    rows=[]
    for nm,X in (("trained",T),("untrained",U)):
        er,pc1=eff_rank(X)
        rows.append({"encoder":nm,"dim":X.shape[1],"eff_rank":er,"rank_ceiling":ceil,"pc1_pct":pc1})
        print(f"{nm}: eff_rank={er:.4f}/{ceil} pc1%={pc1:.2f}")
    df=pd.DataFrame(rows)
    out=config.resultsRoot/"effective_rank.csv"
    df.to_csv(out,index=False)
    print(f"wrote {out}")
    print(df.to_string(index=False))
    return df

if __name__=="__main__":
    main()
