import json,time,warnings
from pathlib import Path
import numpy as np
import ov_common as ov
import nullcal as nc
warnings.filterwarnings("ignore")
OUT=ov.OUTDIR/"e12_representations.json"
DATA=ov.ROOT/"data"/"outputs"
N_DRAWS=2000
SEED=0

STORED={
    "gnn_trained":"trained_fm_embeddings.npy",
    "gnn_untrained":"untrained_fm_embeddings.npy",
    "embeddings_npy":"Embeddings.npy",
    }


def ladder_record(name,X,kind,note=""):
    Xn=nc.l2_normalize(X)
    lr=nc.ladder(X,n_draws=N_DRAWS,n_init=ov.N_INIT,seed=SEED)
    rec=dict(representation=name,kind=kind,note=note,
        shape=list(X.shape),
        pr_unnormalized=float(nc.participation_ratio(X)),
        pr_sphere=float(nc.participation_ratio(X,normalize=True)),
        observed=float(lr.C.observed),observed_k=int(lr.C.observed_k),
        n_draws=N_DRAWS,p_spread=float(lr.p_spread),
        null_mean_spread=float(lr.null_mean_spread))
    for c in("M","G","S","C"):
        r=getattr(lr,c)
        rec[c]=dict(p=float(r.p),null_mean=float(r.null_mean),
            null_sd=float(r.null_sd),n_exceed=int(r.n_exceed),
            n_degenerate=int(r.n_degenerate),
            null_k_counts={int(k):int(v)
            for k,v in r.null_k_counts.items()})
    return rec


def main():
    t0=time.time()
    st=ov.load_state(OUT,"E12_reduced_substitute",dict(
        n_draws=N_DRAWS,n_init=ov.N_INIT,k_grid=list(ov.K_GRID),seed=SEED,
        specified_design_skipped=True,
        skip_reason=("Fisher-z connectivity features for the 28 fibromyalgia "
        "patients are not on disk; only 28x119 embeddings exist. "
        "Building random-projection / PCA / autoencoder embeddings "
        "from raw features is therefore impossible without "
        "recomputing connectivity, which is out of scope.")))
    for n,fn in STORED.items():
        if n in st["cells"]:
            print("  skip %s"%n,flush=True)
            continue
        p=DATA/fn
        if not p.exists():
            st["cells"][n]=dict(representation=n,skipped=True,
                reason="%s not on disk"%fn)
            print("[E12] %-16s SKIPPED (%s absent)"%(n,fn),flush=True)
            continue
        X=np.load(p).astype(float)
        rec=ladder_record(n,X,kind="stored representation",
            note="loaded from data/outputs/%s"%fn)
        st["cells"][n]=rec
        ov.save_state(OUT,st)
        print("[E12] %-16s PR %.2f raw / %.2f sphere  obs=%.4f k=%d | "
            "M=%.4f G=%.4f S=%.4f C=%.4f  spread=%.4f"
            %(n,rec["pr_unnormalized"],rec["pr_sphere"],rec["observed"],
            rec["observed_k"],rec["M"]["p"],rec["G"]["p"],rec["S"]["p"],
            rec["C"]["p"],rec["p_spread"]),flush=True)
    b=DATA/STORED["gnn_trained"]
    if b.exists():
        X0=np.load(b).astype(float)
        d=X0.shape[1]
        for j,s in enumerate([11,22,33]):
            n="randproj_trained_%d"%j
            if n in st["cells"]:
                continue
            rng=np.random.default_rng(s)
            R=rng.standard_normal((d,d))/np.sqrt(d)
            rec=ladder_record(n,X0@R,kind="derived (random projection)",
                note=("random Gaussian projection of "
                "trained_fm_embeddings at matched output "
                "dimension %d, seed %d"%(d,s)))
            st["cells"][n]=rec
            ov.save_state(OUT,st)
            print("[E12] %-16s PR %.2f raw / %.2f sphere  obs=%.4f k=%d | "
                "M=%.4f G=%.4f S=%.4f C=%.4f  spread=%.4f"
                %(n,rec["pr_unnormalized"],rec["pr_sphere"],
                rec["observed"],rec["observed_k"],rec["M"]["p"],
                rec["G"]["p"],rec["S"]["p"],rec["C"]["p"],
                rec["p_spread"]),flush=True)
    st["wall_s"]=round(time.time()-t0,1)
    ov.save_state(OUT,st)
    print("E12 (reduced) done in %.0fs"%(time.time()-t0))

if __name__=="__main__":
    main()
