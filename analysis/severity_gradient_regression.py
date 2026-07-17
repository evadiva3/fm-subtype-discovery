import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

ROOT=Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT/"src", ROOT/"models", ROOT/"analysis", ROOT/"preprocessing"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import config
from torch_geometric.data import Batch
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut

from dataset import datasetPreparation
from gnn_encoder import GNNEncoder
from models.attention_pool import condition_attention_pool

def make_pool():
    return condition_attention_pool(d_model=config.dModel, num_cons=config.nConditions)

SEED=config.randomSeed
NPERM=config.nPermutations
ALPHAS=np.logspace(-3, 4, 20)

def get_attention_embeddings(enc, pool, subs):
    enc.eval()
    pool.eval()
    embs, groups={}, {}
    with torch.no_grad():
        for s in subs:
            batch=Batch.from_data_list(s["graphs"])
            pc=enc(batch)
            pl, _w, _tau=pool(pc)
            embs[s["subject_id"]]=pl
            groups[s["subject_id"]]=int(s["group_label"].view(-1)[0].item())
    return embs, groups
def project_ortho(fm, hc):
    fmc=fm.mean(dim=0)
    hcc=hc.mean(dim=0)
    v=fmc - hcc
    v=v / v.norm()
    coeff=fm @ v
    return fm - torch.outer(coeff, v)
def build_severity_index(clin, fm_ids):
    df=clin.set_index("subject_id").loc[list(fm_ids)]
    cols=["vas_pain", "hamd_total", "hama_total"]
    z=(df[cols] - df[cols].mean()) / df[cols].std(ddof=1)
    return z.mean(axis=1).to_numpy()
def make_folds(X):
    folds=[]
    for tr, te in LeaveOneOut().split(X):
        sc=StandardScaler().fit(X[tr])
        folds.append((tr, te, sc.transform(X[tr]), sc.transform(X[te])))
    return folds
def loo_predictions(folds, y):
    preds=np.zeros(len(y))
    for tr, te, Xtr, Xte in folds:
        m=RidgeCV(alphas=ALPHAS).fit(Xtr, y[tr])
        preds[te]=m.predict(Xte)
    return preds
def score(preds, y):
    r=np.corrcoef(preds, y)[0, 1]
    ss_res=float(((y - preds) ** 2).sum())
    ss_tot=float(((y - y.mean()) ** 2).sum())
    r2=1.0 - ss_res / ss_tot
    return r, r2
def permutation_p(folds, y, r_obs, nperm=NPERM, seed=SEED):
    rng=np.random.default_rng(seed)
    null=np.empty(nperm)
    for i in range(nperm):
        yp=rng.permutation(y)
        preds=loo_predictions(folds, yp)
        null[i]=np.corrcoef(preds, yp)[0, 1]
    p=float(np.mean(null > r_obs))
    return p, null
def run_space(name, X, y):
    folds=make_folds(X)
    preds=loo_predictions(folds, y)
    r, r2=score(preds, y)
    p, null=permutation_p(folds, y, r)
    print(f"{name}: n={len(y)} dim={X.shape[1]} r={r:+.4f} R2={r2:+.4f} perm_p={p:.4f} "
          f"(null mean={null.mean():+.3f} 95pct={np.percentile(null,95):+.3f})")
    return {"space": name, "n": len(y), "dim": X.shape[1], "r": r, "R2": r2, "perm_p": p}
def complete_subjects():
    out=[]
    for d in sorted(Path(config.subjectDataFolder).iterdir()):
        if not (d.is_dir() and d.name.startswith("sub-")):
            continue
        ok=all(
            (d / f"{d.name}_FCMatrixCondition{c.replace(' ', '')}.npy").exists()
            and (d / f"{d.name}_ROITimeSeries{c.replace(' ', '')}.npy").exists()
            for c in config.conditions
        )
        if ok:
            out.append(d.name)
    return out
def main():
    import subject_filter
    import dataset as dataset_mod
    inc=set(subject_filter.get_included_subjects())
    usable=[s for s in complete_subjects() if s in inc]
    dataset_mod.get_included_subjects=lambda *a, **k: usable
    print(f"usable subjects (included and complete): {len(usable)}")

    print("building dataset")
    ds=datasetPreparation(avgCond=False, fm_only=False)
    subs=ds.subjectData

    clin=pd.read_csv(config.clinicalCsv)
    clin["subject_id"]=clin["subject_id"].astype(str)

    print("loading checkpoint, extracting embeddings")
    ck=torch.load(config.trainSave, map_location="cpu")
    enc=GNNEncoder()
    enc.load_state_dict(ck["model"])
    pool=make_pool()
    pool.load_state_dict(ck["pool"])
    embs, groups=get_attention_embeddings(enc, pool, subs)

    fm_ids=sorted(sid for sid in embs if groups[sid]==0)
    hc_ids=[sid for sid in embs if groups[sid]==1]
    print(f"FM {len(fm_ids)}, HC {len(hc_ids)}")

    fm=torch.stack([embs[i] for i in fm_ids])
    hc=torch.stack([embs[i] for i in hc_ids])
    fmo=project_ortho(fm, hc)

    y=build_severity_index(clin, fm_ids)
    print(f"severity index mean={y.mean():.3f} std={y.std(ddof=1):.3f}")

    xm=fm.detach().cpu().numpy()
    xo=fmo.detach().cpu().numpy()

    def rand_emb(seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        e=GNNEncoder()
        pp=make_pool()
        em, _=get_attention_embeddings(e, pp, subs)
        return torch.stack([em[i] for i in fm_ids]).detach().cpu().numpy()
    print("ridge loo-cv: severity ~ embeddings")
    xr=rand_emb(SEED)
    results=[
        run_space("main-trained", xm, y),
        run_space("orthogonal-trained", xo, y),
        run_space(f"random-encoder-control (seed {SEED})", xr, y),
    ]
    print("random-encoder control seeds 0-9")
    rrows=[]
    for seed in range(10):
        res=run_space(f"random-seed-{seed}", rand_emb(seed), y)
        rrows.append({"seed": seed, "r": res["r"], "R2": res["R2"], "perm_p": res["perm_p"]})
    rdf=pd.DataFrame(rrows)
    print(rdf.to_string(index=False))
    r_mean, r_min, r_max=rdf["r"].mean(), rdf["r"].min(), rdf["r"].max()
    print(f"random seeds 0-9: r mean={r_mean:+.4f} min={r_min:+.4f} max={r_max:+.4f} | "
          f"R2 mean={rdf['R2'].mean():+.4f} min={rdf['R2'].min():+.4f} max={rdf['R2'].max():+.4f} | "
          f"perm_p mean={rdf['perm_p'].mean():.4f}")

    table=pd.DataFrame(results)[["space", "n", "dim", "r", "R2", "perm_p"]]
    print(table.to_string(index=False))
    print(f"random seeds 0-9: r mean={r_mean:+.4f} range=[{r_min:+.4f}, {r_max:+.4f}]")

    out=config.resultsRoot / "severity_gradient_regression.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    rdf.to_csv(config.resultsRoot / "severity_gradient_random_seeds.csv", index=False)
    print(f"wrote {out}")
    print(f"wrote {config.resultsRoot / 'severity_gradient_random_seeds.csv'}")
    rt=table.loc[table.space=="main-trained", "r"].iloc[0]
    rr=table.loc[table.space==f"random-encoder-control (seed {SEED})", "r"].iloc[0]
    verdict="outperforms" if rt > rr else "does not outperform"
    print(f"verdict: trained (r={rt:+.4f}) {verdict} random control "
          f"(seed {SEED} r={rr:+.4f}, seeds 0-9 mean r={r_mean:+.4f})")
if __name__ == "__main__":
    main()
