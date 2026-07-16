
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

ROOT=Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "models", ROOT / "analysis", ROOT / "preprocessing"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import config

config.dModel=64
config.heads=4
config.output=32
config.layers=3
config.dropout=0.0

from torch_geometric.data import Batch
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut

import torch.nn as nn
from dataset import datasetPreparation
from gnn_encoder import GNNEncoder
from models.attention_pool import condition_attention_pool


def make_pool():
    p=condition_attention_pool(d_model=config.dModel, num_cons=config.nConditions)
    p.input_norm=nn.Identity()
    return p

SEED=config.randomSeed
NPERM=config.nPermutations
ALPHAS=np.logspace(-3, 4, 20)


def get_attention_embeddings(encoder, pool, subject_data):
    encoder.eval()
    pool.eval()
    embs, groups={}, {}
    with torch.no_grad():
        for subj in subject_data:
            batch=Batch.from_data_list(subj["graphs"])
            per_condition=encoder(batch)          
            pooled, _w, _tau=pool(per_condition)  
            embs[subj["subject_id"]]=pooled
            groups[subj["subject_id"]]=int(subj["group_label"].view(-1)[0].item())
    return embs, groups


def project_ortho(fm, hc):
    fm_centroid=fm.mean(dim=0)
    hc_centroid=hc.mean(dim=0)
    v_sep=fm_centroid - hc_centroid
    v_sep=v_sep / v_sep.norm()
    coeff=fm @ v_sep
    return fm - torch.outer(coeff, v_sep)


def build_severity_index(clinical_df, fm_ids):
    df=clinical_df.set_index("subject_id").loc[list(fm_ids)]
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
    n=len(y)
    preds=np.zeros(n)
    for tr, te, Xtr, Xte in folds:
        model=RidgeCV(alphas=ALPHAS).fit(Xtr, y[tr]) 
        preds[te]=model.predict(Xte)
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
    print(f"[{name}] n={len(y)} dim={X.shape[1]}  r={r:+.4f}  R2={r2:+.4f}  "
          f"perm_p={p:.4f}  (null r mean={null.mean():+.3f}, 95pct={np.percentile(null,95):+.3f})")
    return {"space": name, "n": len(y), "dim": X.shape[1], "r": r, "R2": r2, "perm_p": p}


def complete_subjects():
    base=config.subjectDataFolder
    out=[]
    for d in sorted(Path(base).iterdir()):
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
    manifest_inc=set(subject_filter.get_included_subjects())
    usable=[s for s in complete_subjects() if s in manifest_inc]
    dataset_mod.get_included_subjects=lambda *a, **k: usable
    print(f"Usable subjects (included AND complete imaging): {len(usable)}")

    print("Loading dataset (all subjects, per-condition graphs)...")
    ds=datasetPreparation(avgCond=False, fm_only=False)
    subject_data=ds.subjectData

    clinical=pd.read_csv(config.clinicalCsv)
    clinical["subject_id"]=clinical["subject_id"].astype(str)

    print("Loading trained checkpoint and extracting embeddings...")
    ck=torch.load(config.trainSave, map_location="cpu")
    enc=GNNEncoder()
    enc.load_state_dict(ck["model"])
    pool=make_pool()
    pool.load_state_dict(ck["pool"])
    embs, groups=get_attention_embeddings(enc, pool, subject_data)

    fm_ids=[sid for sid in embs if groups[sid] == 0]
    hc_ids=[sid for sid in embs if groups[sid] == 1]
    fm_ids=sorted(fm_ids)
    print(f"FM subjects: {len(fm_ids)}  HC subjects: {len(hc_ids)}")

    fm_tensor=torch.stack([embs[i] for i in fm_ids])
    hc_tensor=torch.stack([embs[i] for i in hc_ids])
    fm_ortho=project_ortho(fm_tensor, hc_tensor)

    y=build_severity_index(clinical, fm_ids)
    print(f"Severity index (z-mean of VAS/HAMD/HAMA): mean={y.mean():.3f} std={y.std(ddof=1):.3f}")

    X_main=fm_tensor.detach().cpu().numpy()
    X_ortho=fm_ortho.detach().cpu().numpy()

    print("Building random-weight (untrained) encoder control...")
    torch.manual_seed(SEED)
    rand_enc=GNNEncoder()
    rand_pool=make_pool()
    rand_embs, _=get_attention_embeddings(rand_enc, rand_pool, subject_data)
    X_rand=torch.stack([rand_embs[i] for i in fm_ids]).detach().cpu().numpy()

    print("\n=== Ridge LOO-CV regression: severity index ~ embeddings ===")
    results=[
        run_space("main-trained", X_main, y),
        run_space("orthogonal-trained", X_ortho, y),
        run_space("random-encoder-control", X_rand, y),
    ]

    table=pd.DataFrame(results)[["space", "n", "dim", "r", "R2", "perm_p"]]
    print("\n=== SUMMARY ===")
    print(table.to_string(index=False))
    out=config.resultsRoot / "severity_gradient_regression.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"\nwrote {out}")

    r_trained=table.loc[table.space == "main-trained", "r"].iloc[0]
    r_rand=table.loc[table.space == "random-encoder-control", "r"].iloc[0]
    verdict="OUTPERFORMS" if r_trained > r_rand else "DOES NOT outperform"
    print(f"\nVERDICT: trained main encoder (r={r_trained:+.4f}) {verdict} "
          f"random-encoder control (r={r_rand:+.4f}).")


if __name__ == "__main__":
    main()
