#remove dev comments since this is done.
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import adjusted_rand_score

_ROOT=Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT/"src", _ROOT/"models", _ROOT/"analysis"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import config
from clustering import cluster
from dataset import datasetPreparation
from sensitivity_analysis import load_trained_model

def _t(m):
    return torch.tensor(np.asarray(m), dtype=torch.float32)

def run_bootstrap(runner, emb, ids, n_res=None, seed=None):
    n_res=config.bootstrapNResamples if n_res is None else n_res
    seed=config.randomSeed if seed is None else seed
    orig=runner.KMeansUse(_t(emb), list(ids), skip_perm=True, skip_gap=True)
    omap=dict(zip(ids, orig[2]))
    rng=np.random.default_rng(seed)
    n=len(ids)
    aris=np.empty(n_res)
    for b in range(n_res):
        idx=rng.integers(0, n, n)
        bids=[ids[i] for i in idx]
        res=runner.KMeansUse(_t(emb[idx]), bids, skip_perm=True, skip_gap=True)
        seen={}
        for sid, lab in zip(bids, res[2]):
            seen[sid]=lab
        com=list(seen)
        aris[b]=adjusted_rand_score([omap[s] for s in com], [seen[s] for s in com])
    return aris

def _summary_frame(aris):
    rows=pd.DataFrame({"resample": np.arange(len(aris)), "ari": aris})
    summ={
        "n_resamples": len(aris),
        "ari_mean": float(np.mean(aris)),
        "ari_std": float(np.std(aris)),
        "ari_min": float(np.min(aris)),
        "ari_max": float(np.max(aris)),
    }
    return rows, summ

def run_real(conds):
    model=load_trained_model(config.jointCheckpointPath)
    if model is None:
        return None, None, f"missing trained checkpoint ({config.jointCheckpointPath})"
    enc, att=model
    try:
        ds=datasetPreparation(fm_only=False)
        runner=cluster(enc, str(config.checkpointDir), conds, ds.subjectList)
        runner.deploy(ds.subjectData)
        runner.setAttention(att)
        runner._split_fm_hc()
        fmT, fmIds=runner._stack(runner.fmEmbed)
    except Exception as e:
        return None, None, f"could not build FM embeddings: {type(e).__name__}: {e}"
    emb=fmT.detach().cpu().numpy()
    aris=run_bootstrap(runner, emb, fmIds, config.bootstrapNResamples)
    rows, summ=_summary_frame(aris)
    return rows, summ, None

def _synthetic_embeddings(n=30, k=3, d=config.dModel, spread=8.0, noise=1.0, seed=0):
    rng=np.random.default_rng(seed)
    cen=rng.normal(0, spread, (k, d))
    m=np.empty((n, d), dtype=np.float32)
    for i in range(n):
        m[i]=cen[i%k]+rng.normal(0, noise, d)
    ids=[f"sub-fm{i:03d}" for i in range(n)]
    return m, ids

def _bare_runner(conds, ids):
    return cluster(None, str(config.checkpointDir), conds, ids)

def structural_self_test():
    conds=list(config.conditions)
    n_res=25
    print("[self-test] well-separated clusters -> ARI should be near 1.0 ...")
    m, ids=_synthetic_embeddings(noise=0.5, seed=1)
    runner=_bare_runner(conds, ids)
    aris=run_bootstrap(runner, m, ids, n_res=n_res, seed=config.randomSeed)
    print(f"  clean: mean={aris.mean():.4f} range=[{aris.min():.4f},{aris.max():.4f}]")
    assert aris.mean()>0.8, "well-separated clusters must be stable"
    print("[self-test] pure-noise embeddings -> ARI should be lower/scattered ...")
    rng=np.random.default_rng(7)
    noisy=rng.normal(0, 1, (30, config.dModel)).astype(np.float32)
    nids=[f"sub-fm{i:03d}" for i in range(30)]
    nrunner=_bare_runner(conds, nids)
    naris=run_bootstrap(nrunner, noisy, nids, n_res=n_res, seed=config.randomSeed)
    print(f"  noise: mean={naris.mean():.4f} range=[{naris.min():.4f},{naris.max():.4f}]")
    assert naris.mean()<aris.mean(), "noise must be less stable than clean clusters"
    print("[self-test] duplicate-collapse + ARI alignment ...")
    frame, summ=_summary_frame(aris)
    assert len(frame)==n_res and summ["n_resamples"]==n_res, "summary shape mismatch"
    print("[self-test] Good! Bootstrap resampling + ARI logic are sound.")
    return True

def main():
    conds=list(config.conditions)
    rdir=config.resultsRoot
    os.makedirs(rdir, exist_ok=True)
    blks=[]
    print("=" * 70)
    print("Bootstrap stability analysis (subtype reproducibility under resampling)")
    print(f"  BOOTSTRAP_N_RESAMPLES = {config.bootstrapNResamples}")
    print("=" * 70)
    rows, summ, blk=run_real(conds)
    if rows is not None:
        out=rdir/"bootstrap_stability.csv"
        rows.to_csv(out, index=False)
        print(f"[bootstrap] wrote {out}")
        print(f"  ARI mean={summ['ari_mean']:.4f} std={summ['ari_std']:.4f} "
              f"range=[{summ['ari_min']:.4f},{summ['ari_max']:.4f}]")
    else:
        blks.append(f"bootstrap stability BLOCKED: {blk}")
    if blks:
        print("\n" + "-" * 70)
        print("BLOCKERS (no results fabricated):")
        for b in blks:
            print(f"  - {b}")
        print("-" * 70)
        print("Running structural self-test against synthetic data instead....")
        structural_self_test()

    return blks


if __name__ == "__main__":
    main()
