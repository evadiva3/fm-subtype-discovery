"""Reproduce the corrected, explicitly versioned validation study.

Run a smoke check with --datasets 4 --draws 19 --workers 2 --output <scratch>.
Production defaults use 400 independent datasets per cell and 199 null draws.
Outputs are checkpointed per dataset and never mixed across configurations.
"""
from __future__ import annotations
import os
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
import hashlib
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import scipy
from scipy.stats import binomtest
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packaging"))
import nullcal_validation as nv

CELLS = [
    dict(name="exp_n28", n=28, d=119, family="exp", parameter=2.1984794925044238),
    dict(name="spike_n28", n=28, d=119, family="spike", parameter=198.0956778550339),
    dict(name="five_spikes_n28", n=28, d=119, family="five_spikes", parameter=3162.2776601683795),
    dict(name="isotropic_n28", n=28, d=119, family="isotropic", parameter=1.),
    dict(name="exp_n180", n=180, d=119, family="exp", parameter=4.016066862466169),
    dict(name="spike_n180", n=180, d=119, family="spike", parameter=1/.0392419),
    dict(name="exp_n200", n=200, d=119, family="exp", parameter=1.7276310965477195),
]


def spectrum(cell):
    d, p, family = cell["d"], cell["parameter"], cell["family"]
    if family == "exp":
        return np.exp(-np.arange(d)/p)
    lam = np.ones(d)
    if family == "spike":
        lam[0] = p
    elif family == "five_spikes":
        lam[:5] = p
    elif family != "isotropic":
        raise ValueError(family)
    return lam / lam.max()


def prepare(cell, root_seed, cell_id):
    cell = dict(cell, id=cell_id)
    lam = spectrum(cell)
    rotation_seed = nv.stream_seed(root_seed, 7, cell_id, 0, 0, 9)
    q, _ = np.linalg.qr(np.random.default_rng(rotation_seed).standard_normal((cell["d"], cell["d"])))
    cell["eigenvalues"] = lam.tolist()
    cell["population_factor"] = (np.sqrt(lam)[:,None] * q.T).tolist()
    cell["moment_factor"] = (np.sqrt(nv.sphere_moments(lam))[:,None] * q.T).tolist()
    return cell


def one(task):
    cell, index, config = task
    replicate, local_index = divmod(index, config["block_size"])
    data_seed = nv.stream_seed(config["root_seed"], 7, cell["id"], replicate, local_index, 0)
    null_seed = nv.stream_seed(config["root_seed"], 7, cell["id"], replicate, local_index, 1)
    factor = np.asarray(cell["population_factor"])
    x = np.random.default_rng(data_seed).standard_normal((cell["n"], cell["d"])) @ factor
    record = nv.evaluate_dataset(x, factor, np.asarray(cell["moment_factor"]),
                                 n_draws=config["draws"], seed=null_seed,
                                 n_init=config["n_init"], cluster_seed=0)
    return dict(record, cell=cell["name"], index=index, replicate=replicate,
                local_index=local_index, data_seed=str(data_seed), draws=config["draws"])


def interval(c, n):
    ci = binomtest(c, n).proportion_ci(method="exact")
    return [float(ci.low), float(ci.high)]


def summarize(records):
    n = len(records)
    result = {"n":n, "observed_guard_failures":sum(r["observed_guard_failed"] for r in records)}
    seeds = [r["data_seed"] for r in records]
    hashes = [r["data_sha256"] for r in records]
    if len(set(seeds)) != n or len(set(hashes)) != n:
        raise AssertionError("duplicate dataset stream or dataset")
    result["unique_seeds"] = len(set(seeds))
    result["pr_mean"] = float(np.mean([r["pr"] for r in records]))
    result["pc1_mean"] = float(np.mean([r["pc1"] for r in records]))
    result["arms"] = {}
    for arm in records[0]["p"]:
        p = np.array([r["p"][arm] for r in records])
        c = int((p <= .05).sum())
        result["arms"][arm] = dict(count=c, rate=c/n, ci=interval(c,n), mean_p=float(p.mean()),
                                    blocks={str(b):int(sum(r["p"][arm] <= .05 for r in records if r["replicate"]==b)) for b in sorted({r["replicate"] for r in records})})
    result["comparisons"] = {}
    for a,b in (("M","C"),("Mg","S"),("Gg","C"),("S","C"),("M","KM"),("C","KC")):
        left = sum(r["p"][a] <= .05 < r["p"][b] for r in records)
        right = sum(r["p"][b] <= .05 < r["p"][a] for r in records)
        result["comparisons"][a+"_vs_"+b] = dict(a_only=left,b_only=right,
            difference_b_minus_a=(right-left)/n,
            mcnemar_exact_p=float(binomtest(left,left+right,.5).pvalue) if left+right else 1.)
    shifts = [r["shift_sd"] for r in records if r.get("shift_sd") is not None]
    result["mean_shift_sd"] = float(np.mean(shifts)) if shifts else None
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets",type=int,default=400)
    parser.add_argument("--draws",type=int,default=199)
    parser.add_argument("--workers",type=int,default=min(36,os.cpu_count() or 1))
    parser.add_argument("--root-seed",type=int,default=20260914)
    parser.add_argument("--n-init",type=int,default=5)
    parser.add_argument("--cells",nargs="*",default=[c["name"] for c in CELLS])
    parser.add_argument("--output",type=Path,default=ROOT/"results"/"validation_v7")
    args = parser.parse_args()
    if args.datasets < 1 or args.draws < 2 or args.workers < 1 or args.n_init < 1:
        parser.error("positive datasets, workers, n-init and at least two draws required")
    if not set(args.cells) <= {c["name"] for c in CELLS}:
        parser.error("unknown cell")
    out=args.output; out.mkdir(parents=True,exist_ok=True)
    config=dict(version=nv.VERSION,datasets=args.datasets,draws=args.draws,root_seed=args.root_seed,
                block_size=100,n_init=args.n_init,cluster_seed=0,threshold="p <= 0.05",cells=args.cells,
                python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,
                source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(nv.__file__))})
    cfgpath=out/"config.json"
    if cfgpath.exists() and json.loads(cfgpath.read_text()) != config:
        raise RuntimeError("output directory belongs to a different configuration; use a fresh output")
    cfgpath.write_text(json.dumps(config,indent=2))
    specs=[prepare(c,args.root_seed,i) for i,c in enumerate(CELLS) if c["name"] in args.cells]
    (out/"populations.json").write_text(json.dumps(specs,indent=1))
    started=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for cell in specs:
            folder=out/cell["name"]; folder.mkdir(exist_ok=True)
            pending=[i for i in range(args.datasets) if not (folder/f"{i:05d}.json").exists()]
            jobs={pool.submit(one,(cell,i,config)):i for i in pending}
            done=args.datasets-len(pending)
            for future in as_completed(jobs):
                record=future.result()
                (folder/f"{record['index']:05d}.json").write_text(json.dumps(record,sort_keys=True))
                done+=1
                if done%max(1,min(25,args.datasets))==0:
                    print(f"{cell['name']}: {done}/{args.datasets}, elapsed {time.monotonic()-started:.0f}s",flush=True)
            records=[json.loads((folder/f"{i:05d}.json").read_text()) for i in range(args.datasets)]
            summary=summarize(records)
            (folder/"summary.json").write_text(json.dumps(summary,indent=2))
            print(cell["name"],{k:round(v["rate"],4) for k,v in summary["arms"].items()},flush=True)
    final={c["name"]:json.loads((out/c["name"]/"summary.json").read_text()) for c in specs}
    all_seeds=[json.loads(p.read_text())["data_seed"] for c in specs for p in (out/c["name"]).glob("[0-9]*.json")]
    assert len(set(all_seeds))==len(all_seeds)
    (out/"summary.json").write_text(json.dumps(dict(config=config,cells=final,total_unique_datasets=len(all_seeds)),indent=2))


if __name__=="__main__":
    main()
