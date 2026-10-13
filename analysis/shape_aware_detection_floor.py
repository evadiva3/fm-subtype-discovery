# Corrected 2026-09-22: offsets follow covariance eigenvectors.
# NEW DESIGN: do not substitute archived coordinate-offset results.
"""Corrected eigenvector-offset driver. Historical default: 200 datasets, 199 draws.
Requires the original FINAL spectrum JSON; a fresh spectrum defines a new study.
Use --spectrum PATH to supply that file. Per-dataset records and diagnostics are saved.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from scipy.stats import binomtest

_R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_R / "packaging"))
sys.path.insert(0, str(_R / "experiments" / "ds002748_mdd" / "propagated_null"))
import angular_gof as gof
import angular_null as an
import nullcal_validation as nv
import ov_common as ov

OUT = _R / "results" / "revision_20260923" / "shape_aware_detection_floor_eigenvector_corrected.json"
SHAPE = _R / "results" / "overnight" / "e2_spectrum_shape.json"
N = 28
D = ov.D
NDS = 200
DRAWS = 199
AUXILIARY = 200
QS = (5, 10)
DELTA = 4.0
ROOT = 20260922
EXPERIMENT = 41
N_WORKERS = max(1, min(4, os.cpu_count() or 1))

DELTAS = (0.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)
CELLS = [("delta%g" % x, "two_block_pr4.5", x) for x in DELTAS]

_LAM = {}


def spectrum(key):
    cell = json.loads(SHAPE.read_text())["cells"][key]
    if cell["family"] == "exp":
        return ov.spec_exp(D, float(cell["param_value"])), cell
    m, eps = str(cell["param_value"]).split("/")
    return ov.spec_two_block(D, int(m), float(eps)), cell


def _init(lams, draws=None):
    global DRAWS
    if draws is not None:
        DRAWS = draws
    _LAM.update(lams)


def one(task):
    name, q, i = task
    try:
        lam = _LAM[name]
        seed = nv.stream_seed(ROOT, EXPERIMENT, CELL_ID[name], q, i, 0)
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((N, D))
        rotation = np.linalg.qr(rng.standard_normal((D, D)))[0]
        x = (z * np.sqrt(lam)) @ rotation.T
        planted = CELL_PLANTED[name]
        if planted:
            x += (planted * rng.choice([-1.0, 1.0], size=N) * np.sqrt(lam[0]))[:, None] * rotation[:, 0][None, :]
        xn = nv.normalize(x)
        proj_seed = nv.stream_seed(ROOT, EXPERIMENT, 99, q, 0, 2)
        proj = an.fixed_projection(D, q, proj_seed)
        xp = nv.normalize(xn @ proj)
        null_seed = nv.stream_seed(ROOT, EXPERIMENT, CELL_ID[name], q, i, 100)
        boot_seed = nv.stream_seed(ROOT, EXPERIMENT, CELL_ID[name], q, i, 200)
        res = gof.evaluate(xp, draws=DRAWS, seed=int(null_seed),
                           bootstrap_seed=int(boot_seed), auxiliary=AUXILIARY)
        return dict(cell=name, q=q, index=i, planted=planted,
                    source_data_seed=str(seed), null_seed=str(null_seed),
                    projection_seed=str(proj_seed),
                    source_data_sha256=hashlib.sha256(xn.astype("<f8").tobytes()).hexdigest(),
                    p=res["p"], diagnostics=res, bootstrap_seed=str(boot_seed))
    except Exception:
        return dict(cell=name, q=q, index=i, error=traceback.format_exc())


CELL_ID = {name: k for k, (name, _, _) in enumerate(CELLS)}
CELL_PLANTED = {name: delta for name, _, delta in CELLS}


def interval(k, n):
    ci = binomtest(k, n).proportion_ci(method="exact")
    return [float(ci.low), float(ci.high)]


def main():
    global SHAPE, NDS, DRAWS, N_WORKERS, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spectrum', type=Path, default=SHAPE)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--datasets', type=int, default=NDS)
    parser.add_argument('--draws', type=int, default=DRAWS)
    parser.add_argument('--workers', type=int, default=N_WORKERS)
    args = parser.parse_args()
    OUT = args.output
    SHAPE, NDS, DRAWS, N_WORKERS = args.spectrum, args.datasets, args.draws, args.workers
    if not SHAPE.is_file():
        parser.error('Required FINAL spectrum JSON is missing: ' + str(SHAPE) + '. Supply --spectrum PATH; do not relabel fresh tuning as a historical rerun.')
    if NDS < 1 or DRAWS < 2 or N_WORKERS < 1:
        parser.error('Require datasets >= 1, draws >= 2 and workers >= 1')
    if OUT.exists() or OUT.with_suffix('.records.jsonl').exists():
        parser.error('Output already exists; choose a fresh --output path to preserve previous records.')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    import scipy
    import sklearn
    lams = {}
    meta = {}
    for name, key, planted in CELLS:
        lam, cell = spectrum(key)
        lams[name] = lam
        meta[name] = dict(spectrum_key=key, family=cell["family"],
                          param_value=cell["param_value"], planted_delta=planted,
                          archived_pr_mean=cell["pr_mean"])
    tasks = [(name, q, i) for name, _, _ in CELLS for q in QS for i in range(NDS)]
    print("%d analyses: %d cells x %d projections x %d datasets, n = %d, d = %d"
          % (len(tasks), len(CELLS), len(QS), NDS, N, D), flush=True)
    t0 = time.time()
    with Pool(N_WORKERS, initializer=_init, initargs=(lams, DRAWS)) as pool:
        rows = []
        with OUT.with_suffix('.records.jsonl').open('w', encoding='utf-8') as journal:
            for row in pool.imap_unordered(one, tasks, chunksize=1):
                rows.append(row)
                journal.write(json.dumps(row) + '\n')
                journal.flush()
                if len(rows) % 100 == 0:
                    print(f'{len(rows)}/{len(tasks)} complete, {time.time()-t0:.0f}s', flush=True)
    errs = [r for r in rows if "error" in r]
    if errs:
        OUT.with_suffix('.failed.json').write_text(json.dumps(dict(errors=errs, per_dataset_results=rows), indent=1))
        raise RuntimeError(errs[0]["error"])
    out = dict(
        experiment="corrected leading-eigenvector separation sweep",
        written_utc=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        n=N, d=D, n_datasets=NDS, draws=DRAWS, auxiliary=AUXILIARY, q=list(QS),
        planted_deltas=list(DELTAS), planted_axis=0, alpha=0.05, arms=list(gof.ARMS), cells=meta,
        seed_rule="nullcal_validation.stream_seed(%d, %d, cell, q, dataset, purpose)"
                  % (ROOT, EXPERIMENT),
        versions={"python": platform.python_version(), "numpy": np.__version__,
                  "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
                  "platform": platform.platform(), "n_workers": N_WORKERS},
        results={})
    hashes = set()
    for name, _, _ in CELLS:
        for q in QS:
            sub = [r for r in rows if r["cell"] == name and r["q"] == q]
            hashes |= {r["source_data_sha256"] for r in sub}
            arms = {}
            for a in gof.ARMS:
                c = sum(r["p"][a] <= 0.05 for r in sub)
                arms[a] = dict(count=c, rate=c / len(sub), ci=interval(c, len(sub)))
            out["results"]["%s_q%02d" % (name, q)] = dict(
                cell=name, q=q, n_datasets=len(sub), kind="alternative" if CELL_PLANTED[name] else "null", delta=CELL_PLANTED[name],
                arms=arms)
            print("[%s q=%d] " % (name, q) + "  ".join(
                "%s %.1f%% [%.1f, %.1f]" % (a, 100 * arms[a]["rate"], 100 * arms[a]["ci"][0],
                                            100 * arms[a]["ci"][1]) for a in gof.ARMS), flush=True)
    out["per_dataset_results"] = sorted(rows, key=lambda row: (row['cell'], row['q'], row['index']))
    out["module_sha256"] = {m.__name__: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                            for m in (gof, an, nv, ov)}
    out["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out["spectrum_file_sha256"] = hashlib.sha256(SHAPE.read_bytes()).hexdigest()
    out["distinct_source_datasets"] = len(hashes)
    out["total_wall_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, indent=1))
    print("distinct source datasets %d, wall %.0f min" % (len(hashes), out["total_wall_s"] / 60))
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
