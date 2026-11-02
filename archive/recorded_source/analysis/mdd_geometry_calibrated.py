"""S35: S24 and S29b repeated at the depression embedding geometry; see docs/FOLLOWUP_PROTOCOL_S35_S36_20261008.md."""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"
import argparse, hashlib, json, platform, sys, time, traceback
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import scipy, sklearn
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "experiments/ds002748_mdd/propagated_null"))
sys.path.insert(0, str(R / "analysis"))
import nullcal as nc
import ov_common as ov
from summarize_detection import cutoff, reject, ci, summarize

EXP_A, BOOT_A, EXP_B, BOOT_B = 90, 91, 92, 93
GA = ("calibration", "null") + tuple(f"{f}{d}" for f in ("two", "three") for d in ("0.5", "1.0", "1.5", "2.0"))
GB = ("skew", "heavy")
PROTOCOL = R / "docs/FOLLOWUP_PROTOCOL_S35_S36_20261008.md"
EMB = R / "results/mdd/canonical_single/Embeddings.npy"
_X = nc.l2_normalize(np.load(EMB).astype(np.float64))
MEAN, FACTOR = nc._gaussian_factor(_X)
N = len(_X)
_lam, _vec = np.linalg.eigh(FACTOR.T @ FACTOR)
BASIS = (np.sqrt(np.clip(_lam[::-1][:2], 0, None)) * _vec[:, ::-1][:, :2]).T  # rows m1, m2


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sd(root, exp, *a):
    return int.from_bytes(hashlib.sha256("|".join(map(str, (root, exp, *a))).encode()).digest()[:16], "big")


def means(name, perm):
    if name in ("calibration", "null"):
        return None, None
    fam = name.rstrip("0123456789.")
    de = float(name[len(fam):])
    if fam == "two":
        c = np.array([[1., 0.], [-1., 0.]]); sizes = [N - N // 2, N // 2]
    else:
        c = np.array([[np.cos(t), np.sin(t)] for t in (0., 2 * np.pi / 3, 4 * np.pi / 3)]); sizes = [N - 2 * (N // 3), N // 3, N // 3]
    M = de * c @ BASIS
    lab = np.repeat(np.arange(len(sizes)), sizes)[np.argsort(np.argsort(perm))]
    w = np.bincount(lab, minlength=len(M)) / N
    return M - w @ M, lab


def noise(law, rng, shape):
    if law == "skew":
        return rng.exponential(size=shape) - 1.
    if law == "heavy":
        return rng.standard_t(4, size=shape) / np.sqrt(2.)
    raise ValueError(law)


def test(x, rs, i):
    s, k, _ = nc.best_silhouette(x, k_grid=ov.K_GRID, min_size=nc.min_cluster_guard(N), n_init=5, random_state=i)
    out = dict(observed=s, selected_k=k, guard_failed=k is None)
    if k is None:
        out.update(p={"M": 1., "C": 1.}, reference_guard_failures=0)
    else:
        kw = dict(k_grid=ov.K_GRID, n_draws=499, min_size=nc.min_cluster_guard(N), n_init=5, seed=rs, observed=(s, k))
        res = {a: nc.null_test(x, reproject=z, match_selection=z, **kw) for a, z in (("M", False), ("C", True))}
        out.update(p={a: float(v.p) for a, v in res.items()}, reference_guard_failures=int(res["C"].n_degenerate))
    return out


def one_a(t):
    root, g, i = t
    ds = sd(root, EXP_A, 0, 0 if g == 0 else 1, i)
    r = dict(spectrum=0, q=0, group=GA[g], index=i, data_seed=str(ds))
    try:
        rng = np.random.default_rng(ds)
        x = nc._draw(MEAN, FACTOR, N, rng)
        perm = rng.permutation(N)
        M, lab = means(GA[g], perm)
        if M is not None:
            x = x + M[lab]
        x = nc.l2_normalize(x)
        rs = sd(root, EXP_A, 2, g, i)
        u = float(np.random.default_rng(sd(root, EXP_A, 4, g, i)).uniform())
        r.update(labels=None if lab is None else lab.tolist(), source_sha256=hashlib.sha256(x.astype("<f8").tobytes()).hexdigest(),
                 reference_seed=str(rs), boundary_uniform=u,
                 standardized_between=0. if M is None else float(np.sum((np.bincount(lab) / N)[:, None] * (M @ np.linalg.pinv(BASIS)) ** 2)))
        r.update(test(x, rs, i))
    except Exception:
        r["error"] = traceback.format_exc()
    return r


def one_b(t):
    root, g, i = t
    name = GB[g]
    ds = sd(root, EXP_B, 0, g, i)
    r = dict(group=name, index=i, data_seed=str(ds))
    try:
        rng = np.random.default_rng(ds)
        x = nc.l2_normalize(MEAN + noise(name, rng, (N, FACTOR.shape[0])) @ FACTOR)
        rs = sd(root, EXP_B, 2, g, i)
        u = float(np.random.default_rng(sd(root, EXP_B, 4, g, i)).uniform())
        r.update(source_sha256=hashlib.sha256(x.astype("<f8").tobytes()).hexdigest(), reference_seed=str(rs), boundary_uniform=u)
        r.update(test(x, rs, i))
    except Exception:
        r["error"] = traceback.format_exc()
    return r


def base_config(root):
    return dict(root_seed=root, n=N, d=int(_X.shape[1]), arms=["M", "C"], draws=499, k_grid=list(ov.K_GRID), restarts=5,
                guard=nc.min_cluster_guard(N), bootstrap_resamples=2000, input_sha256=sha(EMB), protocol_sha256=sha(PROTOCOL),
                source_sha256={str(f.resolve().relative_to(R)).replace("\\", "/"): sha(f) for f in (Path(__file__), Path(nc.__file__), Path(ov.__file__), R / "analysis/summarize_detection.py")},
                versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__))


def summarize_b(d, cal_dir):
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    cal = [json.loads(l) for l in (cal_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    cal = sorted([r for r in cal if r["group"] == "calibration"], key=lambda r: r["index"])
    assert len(cal) == 1000 and not any("error" in r for r in cal)
    rows = [json.loads(l) for l in (d / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {(r["group"], r["index"]) for r in rows} == {(g, i) for g, n in cfg["groups"].items() for i in range(n)} and len(rows) == sum(cfg["groups"].values())
    assert not any("error" in r for r in rows)
    out = dict(config=cfg, records_sha256=sha(d / "records.jsonl"), calibration_records_sha256=sha(cal_dir / "records.jsonl"), cells={})
    nb = cfg["bootstrap_resamples"]
    for gi, g in enumerate(cfg["groups"]):
        rs = sorted([r for r in rows if r["group"] == g], key=lambda r: r["index"])
        cell = dict(n=len(rs), observed_guard_failures=sum(r["guard_failed"] for r in rs), reference_guard_failures=sum(r["reference_guard_failures"] for r in rs), arms={})
        rng = np.random.default_rng(np.random.SeedSequence([20261008, cfg["bootstrap_address"], gi]))
        ic = rng.integers(0, len(cal), size=(nb, len(cal)))
        ie = rng.integers(0, len(rs), size=(nb, len(rs)))
        u = np.array([r["boundary_uniform"] for r in rs])
        for a in cfg["arms"]:
            cp = np.array([r["p"][a] for r in cal]); p = np.array([r["p"][a] for r in rs])
            t, rho = cutoff(cp)
            v = reject(p, u, t, rho)
            bt = np.array([reject(p[ie[b]], u[ie[b]], *cutoff(cp[ic[b]])).mean() for b in range(nb)])
            c = int(v.sum()); cn = int((p <= .05).sum())
            cell["arms"][a] = dict(cutoff=t, boundary_probability=rho, calibrated_events=c, calibrated_rate=c / len(rs), conditional_wilson95=ci(c, len(rs)),
                                   two_stage_bootstrap95=np.quantile(bt, [.025, .975]).tolist(), nominal_events=cn, nominal_rate=cn / len(rs), nominal_wilson95=ci(cn, len(rs)))
        out["cells"][g] = cell
    return out


def run(out, cfg, tasks, fn, key, workers):
    out.mkdir(parents=True, exist_ok=True)
    cp = out / "config.json"
    if cp.exists():
        assert json.loads(cp.read_text(encoding="utf-8")) == cfg, "Configuration mismatch"
    else:
        cp.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    jp = out / "records.jsonl"
    rows = [json.loads(l) for l in jp.read_text(encoding="utf-8").splitlines()] if jp.exists() else []
    done = {key(r) for r in rows}
    ts = [t for t, k in tasks if k not in done]
    t0 = time.time()
    with Pool(workers) as pool, jp.open("a", encoding="utf-8") as j:
        for r in pool.imap_unordered(fn, ts, chunksize=1):
            j.write(json.dumps(r) + "\n"); j.flush(); rows.append(r)
            if len(rows) % 100 == 0:
                print(f"{len(rows)}/{len(tasks)}; {time.time() - t0:.0f}s", flush=True)
    end = dict(n=len(rows), errors=sum("error" in r for r in rows), records_sha256=sha(jp), execution_wall_s=time.time() - t0)
    (out / "execution.json").write_text(json.dumps(end, indent=2), encoding="utf-8")
    print(json.dumps(end), flush=True)
    if end["errors"]:
        raise RuntimeError("Errors retained; do not summarize as completed")


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--part", required=True, choices=("a", "b"))
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--calibration", type=Path, help="S35a output directory (part b)")
    a.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a.add_argument("--pilot", action="store_true")
    a.add_argument("--summarize", action="store_true")
    a = a.parse_args()
    root = 2026100801 if a.pilot else 20261008
    if a.part == "a":
        if a.summarize:
            (a.output / "summary.json").write_text(json.dumps(summarize(a.output), indent=2), encoding="utf-8"); return
        ns = [3] * len(GA) if a.pilot else [1000, 1000] + [250] * (len(GA) - 2)
        cfg = dict(study="S35a", experiment=EXP_A, bootstrap_address=BOOT_A, qs=[0], spectra=["fitted_mdd_embedding"],
                   groups=dict(zip(GA, ns)), diagnostics=["standardized_between"], **base_config(root))
        tasks = [((root, g, i), (GA[g], i)) for g in range(len(GA)) for i in range(ns[g])]
        run(a.output, cfg, tasks, one_a, lambda r: (r["group"], r["index"]), a.workers)
    else:
        assert a.calibration is not None, "--calibration is required for part b"
        if a.summarize:
            (a.output / "summary.json").write_text(json.dumps(summarize_b(a.output, a.calibration), indent=2), encoding="utf-8"); return
        ns = [3] * len(GB) if a.pilot else [1000, 1000]
        cfg = dict(study="S35b", experiment=EXP_B, bootstrap_address=BOOT_B, groups=dict(zip(GB, ns)),
                   calibration=str(a.calibration.resolve().relative_to(R)).replace("\\", "/") if a.calibration.resolve().is_relative_to(R) else str(a.calibration),
                   **base_config(root))
        tasks = [((root, g, i), (GB[g], i)) for g in range(len(GB)) for i in range(ns[g])]
        run(a.output, cfg, tasks, one_b, lambda r: (r["group"], r["index"]), a.workers)


if __name__ == "__main__":
    main()
