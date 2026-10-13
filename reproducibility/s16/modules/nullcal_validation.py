"""Auditable simulation controls; these are not a certified test for real embeddings."""
from __future__ import annotations

import hashlib
import numpy as np
from scipy.integrate import simpson
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, silhouette_score

VERSION = "2026-09-14.1"
K_GRID = (2, 3, 4, 5, 6)
INVALID = -1.0


def stream_seed(root, experiment, cell, replicate, dataset, purpose=0):
    """Hierarchical streams, independent of worker order and draw resolution."""
    address = [int(v) for v in (root, experiment, cell, replicate, dataset, purpose)]
    if min(address) < 0:
        raise ValueError("seed addresses must be nonnegative")
    words = np.random.SeedSequence(address).generate_state(4)
    return int.from_bytes(words.astype('<u4').tobytes(), 'little')


def normalize(x):
    x = np.asarray(x, dtype=float)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.isfinite(x).all() or np.any(norms == 0):
        raise ValueError("finite nonzero rows required")
    return x / norms


def factor_covariance(cov):
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or not np.isfinite(cov).all():
        raise ValueError("finite square covariance required")
    if not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-12):
        raise ValueError("covariance must be symmetric")
    vals, vecs = np.linalg.eigh((cov + cov.T) / 2)
    tol = 1e-10 * max(float(np.max(np.abs(vals))), np.finfo(float).tiny)
    if vals.min() < -tol:
        raise ValueError("covariance must be positive semidefinite")
    return np.sqrt(np.maximum(vals, 0))[:, None] * vecs.T


def scores(x, k_grid=K_GRID, min_size=None, n_init=5, cluster_seed=0):
    """One partition per k; fixed and searched statistics use the same scores."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if min_size is None:
        min_size = max(4, round(.15 * n))
    distances = pairwise_distances(x)
    np.fill_diagonal(distances, 0)
    raw, guarded = {}, {}
    for k in k_grid:
        if not 2 <= k < n:
            continue
        labels = KMeans(n_clusters=k, n_init=n_init, random_state=cluster_seed).fit_predict(x)
        counts = np.bincount(labels, minlength=k)
        s = INVALID if len(np.unique(labels)) < 2 else float(silhouette_score(distances, labels, metric="precomputed"))
        raw[int(k)] = s
        guarded[int(k)] = s if counts.min() >= min_size else INVALID
    valid = [k for k, s in guarded.items() if s > INVALID]
    selected = max(valid, key=lambda k: guarded[k]) if valid else None
    return raw, guarded, selected


def sphere_moments(lam):
    """Covariance eigenvalues of normalized N(0, diag(lam)), not its ACG shape."""
    lam = np.asarray(lam, dtype=float)
    if lam.ndim != 1 or np.any(lam <= 0) or not np.isfinite(lam).all():
        raise ValueError("positive finite eigenvalues required")
    lam = lam / lam.max()
    u = np.linspace(-45, max(90., -np.log(lam.min()) + 45), 16001)
    t = np.exp(u)
    tl = t[:, None] * lam
    g = np.exp(-.5 * np.log1p(2 * tl).sum(axis=1))
    c = simpson(lam[None, :] / (1 + 2 * tl) * (g * t)[:, None], x=u, axis=0)
    if not np.isclose(c.sum(), 1, atol=1e-8, rtol=0):
        raise ArithmeticError("population moment quadrature failed")
    return c


def geometry(x):
    vals = np.linalg.svd(x - x.mean(axis=0), compute_uv=False) ** 2
    return dict(pr=float(vals.sum() ** 2 / np.square(vals).sum()), pc1=float(vals.max() / vals.sum()))


def evaluate_dataset(x, population_factor, moment_factor, *, n_draws, seed,
                     k_grid=K_GRID, n_init=5, cluster_seed=0, include_moments=True):
    """M/G retain the historical unguarded comparator; Mg/Gg isolate selection.

    O draws from the actual zero-mean Gaussian population before normalization.
    K arms substitute normalized population covariance and still estimate the
    mean. Every generated dataset remains in the denominator; if its observed
    search fails the guard, the decision is no rejection (p=1).
    """
    if n_draws < 2:
        raise ValueError("at least two draws required")
    x = normalize(x)
    n, d = x.shape
    min_size = max(4, round(.15 * n))
    raw, guarded, k = scores(x, k_grid, min_size, n_init, cluster_seed)
    names = ["M", "Mg", "S", "G", "Gg", "C", "O"]
    if include_moments:
        names += ["KM", "KC"]
    out = dict(geometry(x), observed_k=k, observed=guarded[k] if k is not None else INVALID,
               data_sha256=hashlib.sha256(x.astype('<f8').tobytes()).hexdigest(),
               observed_guard_failed=k is None, seed=str(seed))
    if k is None:
        return dict(out, p={a:1. for a in names}, null_mean={}, null_sd={},
                    null_guard_failures={}, shift=None)
    mean = x.mean(axis=0)
    _, singular, vt = np.linalg.svd(x - mean, full_matrices=False)
    fitted_factor = singular[:, None] / np.sqrt(n - 1) * vt
    fitted = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    oracle = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    known = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    values = {a:np.empty(n_draws) for a in names}
    failures = {a:0 for a in ("S", "C", "O", "KC") if a in names}
    for b in range(n_draws):
        draw = mean + fitted.standard_normal((n, fitted_factor.shape[0])) @ fitted_factor
        for projected, fixed, fixed_g, searched in ((False,"M","Mg","S"),(True,"G","Gg","C")):
            z = normalize(draw) if projected else draw
            r, g, best = scores(z, k_grid, min_size, n_init, cluster_seed)
            values[fixed][b] = r[k]
            values[fixed_g][b] = g[k]
            values[searched][b] = max(g.values(), default=INVALID)
            failures[searched] += best is None
            if values[searched][b] < values[fixed_g][b]:
                raise AssertionError("search statistic must dominate its guarded fixed-k member")
        z = normalize(oracle.standard_normal((n, population_factor.shape[0])) @ population_factor)
        _, g, best = scores(z, k_grid, min_size, n_init, cluster_seed)
        values["O"][b] = max(g.values(), default=INVALID)
        failures["O"] += best is None
        if include_moments:
            z = mean + known.standard_normal((n, moment_factor.shape[0])) @ moment_factor
            r, _, _ = scores(z, (k,), min_size, n_init, cluster_seed)
            values["KM"][b] = r[k]
            _, g, best = scores(normalize(z), k_grid, min_size, n_init, cluster_seed)
            values["KC"][b] = max(g.values(), default=INVALID)
            failures["KC"] += best is None
    p = {a:float((1 + np.count_nonzero(v >= out["observed"])) / (n_draws + 1)) for a,v in values.items()}
    assert p["S"] >= p["Mg"] and p["C"] >= p["Gg"]
    sd = float(values["M"].std(ddof=1))
    shift = float((values["G"] - values["M"]).mean())
    return dict(out, p=p, null_mean={a:float(v.mean()) for a,v in values.items()},
                null_sd={a:float(v.std(ddof=1)) for a,v in values.items()},
                null_guard_failures=failures, shift=shift,
                shift_sd=shift / sd if sd else None)
