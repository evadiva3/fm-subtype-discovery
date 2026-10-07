import numpy as np
import pytest

import nullcal as nc


def anisotropic(n=28,d=119,tau=3.0,seed=0):
    rng=np.random.default_rng(seed)
    lam=np.exp(-np.arange(d)/tau)
    Q,_=np.linalg.qr(rng.standard_normal((d,d)))
    return rng.standard_normal((n,d))@(np.sqrt(lam)[:,None]*Q.T)


def test_guard_matches_paper():
    assert nc.min_cluster_guard(28)==4
    assert nc.min_cluster_guard(51)==8
    assert nc.min_cluster_guard(10)==4
    assert nc.min_cluster_guard(200)==30


def test_l2_normalize_puts_rows_on_sphere():
    X=anisotropic()
    Xn=nc.l2_normalize(X)
    assert np.allclose(np.linalg.norm(Xn,axis=1),1.0)


def test_l2_normalize_handles_zero_row():
    X=np.zeros((3,5))
    X[0,0]=1.0
    out=nc.l2_normalize(X)
    assert np.isfinite(out).all()


def test_participation_ratio_is_not_normalization_invariant():
    X=anisotropic()
    assert nc.participation_ratio(X)!=pytest.approx(
        nc.participation_ratio(X,normalize=True))


def test_participation_ratio_isotropic_upper_bound():
    rng=np.random.default_rng(1)
    X=rng.standard_normal((40,5))
    assert 1.0<=nc.participation_ratio(X)<=5.0


def test_p_value_estimator_never_zero_and_is_a_grid_multiple():
    X=anisotropic()
    B=50
    r=nc.null_test(X,n_draws=B,n_init=3,seed=0)
    assert r.p>0.0
    assert r.p==pytest.approx((r.n_exceed+1)/(B+1))


def test_guard_binds_on_observed_and_draws():
    X=nc.l2_normalize(anisotropic())
    ms=nc.min_cluster_guard(X.shape[0])
    s,k,l=nc.best_silhouette(X,min_size=ms,n_init=5)
    if k is not None:
        assert np.bincount(l).min()>=ms


def test_degenerate_when_no_k_passes_guard():
    X=nc.l2_normalize(anisotropic(n=10,d=20))
    s,k,l=nc.best_silhouette(X,min_size=9,n_init=3)
    assert k is None and l is None
    assert s==nc._DEGENERATE


def test_four_constructions_share_one_observed_statistic():
    X=anisotropic()
    lr=nc.ladder(X,n_draws=40,n_init=3)
    obs={r.observed for r in(lr.M,lr.G,lr.S,lr.C)}
    ks={r.observed_k for r in(lr.M,lr.G,lr.S,lr.C)}
    assert len(obs)==1 and len(ks)==1


def test_construction_labels_map_to_the_two_flags():
    X=anisotropic()
    kw=dict(n_draws=20,n_init=3,seed=0)
    assert nc.null_test(X,reproject=False,match_selection=False,**kw).construction.startswith("M")
    assert nc.null_test(X,reproject=True,match_selection=False,**kw).construction.startswith("G")
    assert nc.null_test(X,reproject=False,match_selection=True,**kw).construction.startswith("S")
    assert nc.null_test(X,reproject=True,match_selection=True,**kw).construction.startswith("C")


def test_gaussian_factor_reproduces_covariance_in_the_row_space():
    X=nc.l2_normalize(anisotropic())
    m,A=nc._gaussian_factor(X)
    assert np.allclose(A.T@A,np.cov(X,rowvar=False),atol=1e-8)
    assert np.allclose(m,X.mean(axis=0))


def test_gaussian_factor_survives_rank_deficiency():
    X=nc.l2_normalize(anisotropic(n=8,d=40))
    m,A=nc._gaussian_factor(X)
    rng=np.random.default_rng(0)
    D=nc._draw(m,A,8,rng)
    assert D.shape==(8,40)and np.isfinite(D).all()


def test_null_test_rejects_an_observed_statistic_with_no_valid_k():
    X=nc.l2_normalize(anisotropic(n=10,d=20))
    with pytest.raises(ValueError):
        nc.null_test(X,min_size=9,n_draws=10,n_init=3)


def test_separation_ratio_requires_two_groups():
    X=anisotropic(n=12,d=6)
    with pytest.raises(ValueError):
        nc.separation_ratio(X,np.zeros(12,dtype=int))


def test_separation_ratio_grows_with_planted_separation():
    rng=np.random.default_rng(3)
    n,d=40,10
    lab=np.repeat([0,1],n//2)
    b=rng.standard_normal((n,d))
    off=np.zeros((n,d))
    off[lab==1,0]=6.0
    assert(nc.separation_ratio(b+off,lab)
        >nc.separation_ratio(b,lab))


def test_reproject_raises_the_null_mean_on_a_graded_spectrum():
    X=nc.l2_normalize(anisotropic(tau=2.0,seed=5))
    kw=dict(n_draws=120,n_init=3,seed=0,match_selection=False)
    a=nc.null_test(X,reproject=False,**kw).null_mean
    s=nc.null_test(X,reproject=True,**kw).null_mean
    assert s>a


def test_calibrate_paired_returns_one_p_per_dataset():
    X=anisotropic()
    cal=nc.calibrate_paired(X,n_datasets=4,n_draws=15,n_init=3,verbose=False)
    assert cal.p_a.shape==cal.p_b.shape==(4,)
    assert((cal.p_a>0)&(cal.p_a<=1)).all()


def test_type1_surface_cell_reports_measured_not_target_pr():
    c=nc.type1_surface_cell(28,40,3.0,n_datasets=4,n_draws=15,n_init=3)
    assert c["n_datasets"]<=4
    assert np.isfinite(c["pr_mean"])
    assert 0.0<=c["rej_M"]<=1.0 and 0.0<=c["rej_C"]<=1.0


def test_wilson_interval_brackets_the_point_estimate():
    lo,hi=nc._wilson(5,400)
    assert lo<5/400<hi and lo>=0.0 and hi<=1.0
    lo0,hi0=nc._wilson(0,400)
    assert lo0==0.0 and hi0>0.0


def test_ks_uniform_is_zero_for_a_perfect_grid():
    p=(np.arange(1,1001)-0.5)/1000.0
    assert nc._ks_uniform(p)<1e-3


def test_propagated_null_checklist_is_text():
    t=nc.propagated_null_checklist()
    assert isinstance(t,str)and "PROPAGATED NULL" in t
