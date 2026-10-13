import sys
from pathlib import Path
import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"packaging"))
import nullcal_validation as nv

def test_hierarchical_streams_do_not_overlap():
    seeds=[nv.stream_seed(20260914,7,a,b,c,d) for a in range(7) for b in range(4) for c in range(100) for d in range(3)]
    assert len(seeds)==len(set(seeds))
    assert nv.stream_seed(9,7,3,2,17,0)==nv.stream_seed(9,7,3,2,17,0)

def test_old_seed_bug_is_reproduced_by_audit():
    seeds=[base+1000*i for base in [6000000,6100000,6200000,6300000] for i in range(400)]
    assert len(seeds)==1600 and len(set(seeds))==700

def test_covariance_rejects_negative_eigenvalues():
    with pytest.raises(ValueError,match="positive semidefinite"):
        nv.factor_covariance(np.diag([1.0,-1.0]))
    cov=np.array([[1.0,1.0],[1.0,1.0]])
    fac=nv.factor_covariance(cov)
    np.testing.assert_allclose(fac.T@fac,cov,atol=1e-12)

def test_spherical_moments_are_not_angular_shape():
    m=nv.sphere_moments(np.array([9.0,1.0]))
    np.testing.assert_allclose(m,[0.75,0.25],atol=1e-08)
    assert not np.allclose(m,[0.9,0.1])
    np.testing.assert_allclose(nv.sphere_moments(np.ones(6)),np.ones(6)/6,atol=1e-08)

def test_score_uses_the_same_euclidean_silhouette():
    x=nv.normalize(np.random.default_rng(4).normal(size=(28,12)))
    best,fixed,extra=nv.scores(x,(2,3),n_init=3,cluster_seed=0)
    for k in (2,3):
        lab=KMeans(n_clusters=k,n_init=3,random_state=0).fit_predict(x)
        assert best[k]==pytest.approx(silhouette_score(x,lab),abs=1e-12)
        assert fixed[k]<=best[k]

def test_paired_search_dominates_fixed_k_and_draw_grid_is_exact():
    fac=np.diag(np.sqrt([9.0,3.0,1.0,0.5]))
    x=np.random.default_rng(5).normal(size=(28,4))@fac
    res=nv.evaluate_dataset(x,fac,fac,n_draws=19,seed=77,n_init=2)
    assert res["p"]["S"]>=res["p"]["Mg"]
    assert res["p"]["C"]>=res["p"]["Gg"]
    assert all((p>0 and p<=1 and (p*20==pytest.approx(round(p*20))) for p in res["p"].values()))

def test_guard_failure_is_retained_as_no_rejection():
    x=np.random.default_rng(6).normal(size=(7,4))
    res=nv.evaluate_dataset(x,np.eye(4),np.eye(4),n_draws=19,seed=0,n_init=2)
    assert res["observed_guard_failed"]
    assert set(res["p"].values())=={1.0}

def test_results_do_not_depend_on_execution_order():
    eye=np.eye(4)
    x=np.random.default_rng(7).normal(size=(18,4))
    kw=dict(n_draws=9,seed=nv.stream_seed(1,2,3,4,5),n_init=2)
    r1=nv.evaluate_dataset(x,eye,eye,**kw)
    np.random.seed(999)
    r2=nv.evaluate_dataset(x,eye,eye,**kw)
    assert r1==r2
