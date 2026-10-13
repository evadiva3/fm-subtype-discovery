import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"packaging"))
import angular_null as an

def test_tyler_whitened_moment_equation():
    x=np.random.default_rng(9).normal(size=(40,5))*np.array([10,3,2,1,.5])
    s,info=an.tyler_shape(x)
    assert np.trace(s)==pytest.approx(5)
    w=an.whiten(x,s)
    np.testing.assert_allclose((5/40)*w.T@w,np.eye(5),atol=1e-7)

def test_affine_transform_preserves_whitened_distances_and_statistic():
    rng=np.random.default_rng(11)
    x=rng.normal(size=(40,5))
    a=rng.normal(size=(5,5))+5*np.eye(5)
    a=a@np.diag([8,4,2,1,.5])
    u=an.normalize(x);v=an.normalize(x@a)
    su,_=an.tyler_shape(u);sv,_=an.tyler_shape(v)
    wu=an.whiten(u,su);wv=an.whiten(v,sv)
    np.testing.assert_allclose(wu@wu.T,wv@wv.T,atol=1e-7)
    assert an.statistic(wu)[0]==pytest.approx(an.statistic(wv)[0],abs=1e-8)
    assert an.statistic(wu)[1]==an.statistic(wv)[1]

def test_positive_row_scale_does_not_change_shape():
    rng=np.random.default_rng(12);x=rng.normal(size=(30,5))
    a,_=an.tyler_shape(x);b,_=an.tyler_shape(x*np.exp(rng.normal(size=(30,1))))
    np.testing.assert_allclose(a,b,atol=1e-10)

def test_projection_commutes_with_row_normalization():
    x=np.random.default_rng(5).normal(size=(30,15))
    p=an.fixed_projection(15,5,100)
    np.testing.assert_allclose(p.T@p,np.eye(5),atol=1e-12)
    np.testing.assert_allclose(an.normalize(x@p),an.normalize(an.normalize(x)@p),atol=1e-12)

def test_nonidentifiable_shape_is_not_regularized_silently():
    with pytest.raises(ValueError,match="requires n"):
        an.tyler_shape(np.eye(5))
    x=np.ones((30,5))
    with pytest.raises(ValueError,match="rank deficient"):
        an.tyler_shape(x)

def test_complete_refit_reference_is_affine_invariant():
    x=np.random.default_rng(20).normal(size=(24,4))
    a=np.array([[2.,1,0,0],[0,3,1,0],[0,0,1,.5],[0,0,0,.7]])
    first=an.evaluate(x,draws=19,seed=99,n_init=2)
    second=an.evaluate(x@a,draws=19,seed=99,n_init=2)
    assert first["p"]["W"]==second["p"]["W"]
    assert first["observed"]["W"]==pytest.approx(second["observed"]["W"],abs=1e-8)
    assert all(p*20==pytest.approx(round(p*20)) and 0<p<=1 for p in first["p"].values())

def test_failed_guard_is_kept_in_denominator():
    x=np.random.default_rng(21).normal(size=(7,3))
    result=an.evaluate(x,draws=9,seed=10,n_init=2)
    assert result["raw_guard_failed"] and result["whitened_guard_failed"]
    assert set(result["p"].values())=={1.}

def test_iteration_failure_is_explicit():
    x=np.random.default_rng(22).normal(size=(30,5))
    with pytest.raises(ArithmeticError,match="did not converge"):
        an.tyler_shape(x,max_iter=1)
