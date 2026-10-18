#supplied covariance, plus a regression check against the frozen pre-v7 copy
import importlib.util
import json
import os
import sys
import warnings
from dataclasses import asdict
from pathlib import Path
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ.setdefault(_v,"1")
import numpy as np
import pytest
_R=Path(__file__).resolve().parents[1];_NC=_R/"experiments"/"ds002748_mdd"/"propagated_null"
if str(_NC) not in sys.path:sys.path.insert(0,str(_NC))
def load_module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
nc=load_module("nullcal_cov_test",_NC/"nullcal.py");baseline=load_module("nullcal_frozen_pre_v7",Path(__file__).parent/"fixtures/nullcal_pre_v7.py")
REF=json.loads((Path(__file__).resolve().parent/"nullcal_reference.json").read_text());CONSTRUCTIONS={"M":(False,False),"G":(True,False),"S":(False,True),"C":(True,True)}
def fixture():
    f=REF["fixture"];r=np.random.default_rng(f["seed"]);l=np.exp(-np.arange(f["d"])/f["tau"]);Q,_=np.linalg.qr(r.standard_normal((f["d"],f["d"])))
    X=r.standard_normal((f["n"],f["d"]))@(np.sqrt(l)[:,None]*Q.T);return (nc.l2_normalize(X),f)
def test_fitted_factor_matches_stored_reference():
    Xn,_=fixture();m,A=nc._gaussian_factor(Xn);assert list(A.shape)==REF["gaussian_factor"]["A_shape"];np.testing.assert_allclose(m,Xn.mean(axis=0),rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(A.T@A,np.cov(Xn,rowvar=False),rtol=1e-12,atol=1e-12)
def test_cov_none_reproduces_frozen_baseline_exactly_on_current_runtime():
    Xn,f=fixture()
    for n,(r,m) in CONSTRUCTIONS.items():
        kw=dict(reproject=r,match_selection=m,k_grid=tuple(f["k_grid"]),n_draws=f["n_draws"],min_size=f["min_size"],n_init=f["n_init"],seed=f["null_test_seed"]);g=asdict(nc.null_test(Xn,**kw))
        w=asdict(baseline.null_test(Xn,**kw));assert g==w,n
def test_supplied_cov_reproduces_that_covariance():
    r=np.random.default_rng(11);n,d=(28,12);B=r.standard_normal((d,d));S=B@B.T;X=r.standard_normal((n,d));m,A=nc._gaussian_factor(X,S);assert A.shape==(d,d);assert np.allclose(A.T@A,S,atol=1e-09)
    assert np.allclose(m,X.mean(axis=0));f=nc._gaussian_factor(X)[1];assert not np.allclose(f.T@f,S,atol=1e-06)
def test_supplied_cov_handles_rank_deficiency():
    rg=np.random.default_rng(12);n,d,r=(20,15,5);B=rg.standard_normal((d,r));S=B@B.T;X=rg.standard_normal((n,d));_,A=nc._gaussian_factor(X,S);assert np.allclose(A.T@A,S,atol=1e-09)
    assert np.isfinite(A).all();t=1e-06*np.linalg.norm(A,2);assert np.linalg.matrix_rank(A,tol=t)==r;ds=nc._draw(X.mean(axis=0),A,500,np.random.default_rng(0));assert np.isfinite(ds).all()
    c=ds-ds.mean(axis=0);assert np.linalg.matrix_rank(c,tol=1e-06*np.linalg.norm(c,2))==r
def test_supplied_cov_changes_the_null_and_is_recorded():
    r=np.random.default_rng(13);n,d=(28,20);B=r.standard_normal((d,d));S=B@B.T/d;Xn=nc.l2_normalize(r.standard_normal((n,d))@B.T)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore");a=nc.null_test(Xn,n_draws=60,n_init=3,seed=1);b=nc.null_test(Xn,n_draws=60,n_init=3,seed=1,cov=S)
    assert a.cov_supplied is False and b.cov_supplied is True;assert a.observed==b.observed and a.observed_k==b.observed_k;assert a.null_mean!=b.null_mean
def test_supplied_cov_rejects_a_mismatched_shape():
    X=np.random.default_rng(14).standard_normal((10,6))
    with pytest.raises(ValueError):nc._gaussian_factor(X,np.eye(5))
