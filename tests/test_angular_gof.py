import sys
from pathlib import Path
import numpy as np
import pytest
from scipy.special import hyp0f1
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"packaging"));sys.path.insert(0,str(ROOT/"analysis"))
import angular_gof as gof
import validate_angular_gof_v9 as study

def test_published_kernel_formula_against_literal_distances():
    rng=np.random.default_rng(91);x=gof.normalize(rng.normal(size=(9,3)));y=gof.normalize(rng.normal(size=(11,3)))
    def total(a,b):return sum(np.exp(-np.sum((u-v)**2)) for u in a for v in b)
    expected=(len(y)/len(x)*total(x,x)-2*total(x,y)+len(x)/len(y)*total(y,y))/(len(x)+len(y))
    assert gof.two_sample_kernel(x,y)==pytest.approx(expected,abs=1e-12)

def test_uniform_kernel_expectation_in_three_dimensions():
    #dot product with a fixed direction is uniform(-1,1) on S^2
    expected=(1-np.exp(-4))/4
    assert np.exp(-2)*hyp0f1(1.5,1)==pytest.approx(expected,abs=1e-14)

def test_whitened_scores_affine_invariance():
    rng=np.random.default_rng(92);x=rng.normal(size=(30,4));a=rng.normal(size=(4,4))+4*np.eye(4)
    first,_=gof.tyler_shape(x);second,_=gof.tyler_shape(x@a)
    v1,k1=gof.whitened_scores(gof.whiten(x,first));v2,k2=gof.whitened_scores(gof.whiten(x@a,second))
    for arm in v1:assert v1[arm]==pytest.approx(v2[arm],abs=1e-8)
    assert k1==k2

def test_bootstrap_refits_both_reference_constructions(monkeypatch):
    original=gof.tyler_shape;calls=[]
    def counted(x):calls.append(x.shape);return original(x)
    monkeypatch.setattr(gof,"tyler_shape",counted)
    x=np.random.default_rng(93).normal(size=(14,3))
    first=gof.evaluate(x,draws=9,seed=94,bootstrap_seed=95,auxiliary=12)
    assert len(calls)==1+2*9
    assert all(0<p<=1 and p*10==pytest.approx(round(p*10)) for p in first["p"].values())
    assert first==gof.evaluate(x,draws=9,seed=94,bootstrap_seed=95,auxiliary=12)

def test_full_dimension_projection_preserves_data():
    x,_=study.generate("two_modes",60,0);p,seed=study.projection(20)
    np.testing.assert_array_equal(x@p,x);assert seed is None

def test_sources_are_shared_only_across_projections():
    hashes=[]
    for family in study.FAMILIES:
        x,a=study.prepare(family,60,5,0);y,b=study.prepare(family,60,20,0)
        assert a["source_data_sha256"]==b["source_data_sha256"]
        assert a["null_seed"]!=b["null_seed"];hashes.append(a["source_data_sha256"])
    assert len(set(hashes))==len(study.FAMILIES)
