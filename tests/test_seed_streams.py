#seed streams for the four-seed checks, and the supplied mean
import sys
from pathlib import Path
import numpy as np
import pytest
_NC=Path(__file__).resolve().parents[1]/"experiments"/"ds002748_mdd"/"propagated_null";sys.path.insert(0,str(_NC))
import nullcal as nc
import ov_common as ov
@pytest.mark.parametrize("bases",[[6000000,6100000,6200000,6300000],[300000,810000,820000,830000],[900000,710000,720000,730000],[100000,910000,920000,930000]])
def test_archived_bases_collide(bases):
    with pytest.raises(ValueError):ov.assert_seed_blocks(bases,400)
def test_hashed_seeds_are_disjoint_across_four_replications():
    for c in (0,1,21):
        s=[ov.hashed_seed(20260916,30,c,r,i) for r in range(4) for i in range(400)];assert len(set(s))==1600
    e=[ov.hashed_seed(20260916,30,c,r,i,p) for c in (0,1,21) for r in range(4) for i in range(400) for p in (0,1)];assert len(set(e))==len(e)
def test_hashed_seed_is_deterministic_and_nonnegative():
    a=ov.hashed_seed(1,2,3,4,5);assert a==ov.hashed_seed(1,2,3,4,5);assert 0<=a<2**64
    with pytest.raises(ValueError):ov.hashed_seed(1,2,-3,4,5)
def _data():
    r=np.random.default_rng(3);return nc.l2_normalize(r.standard_normal((20,6))*np.linspace(2,0.5,6))
def test_supplied_mean_centres_draws():
    X=_data();c=np.diag(np.linspace(1,0.1,6));m,A=nc._gaussian_factor(X,c,np.zeros(6));assert np.array_equal(m,np.zeros(6));np.testing.assert_allclose(A.T@A,c,atol=1e-12)
    m2,A2=nc._gaussian_factor(X,c);np.testing.assert_allclose(m2,X.mean(axis=0));assert np.array_equal(A,A2)
def test_supplied_mean_validation():
    X=_data()
    with pytest.raises(ValueError):nc._gaussian_factor(X,None,np.zeros(6))
    with pytest.raises(ValueError):nc._gaussian_factor(X,np.eye(6),np.zeros(5))
    with pytest.raises(ValueError):nc._gaussian_factor(X,np.eye(6),np.full(6,np.nan))
def test_mean_none_is_bit_identical_in_null_test():
    X=_data();cv=np.eye(6)/6;kw=dict(n_draws=15,n_init=2,seed=11,k_grid=(2,3),min_size=4)
    for c in (None,cv):
        a=nc.null_test(X,cov=c,**kw);b=nc.null_test(X,cov=c,mean=None,**kw);assert a==b
    z=nc.null_test(X,cov=cv,mean=np.zeros(6),**kw);assert z.p>0 and z.cov_supplied
