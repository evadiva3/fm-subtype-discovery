import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"analysis"))
import robustness_detection as rd

def test_s29_laws_have_zero_mean_and_unit_variance():
    rng=np.random.default_rng(0)
    #t(4) has infinite fourth moment so its variance converges slowly
    for law,tol in (("skew",.05),("heavy",.2)):
        z=rd.noise(law,rng,(200000,2))
        assert np.allclose(z.mean(0),0,atol=.02) and np.allclose(z.var(0),1,atol=tol)
    z=rd.noise("skew",rng,(200000,1))
    assert np.mean(z**3)>1.5

def test_s29_alternatives_are_centered_and_sized():
    rng=np.random.default_rng(1)
    rot=np.linalg.qr(rng.normal(size=(119,119)))[0]
    dirs=np.linalg.qr(rng.normal(size=(119,2)))[0]
    lam=rd.spectrum(1)
    for n in (28,100):
        perm=rng.permutation(n)
        for g in ("rand34","hexa4"):
            M,lab=rd.means(g,rot,lam,perm,dirs)
            assert np.allclose(np.bincount(lab)/n@M,0)
            assert np.bincount(lab).tolist()==rd.sizes(g,n)
    assert rd.sizes("hexa4",28)==[8,4,4,4,4,4] and rd.sizes("rand34",100)==[34,33,33]
    for g in ("null","skew","heavy"):
        assert rd.means(g,rot,lam,perm,dirs)==(None,None)

def test_s29_random_directions_are_standardized():
    rng=np.random.default_rng(2)
    rot=np.linalg.qr(rng.normal(size=(119,119)))[0]
    dirs=np.linalg.qr(rng.normal(size=(119,2)))[0]
    lam=rd.spectrum(0)
    M,lab=rd.means("rand34",rot,lam,np.arange(28),dirs)
    cov=(rot*lam)@rot.T
    u=dirs[:,0]
    #before recentering the first group sits 4 standard deviations along u
    raw=M+np.bincount(lab)/28@(4*np.array([[1,0],[-.5,np.sqrt(3)/2],[-.5,-np.sqrt(3)/2]])@np.stack([np.sqrt(v@cov@v)*v for v in dirs.T]))
    assert np.isclose(raw[0]@u/np.sqrt(u@cov@u),4)

def test_s29_records_run():
    r=rd.one_a((98765,28,0,5,rd.GA.index("skew"),0,9))
    assert "error" not in r and set(r["p"])=={"W","D4","K","E"}
    r=rd.one_b((98765,rd.GB.index("heavy"),0,2))
    assert "error" not in r and set(r["p"])=={"M","C"}
