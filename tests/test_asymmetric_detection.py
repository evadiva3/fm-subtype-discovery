import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"analysis"))
from asymmetric_detection import means,one,spectrum,G

def _basis():
    rng=np.random.default_rng(1)
    return np.linalg.qr(rng.normal(size=(119,119)))[0],rng.permutation(28)

def test_tri_is_centered_and_not_antipodal():
    rot,perm=_basis();lam=spectrum(1)
    M,lab=means("tri4",rot,lam,perm)
    assert np.allclose(M.sum(axis=0),0)
    assert sorted(np.bincount(lab).tolist())==[9,9,10]
    #no component mean is the negative of another
    assert not any(np.allclose(M[i],-M[j]) for i in range(3) for j in range(3))

def test_antipodal_pair_and_shift_geometry():
    rot,perm=_basis();lam=spectrum(0)
    A,la=means("antipodal2",rot,lam,perm)
    assert np.allclose(A[0],-A[1]) and np.bincount(la).tolist()==[14,14]
    P,_=means("pair4",rot,lam,perm)
    S,ls=means("shift4",rot,lam,perm)
    assert np.allclose(S[0],P.mean(axis=0)) and set(ls.tolist())=={0}
    #each component mean has mahalanobis length delta
    w=rot.T@P.T/np.sqrt(lam)[:,None]
    assert np.allclose(np.linalg.norm(w,axis=0),4)

def test_alternatives_share_null_noise_and_nested_sources():
    a=one((98765,0,5,G.index("null"),0,2))
    b=one((98765,0,10,G.index("null"),0,2))
    c=one((98765,0,5,G.index("tri4"),0,2))
    d=one((98765,0,5,G.index("calibration"),0,2))
    assert not any("error" in r for r in (a,b,c,d))
    assert a["source_sha256"]==b["source_sha256"] and a["data_seed"]==c["data_seed"]!=d["data_seed"]
    assert c["projected_between_scatter"]>0 and a["projected_between_scatter"]==0
