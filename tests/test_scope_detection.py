import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"analysis"))
from scope_detection import means,sizes,one,G
from asymmetric_detection import spectrum

def _basis(n):
    rng=np.random.default_rng(3)
    return np.linalg.qr(rng.normal(size=(119,119)))[0],rng.permutation(n)

def test_centered_families_have_exact_zero_mean_and_protocol_sizes():
    for n in (28,100):
        rot,perm=_basis(n)
        for fam in ("tri","triuneq","pairuneq","tetra"):
            M,lab=means(fam+"4",rot,spectrum(0),perm)
            w=np.bincount(lab)/n
            assert np.allclose(w@M,0)
            assert np.bincount(lab).tolist()==sizes(fam,n)
    assert sizes("triuneq",28)==[14,7,7] and sizes("pairuneq",100)==[75,25] and sizes("tri",100)==[34,33,33]

def test_pairuneq_is_not_symmetric_under_negation_and_tetra_is_regular():
    rot,perm=_basis(28)
    M,_=means("pairuneq4",rot,spectrum(1),perm)
    assert not np.allclose(M[0],-M[1])
    T,_=means("tetra4",rot,spectrum(1),perm)
    lam=spectrum(1)
    w=(rot.T@T.T)/np.sqrt(lam)[:,None]
    g=w.T@w
    off=g[~np.eye(4,dtype=bool)]
    assert np.allclose(off,off[0])

def test_shift_is_unimodal_and_records_run():
    rot,perm=_basis(28)
    S,lab=means("shift4",rot,spectrum(0),perm)
    assert S.shape[0]==1 and set(lab.tolist())=={0}
    r=one((98765,28,0,5,G.index("tetra4"),0,2))
    assert "error" not in r and r["group_sizes"]==[7,7,7,7] and abs(r["projected_mean_shift"])<1e-18
