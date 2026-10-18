import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"packaging"))
import angular_fourth as af

def test_tensor_discrepancy_identity():
    w=af.normalize(np.random.default_rng(81).normal(size=(15,3)))
    tensor=np.einsum("ni,nj,nk,nl->ijkl",w,w,w,w)/len(w)
    eye=np.eye(3)
    uniform=(np.einsum("ij,kl->ijkl",eye,eye)+np.einsum("ik,jl->ijkl",eye,eye)+np.einsum("il,jk->ijkl",eye,eye))/15
    assert af.discrepancy(w)==pytest.approx(np.sum((tensor-uniform)**2),abs=1e-12)

def test_fitted_discrepancy_affine_invariance():
    x=np.random.default_rng(82).normal(size=(30,4))
    a=np.array([[2,1,0,0],[0,3,.5,0],[0,0,1,1],[0,0,0,.5]])
    first=af.evaluate(x,draws=19,seed=83);second=af.evaluate(x@a,draws=19,seed=83)
    assert first["statistic"]==pytest.approx(second["statistic"],abs=1e-8)
    assert first["p"]==second["p"]
    assert 0<first["p"]<=1 and first["p"]*20==pytest.approx(round(first["p"]*20))
