import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"packaging"))
import angular_balanced as ab

def test_exact_even_odd_kernel_decomposition():
    x=ab.normalize(np.random.default_rng(101).normal(size=(30,5)));r=ab.component_scores(x)
    expected=float(np.exp(-2+2*x@x.T).mean())
    assert r["K"]==pytest.approx(expected,abs=1e-14)
    assert r["K"]==r["KE"]+r["KO"]

def test_even_component_preserves_individual_sign_changes():
    rng=np.random.default_rng(102);x=ab.normalize(rng.normal(size=(30,5)));sign=rng.choice([-1,1],size=(30,1))
    a=ab.component_scores(x);b=ab.component_scores(x*sign)
    assert a["KE"]==pytest.approx(b["KE"],abs=1e-14)
    assert a["D4"]==pytest.approx(b["D4"],abs=1e-14)
    assert abs(a["KO"]-b["KO"])>1e-8

def test_odd_component_zero_for_exact_antipodal_pairs():
    x=ab.normalize(np.random.default_rng(103).normal(size=(15,5)))
    assert ab.component_scores(np.vstack([x,-x]))["KO"]==pytest.approx(0,abs=1e-14)

def test_complete_combination_and_affine_invariance():
    rng=np.random.default_rng(104);x=rng.normal(size=(30,4));a=rng.normal(size=(4,4))+4*np.eye(4)
    first=ab.evaluate(x,draws=19,seed=105);second=ab.evaluate(x@a,draws=19,seed=105)
    assert first["p"]==second["p"]
    assert first["p"]["KB"]==min(1.,2*min(first["p"]["KE"],first["p"]["KO"]))
    assert all(0<p<=1 and p*20==pytest.approx(round(p*20)) for p in first["p"].values())
