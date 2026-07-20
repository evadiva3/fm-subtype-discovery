import sys
import math
from pathlib import Path
_R=Path(__file__).resolve().parents[1]
for _p in (_R,_R/"src",_R/"models",_R/"analysis"):
    if str(_p) not in sys.path:
        sys.path.insert(0,str(_p))
import torch
from config import config
from contrastive_loss import NTXentLoss

def test_orthonormal_aligned_analytic():
    b=4
    t=config.ntXentTemp
    v=torch.eye(b)
    q=NTXentLoss(temperature=t)(v,v).item()
    s=1.0/t
    e=math.log(math.exp(s)+(2*b-2))-s
    assert abs(q-e)<1e-4
    assert q>0

def test_all_identical_max_confusion():
    b=4
    t=config.ntXentTemp
    v=torch.ones(b,8)
    q=NTXentLoss(temperature=t)(v,v).item()
    e=math.log(2*b-1)
    assert abs(q-e)<1e-4

def test_aligned_below_confused():
    b=4
    t=config.ntXentTemp
    a=NTXentLoss(temperature=t)(torch.eye(b),torch.eye(b)).item()
    c=NTXentLoss(temperature=t)(torch.ones(b,8),torch.ones(b,8)).item()
    assert a<c

def test_scale_invariant():
    torch.manual_seed(config.randomSeed)
    z1=torch.randn(6,16)
    z2=torch.randn(6,16)
    f=NTXentLoss()
    a=f(z1,z2).item()
    b=f(z1*7.5,z2*7.5).item()
    assert abs(a-b)<1e-4
    assert math.isfinite(a)
