import sys
from pathlib import Path
_R=Path(__file__).resolve().parents[1]
for _p in (_R,_R/"src",_R/"models",_R/"analysis"):
    if str(_p) not in sys.path:
        sys.path.insert(0,str(_p))
import torch
from torch_geometric.data import Data,Batch
from config import config
from gnn_encoder import GNNEncoder

def _g(n=30,s=0):
    r=torch.Generator().manual_seed(s)
    x=torch.randn(n,5,generator=r)
    ei=torch.randint(0,n,(2,n*3),generator=r)
    ea=torch.randn(n*3,generator=r)
    return Data(x=x,edge_index=ei,edge_attr=ea)

def _b(ng=4,n=30):
    return Batch.from_data_list([_g(n,i) for i in range(ng)])

def test_forward_shape():
    torch.manual_seed(config.randomSeed)
    e=GNNEncoder().eval()
    ng=4
    with torch.no_grad():
        o=e(_b(ng))
    assert o.shape==(ng,config.dModel)
    assert torch.isfinite(o).all()

def test_gradient_flow():
    torch.manual_seed(config.randomSeed)
    e=GNNEncoder().train()
    o=e(_b(4))
    o.sum().backward()
    gs=[p.grad for p in e.parameters() if p.requires_grad]
    assert len(gs)>0
    assert any(g is not None for g in gs)
    assert any(g is not None and torch.isfinite(g).all() and torch.any(g!=0) for g in gs)

def test_batch_size_independent():
    torch.manual_seed(config.randomSeed)
    e=GNNEncoder().eval()
    with torch.no_grad():
        o2=e(_b(2))
        o5=e(_b(5))
    assert o2.shape[1]==o5.shape[1]==config.dModel
    assert o2.shape[0]==2 and o5.shape[0]==5
