#execution paths that broke in an earlier restyle
import importlib.util
import sys
from pathlib import Path
import torch
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
ROOT=Path(__file__).resolve().parents[1]

def test_training_warmup_and_checkpoint(tmp_path,monkeypatch):
    from src.train import joint_train,tune
    from models.attention_pool import condition_attention_pool
    from models.contrastive_loss import NTXentLoss
    from models.augmentations import graph_augmentor
    monkeypatch.setattr(tune,"report",lambda metrics:None)
    torch.manual_seed(123)

    class encoder_cls(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.linear=torch.nn.Linear(5,4)

        def forward(self,batch):
            return global_mean_pool(self.linear(batch.x),batch.batch)

    def make_graph():
        return Data(x=torch.randn(4,5),edge_index=torch.tensor([[0,1,2,3],[1,2,3,0]]),edge_attr=torch.ones(4))
    batch=[{"graphs":[make_graph(),make_graph()]} for _ in range(4)]
    enc,att=(encoder_cls(),condition_attention_pool(d_model=4,num_cons=2))
    w0=enc.linear.weight.detach().clone()
    _,_,tl,vl=joint_train(enc,att,NTXentLoss(temperature=0.2),[batch],[batch],graph_augmentor(mask_rate=0.1,noise_std=0.1),"cpu",str(tmp_path),epochs=3,patience=3,lr=0.001,weight_decay=0.001)
    assert len(tl)==len(vl)==3
    assert torch.isfinite(torch.tensor(tl+vl)).all()
    assert not torch.equal(w0,enc.linear.weight)
    assert list(tmp_path.glob("*.pt"))

def test_selective_inference_cell_manifest(tmp_path,monkeypatch):
    d=ROOT/"experiments/ds002748_mdd/propagated_null"
    monkeypatch.syspath_prepend(str(d))
    spec=importlib.util.spec_from_file_location("selective_regression",d/"r1c_selective_inference.py")
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod,"NDS",2)
    monkeypatch.setattr(mod,"PR_TARGETS",[])
    cells=mod.gen_cells(tmp_path)
    assert [c["name"] for c in cells]==["control_iso","control_iso_nltd"]
    assert all((c["n_datasets"]==2 for c in cells))
    assert all(((tmp_path/c["name"]/"manifest.json").exists() for c in cells))
