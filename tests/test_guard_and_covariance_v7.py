import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"packaging"))
from analysis.evaluate import cluster_evaluate
from config import config
import nullcal as nc

def test_evaluator_keeps_guard_failure_as_no_rejection(monkeypatch):
    monkeypatch.setattr(config,"kmeansKRange",[2,3])
    monkeypatch.setattr(config,"minClusterSizeFloor",4)
    x=np.random.default_rng(2).normal(size=(7,5))
    ev=cluster_evaluate()
    assert ev._select_best_silhouette(x,return_k=True)==(-1.0,None)
    p,kc=ev.perm(x,[0,0,0,1,1,1,1],n_permutations=9,return_k_counts=True)
    assert p==1 and kc=={0:9}

def test_selection_recomputes_observed_instead_of_using_supplied_labels(monkeypatch):
    monkeypatch.setattr(config,"kmeansKRange",[2,3])
    monkeypatch.setattr(config,"kmeansNInit",2)
    x=np.random.default_rng(12).normal(size=(28,5))
    ev=cluster_evaluate()
    p2=ev.perm(x,np.arange(28)%2,n_permutations=9,random_state=8)
    p3=ev.perm(x,np.arange(28)%3,n_permutations=9,random_state=8)
    assert p2==p3

@pytest.mark.parametrize("cov",[np.diag([1.0,-1.0]),np.array([[1.0,1.0],[0.0,1.0]]),np.diag([1.0,np.nan])])
def test_legacy_supplied_covariance_rejects_invalid_matrix(cov):
    with pytest.raises(ValueError):
        nc._gaussian_factor(np.ones((5,2)),cov)

def test_seed_blocks_reject_overlap():
    d=ROOT/"experiments"/"ds002748_mdd"/"propagated_null"
    sys.path.insert(0,str(d))
    import ov_common
    with pytest.raises(ValueError,match="overlap"):
        ov_common.assert_seed_blocks([6000000,6100000],400)
    ov_common.assert_seed_blocks([6000000,16000000,26000000,36000000],400)

def test_checkpoint_refuses_changed_configuration(tmp_path):
    import ov_common
    path=tmp_path/"state.json"
    ov_common.save_state(path,ov_common.load_state(path,"test",{"draws":199}))
    with pytest.raises(ValueError,match="configuration"):
        ov_common.load_state(path,"test",{"draws":999})
