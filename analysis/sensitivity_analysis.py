# sensitivity_analysis.py: rebuild graphs at different 
# edge thresholds + retrain configs, 
# check if clustering result stays stable


import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (_REPO_ROOT, _REPO_ROOT / "src", _REPO_ROOT / "models"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import config
from clustering import cluster
from gnn_encoder import GNNEncoder
from models.attention_pool import condition_attention_pool
from dataset import datasetPreparation
from train import joint_train
from contrastive_loss import NTXentLoss
from augmentations import graph_augmentor
from torch_geometric.data import Data
from torch.utils.data import DataLoader, random_split



_EDGE_PERCENTILE={"value": config.EDGE_PERCENTILE}

#rebuild edge_index/edge_attr at a given percentilex
def _patched_edgeIndexAttr(self, loadFile, FCMatrix):
    if loadFile:
        data=np.load(self.FCMatricesFilePath)
    else:
        data=FCMatrix
    pct=_EDGE_PERCENTILE["value"]
    indexing=np.where(np.abs(data) >= np.percentile(np.abs(data), pct))
    indexList=np.array([indexing[0], indexing[1]])
    indexList=np.concatenate((indexList, indexList[::-1]), axis=1)
    edgeAttr=data[indexList[0], indexList[1]]
    return [torch.tensor(indexList, dtype=torch.long), torch.tensor(edgeAttr, dtype=torch.float32)]



datasetPreparation.edgeIndexAttr=_patched_edgeIndexAttr

#change which percentile the patch uses next
def set_edge_percentile(pct):
    _EDGE_PERCENTILE["value"]=pct


#pull the best silhouette score out of a KMeansUse result
def _best_silhouette(kmeans_result):
    return float(kmeans_result[0]["silhouette_score"].max())

#cluster FM subjects both before and after removing the FM-HC axis and return both silhouettes
def compute_original_orthogonal(runner):
    runner._split_fm_hc()
    original=runner.KMeansUse()  
    original_sil=_best_silhouette(original)

    fmTensor, fmIds=runner._stack(runner.fmEmbed)
    hcTensor, hcIds=runner._stack(runner.hcEmbed)
    fmProjected=runner.project_ortho(fmTensor, hcTensor)
    orthogonal=runner.KMeansUse(fmProjected, fmIds)
    orthogonal_sil=_best_silhouette(orthogonal)
    return original_sil, orthogonal_sil

# load encoder+attention weights from a checkpoint file, or return None if it doesn't exist
def load_trained_model(checkpoint_path):
    if not Path(checkpoint_path).exists():
        return None
    encoder=GNNEncoder()
    attention=condition_attention_pool()
    ckpt=torch.load(checkpoint_path, map_location="cpu")
    encoder.load_state_dict(ckpt["model"])
    attention.load_state_dict(ckpt["pool"])
    encoder.eval()
    attention.eval()
    return encoder, attention

#run a subject dataset through the encoder+pool and get clustering silhouettes back
def evaluate_clustering(encoder, attention, dataset, conditionList):
    runner=cluster(
        encoder,
        str(config.CHECKPOINT_DIR),
        config.JOINT_CHECKPOINT_PATH.name,
        conditionList,
        dataset.subjectList,
    )
    runner.deploy(dataset.subjectData)
    runner.setAttention(attention)
    return compute_original_orthogonal(runner)

#build the real subject dataset at the current percentile
def _default_dataset_builder():
    return datasetPreparation(fm_only=False)

#train a fresh encoder+pool from scratch on one dataset via train.joint_train
def _train_encoder(dataset, epochs, patience, pct, device):
    subjects=dataset.subjectData
    n_val=max(1, int(len(subjects)*config.VAL_FRACTION))
    n_train=len(subjects)-n_val
    gen=torch.Generator().manual_seed(config.RANDOM_SEED)
    train_split, val_split=random_split(subjects, [n_train, n_val], generator=gen)
    train_load=DataLoader(train_split, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=lambda b: b)
    val_load=DataLoader(val_split, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=lambda b: b)
    encoder=GNNEncoder().to(device)
    attention=condition_attention_pool().to(device)
    loss_fn=NTXentLoss()
    augmentor=graph_augmentor()
    save_dir=config.CHECKPOINT_DIR / f"sensitivity_pct_{pct}"
    encoder, attention, _, _=joint_train(encoder, attention, loss_fn, train_load, val_load, augmentor, device, str(save_dir), epochs=epochs, patience=patience)
    encoder.to("cpu").eval()
    attention.to("cpu").eval()
    return encoder, attention

#retrains a model from strach at every percentile, then clusters
#this measures threshold robustness of the full train+cluster method, not a fixed
#model tolerating a graph density shift one full training run per percentile (slow, on purpose)
def run_percentile_sweep(conditionList, epochs=None, patience=None, dataset_builder=None, device=None):
    epochs=config.EPOCHS if epochs is None else epochs
    patience=config.PATIENCE if patience is None else patience
    dataset_builder=_default_dataset_builder if dataset_builder is None else dataset_builder
    device=config.DEVICE if device is None else device
    rows=[]
    for pct in config.EDGE_PERCENTILE_SENSITIVITY:
        print(f"[percentile sweep] pct={pct}: retraining from scratch (expensive) ...")
        set_edge_percentile(pct)
        try:
            dataset=dataset_builder()
            encoder, attention=_train_encoder(dataset, epochs, patience, pct, device)
            orig, ortho=evaluate_clustering(encoder, attention, dataset, conditionList)
        except Exception as error:
            set_edge_percentile(config.EDGE_PERCENTILE)
            return None, f"retrain/cluster failed at percentile {pct}: {type(error).__name__}: {error}"
        rows.append(
            {"percentile": pct, "silhouette_original": orig, "silhouette_orthogonal": ortho}
        )
    set_edge_percentile(config.EDGE_PERCENTILE)
    return pd.DataFrame(rows), None


# grab the current main hyperparameter settings as a dict
def _primary_config():
    return {
        "name": "primary",
        "D_MODEL": config.D_MODEL,
        "HEADS": config.HEADS,
        "LAYERS": config.LAYERS,
        "DROPOUT": config.DROPOUT,
        "LR": config.LR,
    }

#read the 3 alternate hyperparameter configs from a JSON file, or report they're missing
def load_alternate_configs():
    path=config.RESULTS_ROOT / "sensitivity_configs.json"
    if not path.exists():
        return None, (
            f"no alternate hyperparameter configs found (expected {path}); "
            "the 3 alternates are undefined anywhere in the repo and must not be invented"
        )
    with open(path) as f:
        alternates=json.load(f)
    return alternates, None

#figure out which checkpoint file belongs to a given hyperparameter config
def _checkpoint_for_config(cfg):
    if cfg["name"]=="primary":
        return config.JOINT_CHECKPOINT_PATH
    return config.CHECKPOINT_DIR / f"best_joint_model_{cfg['name']}.pt"

#test each hyperparameter config (each needs its own trained checkpoint),
#  save results, or report whats missing
def run_hyperparameter_sweep(conditionList):
    set_edge_percentile(config.EDGE_PERCENTILE)

    alternates, blocker=load_alternate_configs()
    if alternates is None:
        return None, blocker

    all_configs=[_primary_config()] + list(alternates)
    missing_ckpts=[c["name"] for c in all_configs if not Path(_checkpoint_for_config(c)).exists()]
    if missing_ckpts:
        return None, (
            "missing trained checkpoint(s) for config(s): "
            + ", ".join(missing_ckpts)
            + " (a fully trained checkpoint per hyperparameter config is required)"
        )

    rows = []
    for cfg in all_configs:
        # Encoder/pool read config.* at construction, so apply the config first.
        _apply_config(cfg)
        model=load_trained_model(_checkpoint_for_config(cfg))
        encoder, attention=model
        dataset=datasetPreparation(fm_only=False)
        orig, _=evaluate_clustering(encoder, attention, dataset, conditionList)
        rows.append(
            {
                "config_name": cfg["name"],
                "d_model": cfg["D_MODEL"],
                "heads": cfg["HEADS"],
                "layers": cfg["LAYERS"],
                "dropout": cfg["DROPOUT"],
                "lr": cfg["LR"],
                "silhouette": orig,
            }
        )
    _apply_config(_primary_config()) 
    return pd.DataFrame(rows), None

#push a hyperparameter configs values into the shared config object before building a model
def _apply_config(cfg):
    # GNNEncoder / attention pool read these off the shared config instance at
    # construction time, so mutate them before building a model for `cfg`
    config.D_MODEL=cfg["D_MODEL"]
    config.HEADS=cfg["HEADS"]
    config.LAYERS=cfg["LAYERS"]
    config.DROPOUT=cfg["DROPOUT"]
    config.LR = cfg["LR"]

def structural_self_test():
   # Proves the pipeline is structurally sound without real checkpoints
    print("[self-test] edge-percentile graph construction ...")
    fake_fc=np.random.randn(config.N_NODES, config.N_NODES).astype(np.float32)
    for pct in config.EDGE_PERCENTILE_SENSITIVITY:
        set_edge_percentile(pct)
        packaged=_patched_edgeIndexAttr(type("S", (), {})(), False, fake_fc)
        edge_index, edge_attr=packaged[0], packaged[1]
        assert edge_index.shape[0]==2, "edge_index must be [2, E]"
        assert edge_index.shape[1]==edge_attr.shape[0], "edge count must match"
        print(f"  pct={pct}: edges={edge_attr.shape[0]}")
    set_edge_percentile(config.EDGE_PERCENTILE)

    print("[self-test] clustering silhouette computation ...")
    conditionList=list(config.CONDITIONS)
    n_fm, n_hc, d=12, 8, config.D_MODEL
    subjectList=[f"sub-fm{i:03d}" for i in range(n_fm)] + [f"sub-hc{i:03d}" for i in range(n_hc)]
    runner=cluster(None, str(config.CHECKPOINT_DIR), config.JOINT_CHECKPOINT_PATH.name, conditionList, subjectList)
    torch.manual_seed(config.RANDOM_SEED)
    runner.attentionEmbeddings={}
    runner.groupLabels={}
    for i in range(n_fm):
        runner.attentionEmbeddings[f"sub-fm{i:03d}"]=torch.randn(d)
        runner.groupLabels[f"sub-fm{i:03d}"]=0
    for i in range(n_hc):
        runner.attentionEmbeddings[f"sub-hc{i:03d}"]=torch.randn(d)
        runner.groupLabels[f"sub-hc{i:03d}"]=1
    orig, ortho=compute_original_orthogonal(runner)
    print(f"  synthetic silhouette_original={orig:.4f} silhouette_orthogonal={ortho:.4f}")

    retrain_sweep_self_test()
    print("[self-test] Good! The pipeline is structurally sound and ready to run.")
    return True


#one tiny random graph: [n_nodes,5] features + a few random edges (matches encoder in_channels=5)
def _tiny_graph(n_nodes=8):
    x=torch.randn(n_nodes, 5)
    ei=torch.randint(0, n_nodes, (2, n_nodes*2))
    ea=torch.randn(ei.shape[1])
    return Data(x=x, edge_index=ei, edge_attr=ea)

#fake dataset with .subjectData/.subjectList shaped exactly like datasetPreparation output
def _synthetic_dataset(n_fm=14, n_hc=6, n_cons=7, n_nodes=8):
    ds=type("S", (), {})()
    ds.subjectData=[]
    ds.subjectList=[]
    for label, tmpl, count in ((0, "sub-fm%03d", n_fm), (1, "sub-hc%03d", n_hc)):
        for i in range(count):
            sid=tmpl % i
            graphs=[_tiny_graph(n_nodes) for _ in range(n_cons)]
            ds.subjectData.append({"subject_id": sid, "graphs": graphs, "group_label": torch.tensor([label])})
            ds.subjectList.append(sid)
    return ds

#fast proof the retrain-per-threshold loop runs end to end on synthetic graphs (no real data)
def retrain_sweep_self_test():
    print("[self-test] retrain-per-threshold sweep on synthetic graphs (2 epochs each, cpu) ...")
    torch.manual_seed(config.RANDOM_SEED)
    conditionList=list(config.CONDITIONS)
    df, blocker=run_percentile_sweep(
        conditionList,
        epochs=2,
        patience=2,
        dataset_builder=_synthetic_dataset,
        device=torch.device("cpu"),
    )
    assert blocker is None, f"retrain sweep blocked: {blocker}"
    assert len(df)==len(config.EDGE_PERCENTILE_SENSITIVITY), "one row per percentile"
    print(df.to_string(index=False))
    return True



def main():
    conditionList=list(config.CONDITIONS)
    results_dir=config.RESULTS_ROOT
    os.makedirs(results_dir, exist_ok=True)
    blockers=[]

    print("=" * 70)
    print("Sensitivity analysis (paper Section 4.5)")
    print(f"  EDGE_PERCENTILE_SENSITIVITY = {config.EDGE_PERCENTILE_SENSITIVITY}")
    print(f"  primary edge percentile     = {config.EDGE_PERCENTILE}")
    print("  paper/draft_v1.md is empty -> using two-separate-tables layout (flagged)")
    print("=" * 70)

    # percentile
    pct_df, pct_blocker=run_percentile_sweep(conditionList)
    if pct_df is not None:
        out=results_dir / "sensitivity_percentile_sweep.csv"
        pct_df.to_csv(out, index=False)
        print(f"[percentile sweep] wrote {out}")
        print(pct_df.to_string(index=False))
    else:
        blockers.append(f"percentile sweep BLOCKED: {pct_blocker}")

    # hyperparameters
    hp_df, hp_blocker=run_hyperparameter_sweep(conditionList)
    if hp_df is not None:
        out=results_dir / "sensitivity_hyperparameter_sweep.csv"
        hp_df.to_csv(out, index=False)
        print(f"[hyperparameter sweep] wrote {out}")
        print(hp_df.to_string(index=False))
    else:
        blockers.append(f"hyperparameter sweep BLOCKED: {hp_blocker}")

    if blockers:
        print("\n" + "-" * 70)
        print("BLOCKERS (no results fabricated):")
        for b in blockers:
            print(f"  - {b}")
        print("-" * 70)
        print("Running structural self-test against synthetic data instead ...")
        structural_self_test()

    return blockers


if __name__ == "__main__":
    main()
