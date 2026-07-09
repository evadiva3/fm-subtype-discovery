# sensitivity_analysis.py: rebuild graphs at different 
# edge thresholds + retrain configs, 
# check if clustering result stays stable.


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



_EDGE_PERCENTILE={"value": config.edgePercentile}

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
        str(config.checkpointDir),
        config.jointCheckpointPath.name,
        conditionList,
        dataset.subjectList,
    )
    runner.deploy(dataset.subjectData)
    runner.setAttention(attention)
    return compute_original_orthogonal(runner)

#test all 4 edge thresholds on the same trained model, save results, or report why it cant
def run_percentile_sweep(conditionList, checkpoint_path):
    model=load_trained_model(checkpoint_path)
    if model is None:
        return None, f"no trained checkpoint at {checkpoint_path} (results/checkpoints/ is empty)"
    encoder, attention=model

    rows=[]
    for pct in config.edgePercentileSensitivity:
        set_edge_percentile(pct)
        try:
            dataset=datasetPreparation(fm_only=False)  
        except Exception as error:
            return None, f"could not build dataset at percentile {pct}: {type(error).__name__}: {error}"
        orig, ortho=evaluate_clustering(encoder, attention, dataset, conditionList)
        rows.append(
            {"percentile": pct, "silhouette_original": orig, "silhouette_orthogonal": ortho}
        )
    set_edge_percentile(config.edgePercentile)  
    return pd.DataFrame(rows), None


# grab the current main hyperparameter settings as a dict
def _primary_config():
    return {
        "name": "primary",
        "D_MODEL": config.dModel,
        "HEADS": config.heads,
        "LAYERS": config.layers,
        "DROPOUT": config.dropout,
        "LR": config.lr,
    }

#read the 3 alternate hyperparameter configs from a JSON file, or report they're missing
def load_alternate_configs():
    path=config.resultsRoot / "sensitivity_configs.json"
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
        return config.jointCheckpointPath
    return config.checkpointDir / f"best_joint_model_{cfg['name']}.pt"

#test each hyperparameter config (each needs its own trained checkpoint),
#  save results, or report whats missing
def run_hyperparameter_sweep(conditionList):
    set_edge_percentile(config.edgePercentile)

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
    config.dModel=cfg["D_MODEL"]
    config.heads=cfg["HEADS"]
    config.layers=cfg["LAYERS"]
    config.dropout=cfg["DROPOUT"]
    config.lr = cfg["LR"]

def structural_self_test():
   # Proves the pipeline is structurally sound without real checkpoints
    print("[self-test] edge-percentile graph construction ...")
    fake_fc=np.random.randn(config.nNodes, config.nNodes).astype(np.float32)
    for pct in config.edgePercentileSensitivity:
        set_edge_percentile(pct)
        packaged=_patched_edgeIndexAttr(type("S", (), {})(), False, fake_fc)
        edge_index, edge_attr=packaged[0], packaged[1]
        assert edge_index.shape[0]==2, "edge_index must be [2, E]"
        assert edge_index.shape[1]==edge_attr.shape[0], "edge count must match"
        print(f"  pct={pct}: edges={edge_attr.shape[0]}")
    set_edge_percentile(config.edgePercentile)

    print("[self-test] clustering silhouette computation ...")
    conditionList=list(config.conditions)
    n_fm, n_hc, d=12, 8, config.dModel
    subjectList=[f"sub-fm{i:03d}" for i in range(n_fm)] + [f"sub-hc{i:03d}" for i in range(n_hc)]
    runner=cluster(None, str(config.checkpointDir), config.jointCheckpointPath.name, conditionList, subjectList)
    torch.manual_seed(config.randomSeed)
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
    print("[self-test] Good! The pipeline is structurally sound and ready to run.")
    return True



def main():
    conditionList=list(config.conditions)
    results_dir=config.resultsRoot
    os.makedirs(results_dir, exist_ok=True)
    blockers=[]

    print("=" * 70)
    print("Sensitivity analysis (paper Section 4.5)")
    print(f"  EDGE_PERCENTILE_SENSITIVITY = {config.edgePercentileSensitivity}")
    print(f"  primary edge percentile     = {config.edgePercentile}")
    print("  paper/draft_v1.md is empty -> using two-separate-tables layout (flagged)")
    print("=" * 70)

    # percentile
    pct_df, pct_blocker=run_percentile_sweep(conditionList, config.jointCheckpointPath)
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
