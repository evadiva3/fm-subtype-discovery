#experimental ignore for now


import sys
import json
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Batch

_R=Path(__file__).resolve().parents[2]
for _p in (Path(__file__).resolve().parent,_R,_R/"src",_R/"models",_R/"analysis"):
    if str(_p) not in sys.path:
        sys.path.insert(0,str(_p))
from config import config
from clustering import cluster
import bootstrap_stability as bs

OUT=Path(config.resultsRoot)/"mdd"

class mddCluster(cluster):
    def deploy(self,subs):
        self.subjectEmbeddings={}
        self.groupLabels={}
        self.GNNEncoder.eval()
        with torch.no_grad():
            for s in subs:
                b=Batch.from_data_list(s["graphs"])
                e=self.GNNEncoder(b)
                sid=s["subject_id"]
                self.subjectEmbeddings[sid]=e.view(-1)
                self.groupLabels[sid]=int(s["group_label"].view(-1)[0].item())
        self.attentionEmbeddings=self.subjectEmbeddings

    def run(self,subs):
        self.deploy(subs)
        self._split_fm_hc()
        t,ids=self._stack(self.fmEmbed)
        tr=self.KMeansUse(t,ids)
        emb=t.detach().cpu().numpy()
        ar=bs.run_bootstrap(self,emb,list(ids))
        return tr,ids,emb,ar

    def save(self,tr,ids,emb,ar):
        OUT.mkdir(parents=True,exist_ok=True)
        tr[0].to_csv(OUT/"silhouette-scores.csv")
        tr[1].to_csv(OUT/"K-Means-Labeling.csv",index=False)
        np.save(OUT/"Embeddings.npy",emb)
        np.save(OUT/"bootstrap_ari.npy",ar)
        with open(OUT/"bootstrap_summary.json","w") as f:
            json.dump({"ari_mean":float(np.mean(ar)),"ari_std":float(np.std(ar)),"ari_min":float(np.min(ar)),"ari_max":float(np.max(ar))},f)

if __name__=="__main__":
    from gnn_encoder import GNNEncoder
    from dataset_mdd import mddDataset
    ds=mddDataset()
    ck=torch.load(config.checkpointDir/"bestMddModel.pt",map_location="cpu")
    enc=GNNEncoder()
    enc.load_state_dict(ck["enc"])
    r=mddCluster(enc,str(config.checkpointDir),[],ds.subjectList)
    tr,ids,emb,ar=r.run(ds.subjectData)
    r.save(tr,ids,emb,ar)
    print(tr[0].to_string())
    print(f"bootstrap ARI mean={np.mean(ar):.4f} range=[{np.min(ar):.4f},{np.max(ar):.4f}]")
