import json
import warnings
import torch
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap
from pathlib import Path
import sys
_ROOT=Path(__file__).resolve().parent.parent
for _p in(_ROOT,_ROOT/"src",_ROOT/"models"):
    if str(_p) not in sys.path:sys.path.insert(0,str(_p))
from torch_geometric.data import Batch
from dataset import datasetPreparation
from analysis.evaluate import cluster_evaluate
from config import config
class cluster:
    def __init__(self,GNNEncoder,directory,conditionList,subjectList):
        self.GNNEncoder=GNNEncoder;self.PTPath=directory;self.subjectDList=[[None for _ in range(0,len(conditionList))] for _ in range(0,len(subjectList))];self.conditionList=conditionList
        self.subjectList=subjectList
    def deploy(self,dataloader):
        self.subjectEmbeddings={};self.groupLabels={};self.GNNEncoder.eval();d=next(self.GNNEncoder.parameters()).device
        with torch.no_grad():
            for s in dataloader:
                b=Batch.from_data_list(s["graphs"]).to(d);e=self.GNNEncoder(b);sd=s["subject_id"];self.subjectEmbeddings[sd]=e;self.groupLabels[sd]=int(s["group_label"].view(-1)[0].item())
    def setAttention(self,attentionModel):
        self.attentionEmbeddings={};self.attentionWeights={};attentionModel.eval()
        with torch.no_grad():
            for s,e in self.subjectEmbeddings.items():
                o,w,t=attentionModel(e);self.attentionEmbeddings[s]=o;self.attentionWeights[s]=w
        self.tau=t.item();return self.attentionEmbeddings
    def _stack(self,embDict):
        i1=list(embDict.keys());t=torch.stack([embDict[i] for i in i1]);return(t,i1)
    def _split_fm_hc(self):
        self.fmEmbed={};self.hcEmbed={}
        for s,e in self.attentionEmbeddings.items():
            if self.groupLabels[s]==0:self.fmEmbed[s]=e
            elif self.groupLabels[s]==1:self.hcEmbed[s]=e
    def validate_hc_sep(self):
        i1=list(self.attentionEmbeddings.keys());e=torch.stack([self.attentionEmbeddings[i] for i in i1]).detach().cpu().numpy();l=np.array([self.groupLabels[i] for i in i1])
        e=e/(np.linalg.norm(e,axis=1,keepdims=True)+1e-08);self.hcSepSilh=silhouette_score(e,l);self.hcSepPermP=cluster_evaluate().perm(e,l,match_selection=False);return self.hcSepSilh
    def project_ortho(self,fmEmbeddings,hcEmbeddings):
        f=fmEmbeddings.mean(dim=0);h=hcEmbeddings.mean(dim=0);v=f-h;v=v/v.norm();p=fmEmbeddings@v;z=fmEmbeddings-torch.outer(p,v);self.hcC=h-h@v*v;return z
    def compute_centroid_distances(self,fmEmbeddings,labels,hcCentroid):
        d=[]
        for s in sorted(set(labels)):
            m=torch.tensor(labels==s);sd=fmEmbeddings[m].mean(dim=0);d.append((sd-hcCentroid).norm().item())
        return np.array(d)
    def KMeansUse(self,takeTensor=None,subjectIds=None,skip_perm=False,skip_gap=False):
        if takeTensor is None:
            self._split_fm_hc();takeTensor,subjectIds=self._stack(self.fmEmbed)
        takeTensor=takeTensor.detach().cpu().numpy();takeTensor=takeTensor/(np.linalg.norm(takeTensor,axis=1,keepdims=True)+1e-08);n=takeTensor.shape[0]
        me=max(config.minClusterSizeFloor,round(config.minClusterSizeFraction*n));te=[];l=[];ms=[]
        for k in config.kmeansKRange:
            m=KMeans(n_clusters=k,n_init=config.kmeansNInit,random_state=config.randomSeed);ls=m.fit_predict(takeTensor);ms.append(np.bincount(ls).min());s=silhouette_score(takeTensor,ls);te.append(s)
            l.append(ls)
        bt=[None for _ in range(0,len(l))]
        for i in range(0,len(l)):
            if ms[i]<me:bt[i]=False
            else:bt[i]=True
        pe=[]
        for i in range(0,len(bt)):
            if bt[i]==True:pe.append(te[i])
        p=[ix for ix in range(0,len(bt)) if bt[ix]]
        if not p:
            b=te.index(max(te));sk=False
        else:
            b=max(p,key=lambda i:te[i]);sk=True
        bs=l[b];e=cluster_evaluate();kl=config.kmeansKRange[b]
        if skip_gap:
            g={kk:{"gap":np.nan,"s":np.nan} for kk in config.kmeansKRange};kp=np.nan;ky=np.nan
        else:
            g=e.gap_stat(takeTensor,k=config.kmeansKRange);kp=e.gap_k(g,config.kmeansKRange);ky=e.gap_k_at_boundary(g,config.kmeansKRange)
        if skip_perm:pp=np.nan
        else:pp=e.perm(takeTensor,bs)
        pn=[np.nan for _ in config.kmeansKRange];pn[b]=pp;t=pd.DataFrame({"Subject_Id":subjectIds,"Label":bs})
        te=pd.DataFrame({"k":config.kmeansKRange,"silhouette_score":te,"gap_stat":[g[kk]["gap"] for kk in config.kmeansKRange],"gap_se":[g[kk]["s"] for kk in config.kmeansKRange],"permutation_p":pn,"k_selected_silhouette":kl,"k_selected_gap":kp,"k_gap_at_boundary":ky,"min_cluster_size":ms,"min_cluster_size_required":me,"passes_size_guard":bt})
        return[te,t,bs,pp,sk]
    def UMAPPING(self,array):
        array=array.detach().cpu().numpy();c=umap.UMAP(n_neighbors=10,min_dist=0.1,random_state=config.randomSeed);return c.fit_transform(array)
    def saveAll(self,cWeights,embeddingsAtt,kScore,labelFK,coords,orthoLabels,orthoScores):
        p=config.clusterOutput;p.mkdir(parents=True,exist_ok=True);labelFK.to_csv(p/"K-Means-Labeling.csv",index=False);np.save(p/"Embeddings.npy",embeddingsAtt.detach().cpu().numpy())
        kScore.to_csv(p/"silhouette-scores.csv");np.save(p/"UMAP-COORDS.npy",coords);np.save(p/"attentionWeights.npy",cWeights.detach().cpu().numpy())
        with open(p/"hc_separation_silhouette.json","w")as f:json.dump({"hc_separation_silhouette":float(self.hcSepSilh),"permutation_p":float(self.hcSepPermP)},f)
        orthoLabels.to_csv(p/"orthogonal_labels.csv",index=False);orthoScores.to_csv(p/"orthogonal_silhouette_scores.csv");np.save(p/"centroid_distances.npy",self.centroidDistances)
        np.save(p/"centroid_distances_unprojected.npy",self.hcCUnprojDistances)
        with open(p/"tau_value.txt","w")as f:f.write(str(self.tau))
    def clusterEX(self,dataloader,attentionPooler):
        self.deploy(dataloader);self.setAttention(attentionPooler);self._split_fm_hc();self.validate_hc_sep();b=self.KMeansUse();self.fmClusterPermP=b[3]
        b[0]["significant_at_alpha"]=bool(b[3]<config.fdrAlpha);fr,f=self._stack(self.fmEmbed);hr,h=self._stack(self.hcEmbed);fd=self.project_ortho(fr,hr);o=self.KMeansUse(fd,f)
        self.orthoClusterPermP=o[3];o[0]["significant_at_alpha"]=bool(o[3]<config.fdrAlpha);self.centroidDistances=self.compute_centroid_distances(fd,o[2],self.hcC)
        self.hcCUnprojDistances=self.compute_centroid_distances(fr,b[2],hr.mean(dim=0));c=self.UMAPPING(fr);w=torch.stack([self.attentionWeights[i] for i in f]);self.saveAll(w,fr,b[0],b[1],c,o[1],o[0])
if __name__=="__main__":
    from gnn_encoder import GNNEncoder
    from models.attention_pool import condition_attention_pool
    conditionList=["Neutral - OBSERVAR","Negativo - OBSERVAR","Negativo - REDUCIR","Negativo - SUPRIMIR","Happy - OBSERVAR","Happy - SUPRIMIR","Happy - INCREMENTAR"]
    dataset=datasetPreparation(fm_only=False);dataList=dataset.subjectList;data=dataset.subjectData;attention=condition_attention_pool(d_model=config.dModel,num_cons=config.nConditions)
    encoder=GNNEncoder();checkpoint=torch.load(config.trainSave,map_location="cpu")
    if checkpoint.get("nodeMean") is not None:dataset.applyNormalization(checkpoint["nodeMean"],checkpoint["nodeStd"])
    else:warnings.warn("checkpoint has no saved normalization stats; falling back to all-subject statistics (train/inference mismatch)")
    encoder.load_state_dict(checkpoint["model"]);attention.load_state_dict(checkpoint["pool"]);runCluster=cluster(encoder,config.trainSave,conditionList,dataList);runCluster.clusterEX(data,attention)
