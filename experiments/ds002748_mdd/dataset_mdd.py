import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset
from sklearn.covariance import LedoitWolf
from nilearn.connectome import ConnectivityMeasure

_R=Path(__file__).resolve().parents[2]
for _p in (Path(__file__).resolve().parent,_R,_R/"src",_R/"models",_R/"preprocessing"):
    if str(_p) not in sys.path:
        sys.path.insert(0,str(_p))
from config import config
from subject_filter_mdd import get_included_subjects_mdd

COL_OK=True
COLUMN_MAP={
 "subject_id":"participant_id",
 "group":"group",
 "madrs":"MADRS",
 "zsrds":"Zung_SDS",
 "bdi":"BDI",
 "age":"age",
}
GRP_OK=True
MDD_VAL="depr"
HC_VAL="control"
MDD_TR=2.5
MDD_ROOT=_R.parent/"resting_state_dep_data"
PAT_OK=True
TS_PAT="{s}/{s}_rest_ts.npy"
PART_OK=True
PART_TSV=MDD_ROOT/"participants.tsv"

class mddDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.root=MDD_ROOT
        self.pf=pd.read_csv(PART_TSV,sep="\t")
        self.pf[COLUMN_MAP["subject_id"]]=self.pf[COLUMN_MAP["subject_id"]].astype(str)
        self.look=self.pf.set_index(COLUMN_MAP["subject_id"]).to_dict()
        self.subjectList=[]
        self.subjectData=[]
        self.rawX=None
        self.DataList=self.execute()

    def bands(self,x):
        f=np.fft.rfft(x,axis=0,norm="forward")
        f=np.abs(f)**2
        l=np.fft.rfftfreq(n=len(x),d=MDD_TR)
        ix=[np.where((l>=0.01)&(l<0.04)),np.where((l>=0.04)&(l<0.1)),np.where((l>=0.1)&(l<0.25))]
        return [float(np.sum(f[i[0]])) for i in ix]

    def nodes(self,ts):
        d=pd.DataFrame(ts)
        out=[]
        for i in range(0,len(d.columns)):
            m=d.iloc[:,i].mean()
            v=d.iloc[:,i].var(ddof=1)
            b=self.bands(d.iloc[:,i].to_numpy())
            out.append([m,v,b[0],b[1],b[2]])
        return torch.tensor(np.array(out),dtype=torch.float32)

    def fc(self,ts):
        lw=ConnectivityMeasure(kind="correlation",cov_estimator=LedoitWolf())
        r=lw.fit_transform([ts])[0]
        z=np.arctanh(r)
        np.fill_diagonal(z,0)
        z[np.isinf(z)]=0
        return z

    def edges(self,z):
        w=np.where(np.abs(z)>=np.percentile(np.abs(z),config.edgePercentile))
        ei=np.array([w[0],w[1]])
        ea=z[ei[0],ei[1]]
        return torch.tensor(ei,dtype=torch.long),torch.tensor(ea,dtype=torch.float32)

    def label(self,sid):
        g=self.look.get(COLUMN_MAP["group"],{}).get(sid,None)
        if g==MDD_VAL:
            return 0
        if g==HC_VAL:
            return 1
        return -1

    def execute(self):
        dl=[]
        sd=[]
        inc=set(get_included_subjects_mdd())
        for f in sorted(self.root.iterdir()):
            if not (f.is_dir() and f.name in inc):
                continue
            tp=self.root/TS_PAT.format(s=f.name)
            if not tp.exists():
                continue
            ts=np.load(tp)
            x=self.nodes(ts)
            z=self.fc(ts)
            ei,ea=self.edges(z)
            y=torch.tensor(self.label(f.name),dtype=torch.long)
            g=Data(x=x,y=y,edge_index=ei,edge_attr=ea)
            g.subjectID=f.name
            assert g.x.shape[0]==config.nNodes,f"{f.name} has {g.x.shape[0]} nodes, expected {config.nNodes}"
            self.subjectList.append(f.name)
            dl.append(g)
            sd.append({"subject_id":f.name,"graphs":[g],"group_label":torch.tensor([self.label(f.name)],dtype=torch.long)})
        self.DataList=dl
        self.subjectData=sd
        self.normalize()
        return dl

    def normalize(self,trIdx=None):
        if self.rawX is None:
            self.rawX=[g.x.clone() for g in self.DataList]
        for g,r in zip(self.DataList,self.rawX):
            g.x=r.clone()
            g.x[:,2:5]=torch.log1p(g.x[:,2:5])
        idx=range(len(self.DataList)) if trIdx is None else trIdx
        st=torch.stack([self.DataList[i].x for i in idx])
        mu=st.mean(dim=0)
        sig=st.std(dim=0)
        for g in self.DataList:
            g.x=(g.x-mu)/(sig+1e-8)

    def __len__(self):
        return len(self.DataList)

    def __getitem__(self,i):
        return self.DataList[i]
