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
for _p in (Path(__file__).resolve().parent,_R,_R/'src',_R/'models',_R/'preprocessing'):
    if str(_p)not in sys.path:
        sys.path.insert(0,str(_p))
from config import config
from subject_filter_mdd import get_included_subjects_mdd
COL_OK=True
COLUMN_MAP= {'subject_id':'participant_id','group':'group','madrs':'MADRS','zsrds':'Zung_SDS','bdi':'BDI','age':'age'}
GRP_OK=True
MDD_VAL='depr'
HC_VAL='control'
MDD_TR=2.5
MDD_ROOT=_R.parent/'resting_state_dep_data'
PAT_OK=True
TS_PAT='{s}/{s}_rest_ts.npy'
PART_OK=True
PART_TSV=MDD_ROOT/'participants.tsv'

def load_mdd_params():
    B=Path(config.mddRaySavePath)
    if not B.exists():
        return False
    C=pd.read_json(B)
    for A in ('dModel','heads','output','layers'):
        setattr(config,A,int(C.at[0,A]))
    for A in ('dropout','lr','weightDecay','maskRate','ntXentTemp'):
        setattr(config,A,float(C.at[0,A]))
    return True

class mddDataset(Dataset):

    def __init__(self):
        super().__init__()
        self.root=MDD_ROOT
        self.pf=pd.read_csv(PART_TSV,sep='\t')
        self.pf[COLUMN_MAP['subject_id']]=self.pf[COLUMN_MAP['subject_id']].astype(str)
        self.look=self.pf.set_index(COLUMN_MAP['subject_id']).to_dict()
        self.subjectList= []
        self.subjectData= []
        self.rawX=None
        self.mu=None
        self.sig=None
        self.DataList=self.execute()

    def bands(self,x):
        B=np.fft.rfft(x,axis=0,norm='forward')
        B=np.abs(B)**2
        A=np.fft.rfftfreq(n=len(x),d=MDD_TR)
        C= [np.where((A>=0.01)& (A<0.04)),np.where((A>=0.04)& (A<0.1)),np.where((A>=0.1)& (A<0.25))]
        return [float(np.sum(B[i[0]]))for i in C]

    def nodes(self,ts):
        A=pd.DataFrame(ts)
        C= []
        for i in range(0,len(A.columns)):
            D=A.iloc[:,i].mean()
            E=A.iloc[:,i].var(ddof=1)
            B=self.bands(A.iloc[:,i].to_numpy())
            C.append([D,E,B[0],B[1],B[2]])
        return torch.tensor(np.array(C),dtype=torch.float32)

    def fc(self,ts):
        A=ConnectivityMeasure(kind='correlation',cov_estimator=LedoitWolf())
        B=A.fit_transform([ts])[0]
        z=np.arctanh(B)
        np.fill_diagonal(z,0)
        z[np.isinf(z)]=0
        return z

    def edges(self,z):
        B=np.where(np.abs(z)>=np.percentile(np.abs(z),config.edgePercentile))
        A=np.array([B[0],B[1]])
        C=z[A[0],A[1]]
        return (torch.tensor(A,dtype=torch.long),torch.tensor(C,dtype=torch.float32))

    def label(self,sid):
        A=self.look.get(COLUMN_MAP['group'], {}).get(sid,None)
        if A==MDD_VAL:
            return 0
        if A==HC_VAL:
            return 1
        return-1

    def execute(self):
        C= []
        D= []
        F=set(get_included_subjects_mdd())
        for A in sorted(self.root.iterdir()):
            if not (A.is_dir()and A.name in F):
                continue
            E=self.root/TS_PAT.format(s=A.name)
            if not E.exists():
                continue
            ts=np.load(E)
            x=self.nodes(ts)
            z=self.fc(ts)
            G,H=self.edges(z)
            I=torch.tensor(self.label(A.name),dtype=torch.long)
            B=Data(x=x,y=I,edge_index=G,edge_attr=H)
            B.subjectID=A.name
            assert B.x.shape[0]==config.nNodes, f'{A.name} has {B.x.shape[0]} nodes, expected {config.nNodes}'
            self.subjectList.append(A.name)
            C.append(B)
            D.append({'subject_id':A.name,'graphs': [B],'group_label':torch.tensor([self.label(A.name)],dtype=torch.long)})
        self.DataList=C
        self.subjectData=D
        self.normalize()
        return C

    def normalize(self,trIdx=None):
        if self.rawX is None:
            self.rawX= [A.x.clone()for A in self.DataList]
        for A,C in zip(self.DataList,self.rawX):
            A.x=C.clone()
            A.x[:,2:5]=torch.log1p(A.x[:,2:5])
        D=range(len(self.DataList))if trIdx is None else trIdx
        B=torch.stack([self.DataList[i].x for i in D])
        mu=B.mean(dim=0)
        sig=B.std(dim=0)
        self.mu=mu
        self.sig=sig
        for A in self.DataList:
            A.x= (A.x-mu)/ (sig+1e-08)

    def applyNormalization(self,mu,sig):
        if self.rawX is None:
            self.rawX= [A.x.clone()for A in self.DataList]
        for A,B in zip(self.DataList,self.rawX):
            A.x=B.clone()
            A.x[:,2:5]=torch.log1p(A.x[:,2:5])
            A.x= (A.x-mu)/ (sig+1e-08)
        self.mu=mu
        self.sig=sig

    def __len__(self):
        return len(self.DataList)

    def __getitem__(self,i):
        return self.DataList[i]
