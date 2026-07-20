#experimental ignore for now


import sys
import os
import math
from pathlib import Path
import torch
from torch_geometric.data import Batch

_R=Path(__file__).resolve().parents[2]
for _p in (Path(__file__).resolve().parent,_R,_R/"src",_R/"models"):
    if str(_p) not in sys.path:
        sys.path.insert(0,str(_p))
from config import config

CK_NAME="bestMddModel.pt"

def _views(subs,aug,dev):
    a=[]
    b=[]
    for s in subs:
        g=s["graphs"][0].to(dev)
        v1,v2=aug.augment(g)
        a.append(v1)
        b.append(v2)
    return Batch.from_data_list(a).to(dev),Batch.from_data_list(b).to(dev)

def train_mdd(enc,loss,tr,vl,aug,dev,dire,epochs=None,patience=None,lr=None,wd=None):
    epochs=config.epochs if epochs is None else epochs
    patience=config.patience if patience is None else patience
    lr=config.lr if lr is None else lr
    wd=config.weightDecay if wd is None else wd
    os.makedirs(dire,exist_ok=True)
    ck=os.path.join(dire,CK_NAME)
    opt=torch.optim.AdamW(enc.parameters(),lr=lr,weight_decay=wd)
    warm=min(max(1,int(epochs*config.warmupFraction)),epochs)
    def sch_fn(e):
        if e<warm:
            return (e+1)/warm
        p=(e-warm)/max(1,epochs-warm)
        return 0.5*(1.0+math.cos(math.pi*p))
    sch=torch.optim.lr_scheduler.LambdaLR(opt,lr_lambda=sch_fn)
    best=float("inf")
    cnt=0
    trl=[]
    vll=[]
    for e in range(epochs):
        enc.train()
        el=0
        nb=0
        for bt in tr:
            b1,b2=_views(bt,aug,dev)
            z1=enc(b1)
            z2=enc(b2)
            opt.zero_grad()
            l=loss(z1,z2)
            l.backward()
            opt.step()
            el+=l.item()
            nb+=1
        sch.step()
        atl=el/max(nb,1)
        trl.append(atl)
        enc.eval()
        vl2=0
        vb=0
        with torch.no_grad():
            for bt in vl:
                b1,b2=_views(bt,aug,dev)
                z1=enc(b1)
                z2=enc(b2)
                vl2+=loss(z1,z2).item()
                vb+=1
        avl=vl2/max(vb,1)
        vll.append(avl)
        print(f"Epoch {e}: train={atl:.4f} val={avl:.4f}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if avl<best:
            best=avl
            torch.save({"enc":enc.state_dict()},ck)
            cnt=0
        else:
            cnt+=1
            if cnt>=patience:
                print(f"Early stopping at epoch {e}")
                break
    enc.load_state_dict(torch.load(ck,map_location=dev)["enc"])
    return enc,trl,vll

if __name__=="__main__":
    from gnn_encoder import GNNEncoder
    from contrastive_loss import NTXentLoss
    from augmentations import graph_augmentor
    from dataset_mdd import mddDataset
    from torch.utils.data import DataLoader,random_split
    torch.manual_seed(config.randomSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.randomSeed)
    ds=mddDataset()
    class W(torch.utils.data.Dataset):
        def __init__(self,sd):
            self.sd=sd
        def __len__(self):
            return len(self.sd)
        def __getitem__(self,i):
            return self.sd[i]
    gd=W(ds.subjectData)
    n=len(gd)
    nv=int(n*config.valFraction)
    nt=n-nv
    gen=torch.Generator().manual_seed(config.randomSeed)
    ts,vs=random_split(gd,[nt,nv],generator=gen)
    ds.normalize(ts.indices)
    ebs=max(2,nt//4)
    tl=DataLoader(ts,batch_size=ebs,shuffle=True,collate_fn=lambda b:b,drop_last=True)
    vld=DataLoader(vs,batch_size=ebs,shuffle=False,collate_fn=lambda b:b)
    enc=GNNEncoder().to(config.device)
    loss=NTXentLoss()
    aug=graph_augmentor()
    train_mdd(enc,loss,tl,vld,aug,config.device,config.checkpointDir)
