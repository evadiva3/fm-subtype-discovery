import sys
import math
from pathlib import Path
import pandas as pd
import torch
import optuna
_R=Path(__file__).resolve().parents[2]
for _p in (Path(__file__).resolve().parent,_R,_R/"src",_R/"models"):
    if str(_p) not in sys.path:
        sys.path.insert(0,str(_p))
from config import config
from gnn_encoder import GNNEncoder
from contrastive_loss import NTXentLoss
from augmentations import graph_augmentor
from dataset_mdd import mddDataset
from train_mdd import _views
from torch.utils.data import DataLoader, random_split

class W(torch.utils.data.Dataset):
    def __init__(self,sd):
        self.sd=sd
    def __len__(self):
        return len(self.sd)
    def __getitem__(self,i):
        return self.sd[i]
DS=None
WD=None
def _valloss(enc,loss,tr,vl,aug,dev,epochs,trial):
    opt=torch.optim.AdamW(enc.parameters(),lr=config.lr,weight_decay=config.weightDecay)
    warm=min(max(1,int(epochs*config.warmupFraction)),epochs)
    def sch_fn(e):
        if e<warm:
            return (e+1)/warm
        p=(e-warm)/max(1,epochs-warm)
        return 0.5*(1.0+math.cos(math.pi*p))
    sch=torch.optim.lr_scheduler.LambdaLR(opt,lr_lambda=sch_fn)
    best=float("inf")
    for e in range(epochs):
        enc.train()
        for bt in tr:
            b1,b2=_views(bt,aug,dev)
            z1=enc(b1)
            z2=enc(b2)
            opt.zero_grad()
            l=loss(z1,z2)
            l.backward()
            opt.step()
        sch.step()
        enc.eval()
        v=0
        nb=0
        with torch.no_grad():
            for bt in vl:
                b1,b2=_views(bt,aug,dev)
                v+=loss(enc(b1),enc(b2)).item()
                nb+=1
        avl=v/max(nb,1)
        best=min(best,avl)
        trial.report(avl,e)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return best

def objective(trial):
    config.dModel=trial.suggest_int("dModel",16,128)
    config.heads=trial.suggest_int("heads",2,8)
    config.output=trial.suggest_int("output",8,64)
    config.layers=trial.suggest_int("layers",2,4)
    config.dropout=trial.suggest_float("dropout",0.0,0.3)
    config.lr=trial.suggest_float("lr",1e-5,3e-3,log=True)
    config.weightDecay=trial.suggest_float("weightDecay",1e-4,1e-1,log=True)
    config.maskRate=trial.suggest_float("maskRate",0.05,0.25)
    config.ntXentTemp=trial.suggest_float("ntXentTemp",0.2,1.0)
    bs=trial.suggest_int("batchSize",8,23)
    torch.manual_seed(config.randomSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.randomSeed)
    enc=GNNEncoder().to(config.device)
    loss=NTXentLoss()
    aug=graph_augmentor()
    gen=torch.Generator().manual_seed(config.randomSeed)
    n=len(WD)
    nv=int(n*config.valFractionForTune)
    nt=n-nv
    tr,vl=random_split(WD,[nt,nv],generator=gen)
    DS.normalize(tr.indices)
    tl=DataLoader(tr,batch_size=bs,shuffle=True,collate_fn=lambda b:b,drop_last=True)
    vd=DataLoader(vl,batch_size=bs,shuffle=False,collate_fn=lambda b:b)
    return _valloss(enc,loss,tl,vd,aug,config.device,config.tuneEpochs,trial)

def main():
    global DS,WD
    n_trials=int(sys.argv[1]) if len(sys.argv)>1 else config.sampleNum
    DS=mddDataset()
    WD=W(DS.subjectData)
    sampler=optuna.samplers.TPESampler()
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5,n_warmup_steps=5)
    study=optuna.create_study(direction="minimize",sampler=sampler,pruner=pruner)
    study.optimize(objective,n_trials=n_trials)
    best=study.best_params
    Path(config.raySavePath).parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame([best]).to_json(config.raySavePath)
    print(f"best valLoss={study.best_value:.6f} -> {config.raySavePath}")
    print(best)

if __name__=="__main__":
    main()
