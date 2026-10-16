import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT, _ROOT /"models"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
import torch
import os
import math
from torch_geometric.data import Batch
from config import config
from ray import tune
def joint_train(model,attention_pool,loss_fn,dataloader,val_dataloader,augmentor,device,save_dir,epochs=None,patience=None,lr=None,weight_decay=None,tuneSave=None,guardPrimary=False,normStats=None):
    epochs=config.epochs if epochs is None else epochs;patience=config.patience if patience is None else patience;lr=config.lr if lr is None else lr
    weight_decay=config.weightDecay if weight_decay is None else weight_decay;os.makedirs(save_dir,exist_ok=True)
    if tuneSave is not None:
        os.makedirs(config.tuneTrainSave,exist_ok=True);ch=os.path.join(config.tuneTrainSave,tuneSave+".pt")
    else:ch=os.path.join(str(save_dir),os.path.basename(config.trainSave))
    pm=os.path.abspath(config.trainSave)
    if guardPrimary and os.path.abspath(ch)==pm:raise RuntimeError(f"retrain refused: would overwrite primary checkpoint {pm}")
    o=torch.optim.AdamW([{"params":model.parameters(),"lr":lr,"weight_decay":weight_decay},{"params":attention_pool.parameters(),"lr":lr*config.attentionLearningRateMultiplier,"weight_decay":weight_decay}])
    warmup_epochs=min(max(1,int(epochs*config.warmupFraction)),epochs)
    def lr_schedule(epoch):
        if epoch<warmup_epochs:return(epoch+1)/warmup_epochs
        p=(epoch-warmup_epochs)/max(1,epochs-warmup_epochs);return 0.5*(1.0+math.cos(math.pi*p))
    s=torch.optim.lr_scheduler.LambdaLR(o,lr_lambda=lr_schedule);b1=float("inf");p=0;t=[];vs=[]
    for e in range(epochs):
        model.train();attention_pool.train();es=0;n=0
        for sh in dataloader:
            zt=[];z3=[]
            for st in sh:
                gs=st["graphs"];v3=[];v4=[]
                for g in gs:
                    g=g.to(device);v1,v2=augmentor.augment(g);v3.append(v1);v4.append(v2)
                b=Batch.from_data_list(v3).to(device);b2=Batch.from_data_list(v4).to(device);z1=model(b);z2=model(b2);pi,_,_=attention_pool(z1);pj,_,_=attention_pool(z2);zt.append(pi);z3.append(pj)
            z=torch.stack(zt);zj=torch.stack(z3);o.zero_grad();l=loss_fn(z,zj);l.backward();o.step();es+=l.item();n+=1
        s.step();a=es/max(n,1);t.append(a);model.eval();attention_pool.eval();v=0;bs=0
        with torch.no_grad():
            for sh in val_dataloader:
                zt=[];z3=[]
                for st in sh:
                    gs=st["graphs"];v3=[];v4=[]
                    for g in gs:
                        g=g.to(device);v1,v2=augmentor.augment(g);v3.append(v1);v4.append(v2)
                    b=Batch.from_data_list(v3).to(device);b2=Batch.from_data_list(v4).to(device);z1=model(b);z2=model(b2);pi,_,_=attention_pool(z1);pj,_,_=attention_pool(z2);zt.append(pi)
                    z3.append(pj)
                z=torch.stack(zt);zj=torch.stack(z3);l=loss_fn(z,zj);v+=l.item();bs+=1
        a1=v/max(bs,1);tune.report({"valLoss":a1});vs.append(a1);print(f"Epoch {e}: train={a:.4f} val={a1:.4f} tau={attention_pool.tau.item():.4f}")
        if torch.cuda.is_available():torch.cuda.empty_cache()
        if a1<b1:
            b1=a1
            torch.save({"model":model.state_dict(),"pool":attention_pool.state_dict(),"nodeMean":None if normStats is None else normStats[0],"nodeStd":None if normStats is None else normStats[1]},ch)
            p=0
        else:
            p+=1
            if p>=patience:
                print(f"Early stopping at epoch {e}");break
    c=torch.load(ch,map_location=device);model.load_state_dict(c["model"]);attention_pool.load_state_dict(c["pool"]);return(model,attention_pool,t,vs)
if __name__=="__main__":
    from gnn_encoder import GNNEncoder
    from contrastive_loss import NTXentLoss
    from attention_pool import condition_attention_pool
    from augmentations import graph_augmentor
    from dataset import datasetPreparation
    from torch.utils.data import DataLoader,random_split
    torch.manual_seed(config.randomSeed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(config.randomSeed)
    dataset=datasetPreparation()
    class GroupedWrapper(torch.utils.data.Dataset):
        def __init__(self,subject_data):self.subject_data=subject_data
        def __len__(self):return len(self.subject_data)
        def __getitem__(self,idx):return self.subject_data[idx]
    grouped_dataset=GroupedWrapper(dataset.subjectData);n_total=len(grouped_dataset);n_val=int(n_total*config.valFraction);n_train=n_total-n_val
    split_generator=torch.Generator().manual_seed(config.randomSeed);train_split,val_split=random_split(grouped_dataset,[n_train,n_val],generator=split_generator)
    dataset.normalizeData(train_split.indices);normStats=(dataset.nodeMean,dataset.nodeStd)
    train_loader=DataLoader(train_split,batch_size=config.batchSize,shuffle=True,collate_fn=lambda b:b,drop_last=True)
    val_loader=DataLoader(val_split,batch_size=config.batchSize,shuffle=False,collate_fn=lambda b:b);encoder=GNNEncoder().to(config.device);attention=condition_attention_pool().to(config.device)
    loss_fn=NTXentLoss();augmentor=graph_augmentor();device=config.device
    model,attention,train_losses,val_losses=joint_train(encoder,attention,loss_fn,train_loader,val_loader,augmentor,device,config.checkpointDir,normStats=normStats)
