#changes made: joint replace two phase training bc per-grpah NTXENT 
# with a seperate varance maxiumized poool produced unconstrained 
# embeddings so joint makes it learns condition importance directly 
# making thr temo meaninful and siholute scores defensible

import torch
import os
import math
from torch_geometric.data import Batch
from config import config
from ray import tune, train;
class trainer():

    def __init__(self,model,loss,optimize,schedule,device,dire):
        self.model=model
        self.loss=loss
        self.optimize=optimize
        self.schedule=schedule
        self.device=device
        self.dire=dire
        self.best_val_loss=float('inf')
        os.makedirs(self.dire, exist_ok=True)

    def train_epoch(self, dataloader, augmentor):
        self.model.train()
        i=0
        total_loss=0
        for k in dataloader:
            k=k.to(self.device)
            view1,view2=augmentor.augment(k)
            self.optimize.zero_grad()
            a=self.model(view1)
            b=self.model(view2)
            loss=self.loss(a,b)
            loss.backward()
            self.optimize.step()
            total_loss+=loss.item()
            i+=1
        self.schedule.step()
        return(total_loss/i)
    
    def validate(self, dataloader, augmentor):
        self.model.eval()
        i=0
        total_loss=0
        with torch.no_grad():
            for k in dataloader:
                k=k.to(self.device)
                view1,view2=augmentor.augment(k)
                a=self.model(view1)
                b=self.model(view2)
                loss=self.loss(a,b)
                total_loss+=loss.item()
                i+=1
        return(total_loss/i)
    
    def fit(self, train_load, val_load, augmentor, epochs=None, patience=None):
        # remeber no hardcoding
        epochs=config.epochs if epochs is None else epochs
        patience=config.patience if patience is None else patience
        c=0
        path=os.path.join(self.dire, 'best_model.pt')
        for i in range(epochs):
            a=self.train_epoch(train_load,augmentor)
            b=self.validate(val_load,augmentor)
            if b<self.best_val_loss:
                self.best_val_loss=b
                torch.save(self.model.state_dict(), path)
                c=0
            else:
                c+=1
                if c>=patience:
                    print('model has stopped learning :(')
                    break;
            print(f"Epoch {i}: train={a:.4f} val={b:.4f}")
        self.load_best()
    def load_best(self):
        path=os.path.join(self.dire, 'best_model.pt')
        self.model.load_state_dict(torch.load(path, map_location=self.device))  

def joint_train(model,attention_pool,loss_fn,dataloader,val_dataloader,augmentor,device,save_dir,epochs=None,patience=None,lr=None,weight_decay=None, realData = None, tuneSave=None):
    epochs=config.epochs if epochs is None else epochs
    patience=config.patience if patience is None else patience
    lr=config.lr if lr is None else lr
    weight_decay=config.weightDecay if weight_decay is None else weight_decay
    os.makedirs(save_dir, exist_ok=True)
    if tuneSave is not None:
        checkpoint_path = os.path.join(config.tuneTrainSave,tuneSave+".pt");
    else:
        checkpoint_path=config.trainSave;
    optimizer=torch.optim.AdamW([
        {'params': model.parameters(), 'lr': lr, 'weight_decay': weight_decay},
        {
            'params': attention_pool.parameters(),
            'lr': lr * config.attentionLearningRateMultiplier,
            'weight_decay': weight_decay
        }
    ])
    warmup_epochs=min(max(1, int(epochs * config.warmupFraction)), epochs)

    def lr_schedule(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress=(epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)
    best_val_loss=float('inf')
    patience_counter=0
    train_losses=[]
    val_losses=[]
    bestScore=0.0
    for epoch in range(epochs):
        model.train()
        attention_pool.train()
        epoch_loss=0
        n_batches=0
        for subject_batch in dataloader:
            Z_i_list=[]
            Z_j_list=[]
            for subject in subject_batch:
                graphs=subject['graphs'] 
                view1=[]
                view2=[]
                for g in graphs:
                    g=g.to(device)
                    v1, v2=augmentor.augment(g)
                    view1.append(v1)
                    view2.append(v2)
                batch1=Batch.from_data_list(view1).to(device)
                batch2=Batch.from_data_list(view2).to(device)

                z1=model(batch1)
                z2=model(batch2)
                pooled_i, _, _ =attention_pool(z1)
                pooled_j, _, _ =attention_pool(z2)
                Z_i_list.append(pooled_i)
                Z_j_list.append(pooled_j)
            Z_i=torch.stack(Z_i_list)
            Z_j=torch.stack(Z_j_list)
            optimizer.zero_grad()
            loss=loss_fn(Z_i, Z_j)
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.item()
            n_batches+=1
        scheduler.step()
        avg_train_loss=epoch_loss/max(n_batches, 1)
        train_losses.append(avg_train_loss)
        model.eval()
        attention_pool.eval()
        val_loss=0
        batches=0
     
        with torch.no_grad():
            for subject_batch in val_dataloader:
                Z_i_list=[]
                Z_j_list=[]
                for subject in subject_batch:
                    graphs=subject['graphs']
                    view1=[]
                    view2=[]
                    for g in graphs:
                        g=g.to(device)
                        v1, v2=augmentor.augment(g)
                        view1.append(v1)
                        view2.append(v2)
                    batch1=Batch.from_data_list(view1).to(device)
                    batch2=Batch.from_data_list(view2).to(device)
                    z1=model(batch1)
                    z2=model(batch2)
                    pooled_i, _, _=attention_pool(z1)
                    pooled_j, _, _=attention_pool(z2)
                    Z_i_list.append(pooled_i)
                    Z_j_list.append(pooled_j)
                Z_i=torch.stack(Z_i_list)
                Z_j=torch.stack(Z_j_list)
                loss=loss_fn(Z_i, Z_j)
                val_loss+=loss.item()
                batches+=1
       
        avg_val_loss=val_loss/max(batches, 1)
        tune.report({"valLoss": avg_val_loss, "silhouetteScore": bestScore});
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch}: train={avg_train_loss:.4f} val={avg_val_loss:.4f} tau={attention_pool.tau.item():.4f}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if avg_val_loss<best_val_loss:
            best_val_loss=avg_val_loss
            torch.save({
                'model': model.state_dict(),
                'pool': attention_pool.state_dict()
            }, checkpoint_path)
            patience_counter=0
        else:
            patience_counter+=1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    checkpoint=torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    attention_pool.load_state_dict(checkpoint['pool'])
    if realData is not None:
        direct = tune.get_context().get_trial_dir();
        bestScore = intermedCluster(realData, model, attention_pool, direct);
        tune.report({"valLoss": avg_val_loss});
    return model, attention_pool, train_losses, val_losses
def intermedCluster(data, encodeOut, attentionOut, direct):
    from clustering import cluster;
    from torch.utils.data import DataLoader
    clustering = cluster(encodeOut, config.clusterOutput, config.conditions, config.subjectList);
    clustering.deploy(data);
    embeddings = clustering.setAttention(attentionOut);
    clustering._split_fm_hc();
    package = clustering.KMeansUse(skip_perm=False, skip_gap=True);
    trialSave = package[0];
    sizeOk = package[4];
    kSil = trialSave["k_selected_silhouette"].iloc[0];
    bestScore = trialSave.loc[trialSave["k"]==kSil,"silhouette_score"].iloc[0];
    permP = package[3];
    if not(sizeOk) or permP>=config.fdrAlpha:
        bestScore = 0.0;
    return bestScore;
if __name__ == "__main__":
    from gnn_encoder import GNNEncoder
    from contrastive_loss import NTXentLoss
    from attention_pool import condition_attention_pool
    from augmentations import graph_augmentor
    from dataset import datasetPreparation
    from torch.utils.data import DataLoader, random_split
    torch.manual_seed(config.randomSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.randomSeed)
    dataset=datasetPreparation()
    class GroupedWrapper(torch.utils.data.Dataset):
        def __init__(self, subject_data):
            self.subject_data=subject_data
        def __len__(self):
            return len(self.subject_data)
        def __getitem__(self, idx):
            return self.subject_data[idx]

    grouped_dataset=GroupedWrapper(dataset.subjectData)

    n_total=len(grouped_dataset)
    n_val=int(n_total*config.valFraction)
    n_train=n_total-n_val
    split_generator=torch.Generator().manual_seed(config.randomSeed)
    train_split, val_split=random_split(grouped_dataset, [n_train, n_val], generator=split_generator)

    train_loader=DataLoader(train_split, batch_size=config.batchSize, shuffle=True, collate_fn=lambda b: b, drop_last=True)
    val_loader=DataLoader(val_split, batch_size=config.batchSize, shuffle=False, collate_fn=lambda b: b)

    encoder=GNNEncoder().to(config.device)
    attention=condition_attention_pool().to(config.device);           
    loss_fn=NTXentLoss()                            
    augmentor=graph_augmentor()                     

    device=config.device                            

    model, attention, train_losses, val_losses=joint_train(
        encoder, attention, loss_fn, train_loader, val_loader,
        augmentor, device, config.checkpointDir
    )
