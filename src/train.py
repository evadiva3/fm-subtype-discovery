import torch
import os



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
    def fit(self, train_load, val_load, augmentor, epochs=200, patience=10):
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
        self.model.load_state_dict(torch.load(path, map_location=self.device))  # fix: map_location ensures checkpoint loads correctly regardless of device
    def train_attention(self, attention_pool, dataloader, epochs=200):
        self.model.eval()
        pool_opt=torch.optim.AdamW(attention_pool.parameters(), lr=1e-4)
        attention_pool.train()
        for i in range(epochs):
            total_loss=0
            for k in dataloader:
                k=k.to(self.device)
                pool_opt.zero_grad()
                with torch.no_grad():
                    a=self.model(k)
                pooled,weights=attention_pool(a)
                loss=-pooled.var()
                loss.backward()
                pool_opt.step()
                total_loss+=loss.item()
            print(f"Epoch {i}: train={total_loss/len(dataloader):.4f}")

if __name__ == "__main__":
    from gnn_encoder import GNNEncoder;
    from contrastive_loss import NTXentLoss;
    from torch_geometric.data import DataLoader;
    from dataset import datasetPreparation;
    from torch.optim import AdamW;
    from transformers import get_cosine_schedule_with_warmup;
    dataset = datasetPreparation();
    dataSetup = DataLoader(dataset, batch_size=8, shuffle = True);
    stepSize = len(dataSetup);
    epochs = 200;
    totalSteps = stepSize * epochs;
    partitionWarmupSteps = int(totalSteps * .10);
    encoder = GNNEncoder();
    lossing = NTXentLoss(temperature = 0.5);
    optimizer = AdamW(encoder.parameters(), lr=1e-4,weight_decay=1e-4);
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = partitionWarmupSteps, num_training_steps=totalSteps);
    deviceChoice = torch.device('cpu');
    if torch.cuda.is_available():
            deviceChoice = torch.device('cuda');
    training = trainer(encoder,lossing,optimizer,scheduler,deviceChoice,"results/checkpoints");


            
    

