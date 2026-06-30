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
        total_loss = 0
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


            
    

