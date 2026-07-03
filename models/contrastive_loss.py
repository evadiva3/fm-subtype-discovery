import torch  
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    def __init__(self,temperature=0.5):
        super().__init__()
        self.temperature=temperature
    def forward(self,ag1,ag2):
        a=F.normalize(ag1,dim=1)
        b=F.normalize(ag2,dim=1)
        c=torch.cat([a,b],dim=0)
        sim=torch.mm(c,c.T)/self.temperature
        sim.fill_diagonal_(-1e9)
        n=ag1.shape[0]
        labels=torch.cat([torch.arange(n,2*n),torch.arange(n)]).to(ag1.device)  
        loss=F.cross_entropy(sim,labels)
        return loss


        
