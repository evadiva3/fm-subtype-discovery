import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config  

class condition_attention_pool(nn.Module):

    def __init__(self,d_model=None,num_cons=None):
        super().__init__()
        self.d_model=d_model if d_model is not None else config.dModel
        self.num_cons=num_cons if num_cons is not None else config.nConditions
        self.tau=nn.Parameter(torch.ones(1))
        self.attention=nn.Linear(self.d_model, 1) 

    def forward(self, embed):
        a=self.attention(embed)
        b=a.squeeze(-1)
        c=F.softmax(b/self.tau, dim=0)
        weight_sum=torch.einsum('i,ij->j',c,embed)
        return (weight_sum, c, self.tau)
#addressed