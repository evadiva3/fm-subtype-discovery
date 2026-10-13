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
        self.input_norm=nn.LayerNorm(self.d_model)
        self.attention=nn.Linear(self.d_model,1)

    def forward(self,embed):
        normalized_embed=self.input_norm(embed)
        a=self.attention(normalized_embed)
        b=a.squeeze(-1)
        tau=F.softplus(self.tau)
        c=F.softmax(b/tau,dim=0)
        weight_sum=torch.einsum("i,ij->j",c,embed)
        return (weight_sum,c,tau)
