import torch
import torch.nn as nn
import torch.nn.functional as F

class condition_attention_pool(nn.Module):
    def __init__(self,d_model=64,num_cons=7):
        super().__init__()
        self.d_model=d_model
        self.num_cons=num_cons
        self.tau=nn.Parameter(torch.ones(1))
        self.attention = nn.Linear(d_model, 1)
    def forward(self, embed):
        a=self.attention(embed)
        b=a.squeeze(-1)
        c=F.softmax(b/self.tau, dim=0)
        weight_sum=torch.einsum('i,ij->j',c,embed)
        return (weight_sum, c, self.tau)
#addressed