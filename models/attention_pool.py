import torch
import torch.nn as nn
import torch.nn.functional as F

class condition_attention_pool(nn.Module):
    def __init__(self,d_model=64,num_cons=7):
        super().__init__()
        self.d_model=d_model
        self.num_cons=num_cons
        self.attention = nn.Linear(d_model, 1)
    def forward(self, embed):
        a=self.attention(embed)
        b=a.squeeze(-1)
        c=F.softmax(b,dim=0)
        weight_sum=torch.einsum('i,ij->j',c,embed)
        return (weight_sum,c)
#Nikhil: looking at this, attention_pool uses nn.Linear which I am pretty sure is a linear model and has learnable featuree
# and weights so I could've sworn that using it as is and not training it at some point would mean it when I use
# it for cluster it would initialize with random weights which I don't think would be good. idk you can ask claude.wh