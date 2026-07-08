import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config 

class graph_augmentor:
    def __init__(self, mask_rate=None, noise_std=None):
        self.mask_rate=mask_rate if mask_rate is not None else config.MASK_RATE
        self.noise_std=noise_std if noise_std is not None else config.NOISE_STD

    def node_masking(self, data):
        data=data.clone()
        n=data.x.shape[0]
        mask=torch.bernoulli(torch.full((n,), self.mask_rate).to(data.x.device))  
        data.x[mask.bool()]=0
        return data
    
    def edge_noise(self, data):
        data=data.clone()
        n=torch.randn_like(data.edge_attr)*self.noise_std
        data.edge_attr=data.edge_attr+n
        return data

    
    def augment(self, data):
        data=data.clone()
        data1=self.node_masking(data)
        data2=self.edge_noise(data)
        return (data1,data2)
