import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config 

class graph_augmentor:
    def __init__(self, mask_rate=None, noise_std=None):
        self.mask_rate=mask_rate if mask_rate is not None else config.maskRate
        self.noise_std=noise_std if noise_std is not None else config.noiseStd

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

    
    def _one_view(self, data):
        v=data.clone()
        boolMask = torch.rand(1).item()<config.maskApplyProb;
        boolNoise = torch.rand(1).item()<config.noiseApplyProb;
        if(not boolMask and not boolNoise):
            if torch.rand(1).item() < 0.5:
                v = self.node_masking(v);
            else:
                v = self.edge_noise(v);
        if boolMask:
            v=self.node_masking(v)
        if boolNoise:
            v=self.edge_noise(v)
        return v

    def augment(self, data):
        return (self._one_view(data), self._one_view(data))
