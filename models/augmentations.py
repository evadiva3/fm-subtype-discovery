import torch
from config import config
class graph_augmentor:
    def __init__(self,mask_rate=None,noise_std=None):
        self.mask_rate=mask_rate if mask_rate is not None else config.maskRate;self.noise_std=noise_std if noise_std is not None else config.noiseStd
    def node_masking(self,data):
        data=data.clone();n=data.x.shape[0];m=torch.bernoulli(torch.full((n,),self.mask_rate).to(data.x.device));data.x[m.bool()]=0;return data
    def edge_noise(self,data):
        data=data.clone();n=torch.randn_like(data.edge_attr)*self.noise_std;data.edge_attr=data.edge_attr+n;return data
    def _one_view(self,data):
        v=data.clone();b=torch.rand(1).item()<config.maskApplyProb;be=torch.rand(1).item()<config.noiseApplyProb
        if not b and(not be):
            if torch.rand(1).item()<0.5:v=self.node_masking(v)
            else:v=self.edge_noise(v)
        if b:v=self.node_masking(v)
        if be:v=self.edge_noise(v)
        return v
    def augment(self,data):return(self._one_view(data),self._one_view(data))
