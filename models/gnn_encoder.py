import torch;
import torch.nn as nn;
from torch_geometric.nn import GATv2Conv, global_mean_pool;
from torch_geometric.data import Batch;
from torch_geometric.nn import LayerNorm;
from torch.nn import ELU;
from config import config;
class GNNEncoder(nn.Module):
    def __init__(self):
        super().__init__();
        self.convList = nn.ModuleList([conv for convList in [[self.GATv2ConvInstance(5, config.OUTPUT, config.OUTPUT, config.HEADS, True, 1, True, config.DROPOUT)], [self.GATv2ConvInstance(0, config.OUTPUT, config.OUTPUT, config.HEADS, True, 1, False, config.DROPOUT) for _ in range(0,max(0,config.LAYERS-2))], [self.GATv2ConvInstance(0,config.OUTPUT, config.D_MODEL, config.HEADS, False, 1, False, config.DROPOUT)]] for conv in convList]);
        self.layerNormalList = nn.ModuleList([self.layerNormal(config.HEADS, config.OUTPUT, 'node') for _ in range(0,config.LAYERS-1)]);
        self.elu = nn.ELU();
    def GATv2ConvInstance(self, inChannels:int, outChannels:int, outLastChannels:int, head:int, concat:bool, edgeDim:int, isFirst:bool, dropout:float):    
        if(isFirst):
            return GATv2Conv(in_channels=inChannels, out_channels = outChannels, heads=head, concat = concat, edge_dim=edgeDim, dropout = dropout);
        else:
            return GATv2Conv(in_channels=head*outChannels, out_channels = outLastChannels, heads=head, concat=concat, edge_dim=edgeDim, dropout=dropout);
    def layerNormal(self, head:int, out:int, mode:str):
        return LayerNorm(in_channels=head*out, mode=mode)
    def forward(self, data: Batch):
        data.edge_attr = data.edge_attr.unsqueeze(-1);
        for i in range(0,max(0,config.LAYERS-1)): 
                data.x = self.convList[i](data.x, data.edge_index, data.edge_attr);
                data.x = self.layerNormalList[i](data.x, data.batch);
                data.x = self.elu(data.x);
        data.x = self.convList[config.LAYERS-1](data.x, data.edge_index, data.edge_attr);
        out = global_mean_pool(data.x, data.batch);
        return out;