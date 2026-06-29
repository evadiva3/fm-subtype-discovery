import torch;
import torch.nn as nn;
from torch_geometric.nn import GATv2Conv, global_mean_pool;
from torch_geometric.data import Batch;
from torch_geometric.nn import LayerNorm;
from torch.nn import ELU;
class GNNEncoder(nn.Module):
    def __init__(self):
        super().__init__();
        self.conv1 = GATv2Conv(in_channels=5, out_channels=32, heads=4, concat=True, edge_dim=1);
        self.conv2 = GATv2Conv(in_channels=128, out_channels=32, heads=4, concat=True, edge_dim=1);
        self.conv3 = GATv2Conv(in_channels=128, out_channels=64, heads=4, concat=False, edge_dim=1);g
        self.layerNorm1 = LayerNorm(in_channels=128, mode='node');
        self.layerNorm2 = LayerNorm(in_channels=128, mode = 'node');
        self.elu = nn.ELU();
    def forward(self, data: Batch):
        data.edge_attr = data.edge_attr.unsqueeze(-1);
        data.x = self.conv1(data.x, data.edge_index, data.edge_attr);
        data.x = self.layerNorm1(data.x, data.batch);
        data.x = self.elu(data.x);
        data.x = self.conv2(data.x, data.edge_index, data.edge_attr);
        data.x = self.layerNorm2(data.x, data.batch);
        data.x = self.elu(data.x);
        data.x = self.conv3(data.x, data.edge_index, data.edge_attr);
        out = global_mean_pool(data.x, data.batch);
        return out;