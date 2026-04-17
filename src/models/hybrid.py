import torch
import torch.nn as nn
import torch.nn.functional as F
from .gcn import GCN


class HybridModel(nn.Module):
    def __init__(self, input_dim):
        super(HybridModel, self).__init__()

        # MLP branch
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

        # GNN branch
        self.gcn = GCN(input_dim, 64, 2)

    def forward(self, x, edge_index):
        mlp_out = self.mlp(x)
        gcn_out = self.gcn(x, edge_index)

        # Combine both
        out = mlp_out + gcn_out

        # IMPORTANT FIX: return log probabilities
        return F.log_softmax(out, dim=1)