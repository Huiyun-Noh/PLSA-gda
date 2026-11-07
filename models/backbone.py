from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv


class GDABackbone(nn.Module):
    """Shared encoder + classifier for all GDA methods."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def encode(self, x: Tensor, edge_index: Tensor,
               edge_weight: Optional[Tensor] = None) -> Tensor:
        """3-layer GCN encoder -> hidden representation."""
        h = torch.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        h = torch.relu(self.conv2(h, edge_index, edge_weight=edge_weight))
        h = torch.relu(self.conv3(h, edge_index, edge_weight=edge_weight))
        return h

    def classify(self, h: Tensor) -> Tensor:
        """2-layer MLP classifier."""
        return self.mlp(h)

    def forward(self, x: Tensor, edge_index: Tensor,
                edge_weight: Optional[Tensor] = None) -> Tensor:
        return self.classify(self.encode(x, edge_index, edge_weight=edge_weight))