import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class SimpleMLP(nn.Module):

    def __init__(self, input_dim, n_classes=2, depth=3, hidden1=64, hidden2=32):
        super().__init__()
        assert depth in (1, 2, 3, 4)
        if depth == 1:
            self.net = nn.Sequential(
                nn.Linear(input_dim, n_classes),
            )
        elif depth == 2:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden1),
                nn.ReLU(),
                nn.Linear(hidden1, n_classes),
            )
        elif depth == 3:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden1),
                nn.ReLU(),
                nn.Linear(hidden1, hidden2),
                nn.ReLU(),
                nn.Linear(hidden2, n_classes),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden1),
                nn.ReLU(),
                nn.Linear(hidden1, hidden1),
                nn.ReLU(),
                nn.Linear(hidden1, hidden2),
                nn.ReLU(),
                nn.Linear(hidden2, n_classes),
            )

    def forward(self, x):
        return self.net(x)


class TinyGCN(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.head  = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_weight=None, **kwargs):
        h = self.conv1(x, edge_index); h = torch.relu(h)
        h = self.conv2(h, edge_index); h = torch.relu(h)
        return self.head(h)


class GCNwithMLP(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x, edge_index, edge_weight=None):
        h = self.conv1(x, edge_index, edge_weight=edge_weight)
        h = torch.relu(h)
        h = self.conv2(h, edge_index, edge_weight=edge_weight)
        h = torch.relu(h)
        h = self.conv3(h, edge_index, edge_weight=edge_weight)
        h = torch.relu(h)
        return self.mlp(h)