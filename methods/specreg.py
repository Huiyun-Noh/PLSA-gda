from __future__ import annotations

from math import exp
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add

from models.backbone import GDABackbone
from utils.metrics import compute_metrics


class _GradReverse(torch.autograd.Function):
    rate: float = 0.0

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * _GradReverse.rate


class _GRL(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return _GradReverse.apply(x)


class _LaplacianSmoother(MessagePassing):
    def __init__(self):
        super().__init__(aggr="add")

    def forward(self, h: Tensor, edge_index: Tensor) -> Tensor:
        n = h.size(0)
        edge_weight = torch.ones(edge_index.size(1), device=h.device, dtype=h.dtype)
        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=n)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=h, norm=norm, size=None)

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.view(-1, 1) * x_j


_lap_smoother = _LaplacianSmoother()


def _laplacian_filtered(h: Tensor, edge_index: Tensor) -> Tensor:
    return h - _lap_smoother(h, edge_index)

def _entropy_loss(logits: Tensor) -> Tensor:
    probs = F.softmax(logits, dim=1)
    probs = torch.clamp(probs, min=1e-9, max=1.0)
    return torch.mean(torch.sum(-probs * torch.log(probs), dim=1))

def _mmd_loss(x: Tensor, y: Tensor, sample_n: int = 512, times: int = 3) -> Tensor:
    n_x, n_y = x.size(0), y.size(0)
    k = min(sample_n, n_x, n_y)
    total = torch.tensor(0.0, device=x.device)
    for _ in range(times):
        xi = x[torch.randint(n_x, (k,), device=x.device)]
        yi = y[torch.randint(n_y, (k,), device=y.device)]
        all_ = torch.cat([xi, yi], dim=0)
        sq = (all_ ** 2).sum(dim=1, keepdim=True)
        l2 = (sq + sq.t() - 2.0 * (all_ @ all_.t())).clamp_min(0.0)
        bw = (l2.sum() / max((2 * k) ** 2 - 2 * k, 1)).clamp_min(1e-6)
        kern = torch.exp(-l2 / bw)
        XX = kern[:k, :k].mean()
        YY = kern[k:, k:].mean()
        XY = kern[:k, k:].mean()
        total = total + (XX + YY - 2 * XY)
    return total / times


def train_specreg(
    source_data, target_data,
    num_features: int, num_classes: int, device: torch.device,
    hidden: int = 32, lr: float = 1e-3, epochs: int = 50,
    adv_weight: float = 0.1, spec_weight: float = 0.5, ent_weight: float = 0.01,
    src_val_data=None,
    *,
    class_weights: Optional[Tensor] = None,
    **kwargs,
) -> Dict:
    model = GDABackbone(num_features, hidden, num_classes).to(device)
    domain_disc = nn.Sequential(
        nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2),
    ).to(device)
    grl = _GRL()

    _lap_smoother.to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(domain_disc.parameters()), lr=lr
    )

    best_loss = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(epochs):
        model.train()
        domain_disc.train()

        h_s = model.encode(source_data.x, source_data.edge_index,
                           edge_weight=getattr(source_data, 'edge_weight', None))
        h_t = model.encode(target_data.x, target_data.edge_index)

        logits_s = model.classify(h_s)
        loss_cls = F.cross_entropy(logits_s, source_data.y, weight=class_weights)

        lh_s = _laplacian_filtered(h_s, source_data.edge_index)
        lh_t = _laplacian_filtered(h_t, target_data.edge_index)
        loss_spec = _mmd_loss(lh_s, lh_t)


        p = float(epoch) / max(epochs, 1)
        _GradReverse.rate = 2.0 / (1.0 + exp(-10.0 * p)) - 1.0
        d_s = domain_disc(grl(h_s))
        d_t = domain_disc(grl(h_t))
        domain_labels = torch.cat([
            torch.zeros(d_s.size(0), dtype=torch.long, device=device),
            torch.ones(d_t.size(0), dtype=torch.long, device=device),
        ])
        loss_adv = F.cross_entropy(torch.cat([d_s, d_t], 0), domain_labels)

        logits_t = model.classify(h_t)
        loss_ent = _entropy_loss(logits_t)
        ent_scale = epoch / max(epochs, 1) * ent_weight

        loss = (loss_cls + adv_weight * loss_adv
                + spec_weight * loss_spec + ent_scale * loss_ent)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss_cls.item() < best_loss:
            best_loss = loss_cls.item()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        sl = model(source_data.x, source_data.edge_index,
                   edge_weight=getattr(source_data, 'edge_weight', None))
        tl = model(target_data.x, target_data.edge_index)
        out = {
            "source_all": compute_metrics(sl, source_data.y),
            "target_all": compute_metrics(tl, target_data.y),
            "target_logits": tl.detach(),
        }
        if src_val_data is not None:
            svl = model(src_val_data.x, src_val_data.edge_index)
            out["source_val"] = compute_metrics(svl, src_val_data.y)

    return out