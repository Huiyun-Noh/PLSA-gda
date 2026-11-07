from __future__ import annotations

from math import exp
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

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
    def forward(self, x):
        return _GradReverse.apply(x)


def train_a2gnn(
    source_data, target_data,
    num_features: int, num_classes: int, device: torch.device,
    hidden: int = 32, lr: float = 1e-3, epochs: int = 50,
    adv_weight: float = 1e-3,
    src_val_data=None,
    *,
    class_weights: Optional[Tensor] = None,
    **kwargs,
) -> Dict:
    model = GDABackbone(num_features, hidden, num_classes).to(device)
    domain_disc = nn.Linear(hidden, 2).to(device)
    grl = _GRL()

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


        p = float(epoch) / max(epochs, 1)
        _GradReverse.rate = 2.0 / (1.0 + exp(-10.0 * p)) - 1.0

        d_s = domain_disc(grl(h_s))
        d_t = domain_disc(grl(h_t))
        domain_labels = torch.cat([
            torch.zeros(d_s.size(0), dtype=torch.long, device=device),
            torch.ones(d_t.size(0), dtype=torch.long, device=device),
        ])
        loss_adv = F.cross_entropy(torch.cat([d_s, d_t], 0), domain_labels)

        loss = loss_cls + adv_weight * loss_adv
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