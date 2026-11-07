from __future__ import annotations

from collections import Counter
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add

from models.backbone import GDABackbone
from utils.metrics import compute_metrics


class _CachedGCNConv(MessagePassing):

    def __init__(self, in_ch, out_ch, weight=None, bias=None, use_bias=True, **kw):
        super().__init__(aggr="add", **kw)
        self.cache_dict: dict = {}
        self.weight = weight if weight is not None else Parameter(torch.Tensor(in_ch, out_ch).float())
        if weight is None:
            glorot(self.weight)
        if bias is not None:
            self.bias = bias
        elif use_bias:
            self.bias = Parameter(torch.zeros(out_ch, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

    @staticmethod
    def _norm(edge_index, N, ew=None, dtype=None):
        if ew is None:
            ew = torch.ones(edge_index.size(1), dtype=dtype, device=edge_index.device)
        edge_index, ew = add_remaining_self_loops(edge_index, ew, 1, N)
        row, col = edge_index
        deg = scatter_add(ew, row, dim=0, dim_size=N)
        inv = deg.pow(-0.5)
        inv[inv == float("inf")] = 0
        return edge_index, inv[row] * ew * inv[col]

    def forward(self, x, edge_index, cache_name="default"):
        x = x @ self.weight
        if cache_name not in self.cache_dict:
            self.cache_dict[cache_name] = self._norm(edge_index, x.size(0), dtype=x.dtype)
        ei, n = self.cache_dict[cache_name]
        out = self.propagate(ei, x=x, norm=n)
        return out + self.bias if self.bias is not None else out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class _PPMIConv(_CachedGCNConv):

    def __init__(self, in_ch, out_ch, weight=None, bias=None,
                 path_len=10, **kw):
        super().__init__(in_ch, out_ch, weight, bias, **kw)
        self.path_len = path_len

    def _norm(self, edge_index, N, ew=None, dtype=None):
        dev = edge_index.device
        
        adj: dict = {}
        for a, b in edge_index.t().cpu().numpy():
            a, b = int(a), int(b)
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)
        
        wc: dict = {}
        for _ in range(40):
            for a in adj:
                cur = a
                for _ in range(np.random.randint(1, self.path_len + 1)):
                    b = adj[cur][np.random.randint(len(adj[cur]))]
                    wc.setdefault(a, Counter())[b] += 1
                    cur = b
        
        normed = {}
        for a, ctr in wc.items():
            s = sum(ctr.values())
            normed[a] = {b: c / s for b, c in ctr.items()}
        psums: Counter = Counter()
        for d in normed.values():
            for b, p in d.items():
                psums[b] += p
        rows, cols, vals = [], [], []
        nv = len(psums)
        for a, d in normed.items():
            for b, p in d.items():
                v = max(np.log(p / psums[b] * nv / self.path_len), 0.0)
                if v > 0:
                    rows.append(a); cols.append(b); vals.append(v)
        ei = torch.tensor([rows, cols], dtype=torch.long, device=dev)
        ew = torch.tensor(vals, dtype=torch.float32, device=dev)
        ei, ew = add_remaining_self_loops(ei, ew, 1, N)
        row, col = ei
        deg = scatter_add(ew, row, dim=0, dim_size=N)
        inv = deg.pow(-0.5)
        inv[inv == float("inf")] = 0
        return ei, (inv[row] * ew * inv[col]).float()


class _Attention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Linear(dim, 1)

    def forward(self, views: list) -> Tensor:
        stacked = torch.stack(views, dim=1)
        weights = F.softmax(self.w(stacked), dim=1)
        return (stacked * weights).sum(dim=1)


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return grad.neg() * ctx.alpha, None


def train_sagda(
    source_data, target_data,
    num_features: int, num_classes: int, device: torch.device,
    hidden: int = 32, lr: float = 4e-3, epochs: int = 200,
    weight_decay: float = 3e-3, adv_dim: int = 40,
    ppmi_path_len: int = 10,
    src_val_data=None,
    *,
    class_weights: Optional[Tensor] = None,
    **kwargs,
) -> Dict:
    
    model = GDABackbone(num_features, hidden, num_classes).to(device)

    ppmi_layer = _PPMIConv(hidden, hidden, path_len=ppmi_path_len).to(device)
    att = _Attention(hidden).to(device)

    domain_disc = nn.Sequential(
        nn.Linear(hidden, adv_dim), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(adv_dim, 2),
    ).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(ppmi_layer.parameters())
        + list(att.parameters()) + list(domain_disc.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()

    best_loss = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(epochs):
        model.train(); ppmi_layer.train(); att.train(); domain_disc.train()

        alpha = min((epoch + 1) / max(epochs, 1), 0.05)

        h_s = model.encode(source_data.x, source_data.edge_index,
                           edge_weight=getattr(source_data, 'edge_weight', None))

        h_t_gcn = model.encode(target_data.x, target_data.edge_index)
        h_t_ppmi = ppmi_layer(h_t_gcn, target_data.edge_index, "target_ppmi")
        h_t = att([h_t_gcn, h_t_ppmi])

        logits_s = model.classify(h_s)
        loss_cls = F.cross_entropy(logits_s, source_data.y, weight=class_weights)

        d_s = domain_disc(_GradReverse.apply(h_s, alpha))
        d_t = domain_disc(_GradReverse.apply(h_t, alpha))
        loss_adv = (
            loss_fn(d_s, torch.zeros(d_s.size(0), dtype=torch.long, device=device))
            + loss_fn(d_t, torch.ones(d_t.size(0), dtype=torch.long, device=device))
        )

        logits_t = model.classify(h_t)
        probs_t = F.softmax(logits_t, dim=1).clamp(min=1e-9)
        loss_ent = torch.mean(torch.sum(-probs_t * torch.log(probs_t), dim=1))
        ent_scale = epoch / max(epochs, 1) * 0.01

        loss = loss_cls + loss_adv + ent_scale * loss_ent

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss_cls.item() < best_loss:
            best_loss = loss_cls.item()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval(); ppmi_layer.eval(); att.eval()
    with torch.no_grad():
        sl = model(source_data.x, source_data.edge_index,
                   edge_weight=getattr(source_data, 'edge_weight', None))
        h_t_gcn = model.encode(target_data.x, target_data.edge_index)
        h_t_ppmi = ppmi_layer(h_t_gcn, target_data.edge_index, "target_ppmi")
        h_t = att([h_t_gcn, h_t_ppmi])
        tl = model.classify(h_t)
        out = {
            "source_all": compute_metrics(sl, source_data.y),
            "target_all": compute_metrics(tl, target_data.y),
            "target_logits": tl.detach(),
        }
        if src_val_data is not None:
            svl = model(src_val_data.x, src_val_data.edge_index,
                        edge_weight=getattr(src_val_data, 'edge_weight', None))
            out["source_val"] = compute_metrics(svl, src_val_data.y)

    return out