from __future__ import annotations
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, get_laplacian
from scipy.special import comb

from models.backbone import GDABackbone
from utils.metrics import compute_metrics


class _BernProp(MessagePassing):


    def __init__(self, K: int, learnable: bool = True):
        super().__init__(aggr="add")
        self.K = K
        self.temp = nn.Parameter(torch.Tensor(K + 1), requires_grad=learnable)
        self.temp.data.fill_(1.0)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        TEMP = F.relu(self.temp)


        ei1, n1 = get_laplacian(
            edge_index, normalization="sym", dtype=x.dtype,
            num_nodes=x.size(0),
        )

        ei2, n2 = add_self_loops(ei1, -n1, fill_value=2.0, num_nodes=x.size(0))


        tmp = [x]
        h = x
        for _ in range(self.K):
            h = self.propagate(ei2, x=h, norm=n2, size=None)
            tmp.append(h)


        out = (comb(self.K, 0) / (2 ** self.K)) * TEMP[0] * tmp[self.K]
        for i in range(self.K):
            h = tmp[self.K - i - 1]
            for _ in range(i + 1):
                h = self.propagate(ei1, x=h, norm=n1, size=None)
            out = out + (comb(self.K, i + 1) / (2 ** self.K)) * TEMP[i + 1] * h

        return out

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.view(-1, 1) * x_j


def _entropy_loss(logits: Tensor) -> Tensor:
    
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    a = probs.sum(dim=0).clamp_min(1e-8)
    return -(probs * log_probs / (a / a.sum().clamp_min(1e-8))).sum(dim=1).mean()


def _mmd_loss(src_feat: Tensor, tgt_feat: Tensor,
              sample_n: int = 512, times: int = 5,
              kernel_mul: float = 2.0, kernel_num: int = 5) -> Tensor:
    
    n_s, n_t = src_feat.size(0), tgt_feat.size(0)
    k = min(sample_n, n_s, n_t)
    total = 0.0
    for _ in range(times):
        si = torch.randint(n_s, (k,), device=src_feat.device)
        ti = torch.randint(n_t, (k,), device=tgt_feat.device)
        s, t = src_feat[si], tgt_feat[ti]
        all_ = torch.cat([s, t], dim=0)
        sq = (all_ ** 2).sum(dim=1, keepdim=True)
        l2 = (sq + sq.t() - 2.0 * (all_ @ all_.t())).clamp_min(0.0)
        bw = (l2.sum() / max((2 * k) ** 2 - 2 * k, 1)).clamp_min(1e-6)
        bw /= kernel_mul ** (kernel_num // 2)
        kern = sum(torch.exp(-l2 / (bw * kernel_mul ** i)) for i in range(kernel_num))
        total = total + (kern[:k, :k].mean() + kern[k:, k:].mean()
                         - kern[:k, k:].mean() - kern[k:, :k].mean())
    return total / times


def train_dgsda(
    source_data, target_data,
    num_features: int, num_classes: int, device: torch.device,
    hidden: int = 32, lr: float = 1e-3, epochs: int = 50,
    alpha_coeff: float = 1.0, beta_coeff: float = 0.5, gamma_coeff: float = 0.05,
    bern_K: int = 5,
    src_val_data=None,
    *,
    class_weights: Optional[Tensor] = None,
    **kwargs,
) -> Dict:
    model = GDABackbone(num_features, hidden, num_classes).to(device)

    
    bern_src = _BernProp(K=bern_K, learnable=True).to(device)
    bern_tgt = _BernProp(K=bern_K, learnable=True).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(bern_src.parameters())
        + list(bern_tgt.parameters()), lr=lr
    )

    best_loss = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(epochs):
        model.train()
        bern_src.train()
        bern_tgt.train()

    
        x_s = bern_src(source_data.x, source_data.edge_index)
        x_t = bern_tgt(target_data.x, target_data.edge_index)

        h_s = model.encode(x_s, source_data.edge_index,
                           edge_weight=getattr(source_data, 'edge_weight', None))
        h_t = model.encode(x_t, target_data.edge_index)
    
        logits_s = model.classify(h_s)
        loss_cls = F.cross_entropy(logits_s, source_data.y, weight=class_weights)

        loss_filter = F.l1_loss(
            F.relu(bern_src.temp), F.relu(bern_tgt.temp)
        )

        loss_mmd = _mmd_loss(h_s, h_t)

        logits_t = model.classify(h_t)
        loss_ent = _entropy_loss(logits_t)

        loss = (loss_cls + alpha_coeff * loss_filter
                + beta_coeff * loss_mmd + gamma_coeff * loss_ent)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss_cls.item() < best_loss:
            best_loss = loss_cls.item()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():

        x_s_eval = bern_src(source_data.x, source_data.edge_index)
        x_t_eval = bern_src(target_data.x, target_data.edge_index)
        sl = model(x_s_eval, source_data.edge_index,
                   edge_weight=getattr(source_data, 'edge_weight', None))
        tl = model(x_t_eval, target_data.edge_index)
        out = {
            "source_all": compute_metrics(sl, source_data.y),
            "target_all": compute_metrics(tl, target_data.y),
            "target_logits": tl.detach(),
        }
        if src_val_data is not None:
            x_sv_eval = bern_src(src_val_data.x, src_val_data.edge_index)
            svl = model(x_sv_eval, src_val_data.edge_index,
                       edge_weight=getattr(src_val_data, 'edge_weight', None))
            out["source_val"] = compute_metrics(svl, src_val_data.y)

    return out