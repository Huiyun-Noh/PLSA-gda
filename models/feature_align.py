from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAligner(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, hidden_dims: tuple = (64, 32)):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _mmd_rbf(x: torch.Tensor, y: torch.Tensor,
             kernel_mul: float = 2.0, kernel_num: int = 5) -> torch.Tensor:

    max_n = 512
    if x.size(0) > max_n:
        idx = torch.randperm(x.size(0), device=x.device)[:max_n]
        x = x[idx]
    if y.size(0) > max_n:
        idx = torch.randperm(y.size(0), device=y.device)[:max_n]
        y = y[idx]

    n_x, n_y = x.size(0), y.size(0)
    total = torch.cat([x, y], dim=0)

    sq_norms = (total ** 2).sum(dim=1, keepdim=True)
    l2 = sq_norms + sq_norms.t() - 2.0 * (total @ total.t())
    l2 = l2.clamp_min(0.0)

    bw = (l2.sum() / max((n_x + n_y) ** 2 - (n_x + n_y), 1)).clamp_min(1e-6)
    bw /= kernel_mul ** (kernel_num // 2)

    kern = sum(torch.exp(-l2 / (bw * kernel_mul ** i)) for i in range(kernel_num))

    XX = kern[:n_x, :n_x].mean()
    YY = kern[n_x:, n_x:].mean()
    XY = kern[:n_x, n_x:].mean()
    return XX + YY - 2 * XY


def train_feature_aligner(
    X_src: np.ndarray,
    y_src: np.ndarray,
    X_tgt: np.ndarray,
    num_classes: int,
    device: torch.device,
    out_dim: int = 32,
    hidden_dims: tuple = (64, 32),
    lr: float = 1e-3,
    epochs: int = 200,
    lam: float = 1.0,
    batch_size: int = 512,
    verbose: bool = True,
    src_weights: Optional[np.ndarray] = None,
) -> FeatureAligner:

    in_dim = X_src.shape[1]


    x_s = torch.tensor(X_src, dtype=torch.float32, device=device)
    y_s = torch.tensor(y_src, dtype=torch.long, device=device)
    x_t = torch.tensor(X_tgt, dtype=torch.float32, device=device)

    if src_weights is not None:
        w = np.asarray(src_weights, dtype=np.float64)
        w = np.clip(w, 1e-12, None)
        w_src = torch.tensor(w / w.sum(), dtype=torch.float32, device=device)
    else:
        w_src = None


    g = FeatureAligner(in_dim, out_dim, hidden_dims).to(device)
    cls_head = nn.Linear(out_dim, num_classes).to(device)

    optimizer = torch.optim.Adam(
        list(g.parameters()) + list(cls_head.parameters()), lr=lr
    )

    n_src, n_tgt = x_s.size(0), x_t.size(0)

    for ep in range(1, epochs + 1):
        g.train()
        cls_head.train()


        z_s = g(x_s)
        logits = cls_head(z_s)
        loss_ce = F.cross_entropy(logits, y_s)


        if w_src is not None:
            idx_s = torch.multinomial(w_src, batch_size, replacement=True)
        else:
            idx_s = torch.randint(n_src, (batch_size,), device=device)
        idx_t = torch.randint(n_tgt, (batch_size,), device=device)
        loss_mmd = _mmd_rbf(z_s[idx_s], g(x_t[idx_t]))

        loss = loss_ce + lam * loss_mmd

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and (ep % 50 == 0 or ep == 1):
            acc = (logits.argmax(1) == y_s).float().mean().item()
            print(f"  [FeatureAlign {ep:03d}/{epochs}] "
                  f"CE={loss_ce.item():.4f}  MMD={loss_mmd.item():.4f}  "
                  f"src_acc={acc:.4f}")

    g.eval()
    g.cpu()
    return g


@torch.no_grad()
def transform_features(g: FeatureAligner, X: np.ndarray,
                        device: torch.device, batch_size: int = 2048) -> np.ndarray:

    g.eval().to(device)
    n = X.shape[0]
    parts = []
    for i in range(0, n, batch_size):
        xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32, device=device)
        zb = g(xb).cpu().numpy()
        parts.append(zb)
    g.cpu()
    return np.concatenate(parts, axis=0).astype(np.float32)