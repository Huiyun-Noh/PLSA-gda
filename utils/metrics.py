from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

@torch.no_grad()
def compute_metrics(logits: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    pred = logits.argmax(dim=1)
    acc = float((pred == y).float().mean().item())
    y_np = y.cpu().numpy()
    p_np = pred.cpu().numpy()
    macro_f1 = float(f1_score(y_np, p_np, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_np, p_np, average="micro", zero_division=0))
    loss = float(F.cross_entropy(logits, y).item())
    
    try:
        probs = F.softmax(logits, dim=1).cpu().numpy()
        if probs.shape[1] == 2:
            auroc = float(roc_auc_score(y_np, probs[:, 1]))
        else:
            auroc = float(roc_auc_score(y_np, probs, multi_class="ovr", average="macro"))
    except (ValueError, IndexError):
        auroc = float("nan")
    return {"acc": acc, "macro_f1": macro_f1, "micro_f1": micro_f1, "auroc": auroc, "loss": loss}

@torch.no_grad()
def masked_metrics(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    return compute_metrics(logits[mask], y[mask])