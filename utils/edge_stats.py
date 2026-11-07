import numpy as np
import torch


@torch.no_grad()
def predict_labels(model, X_np, device, batch_size=256):

    model.eval()
    N = X_np.shape[0]
    out = np.empty(N, dtype=np.int64)

    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)

        X_slice = X_np[s:e]
        if isinstance(X_slice, np.ndarray) and X_slice.dtype == np.float32 and X_slice.flags['C_CONTIGUOUS']:
            Xb = torch.from_numpy(X_slice).to(device)
        else:
            Xb = torch.as_tensor(X_slice, dtype=torch.float32, device=device)

        pb = model(Xb).argmax(dim=1).detach().cpu().numpy().astype(np.int64)
        out[s:e] = pb

        del Xb, X_slice

    return out


def _upper_tri_nonedge_indices(A_np):
    iu, iv = np.triu_indices(A_np.shape[0], k=1)
    mask = (A_np[iu, iv] == 0)
    return iu[mask], iv[mask]


def _upper_tri_edge_indices(A_np):
    iu, iv = np.triu_indices(A_np.shape[0], k=1)
    mask = (A_np[iu, iv] == 1)
    return iu[mask], iv[mask]


def edge_rate_uppertri(A_np):
    iu, iv = np.triu_indices(A_np.shape[0], k=1)
    M = iu.size
    if M == 0: return np.nan
    return float((A_np[iu, iv] == 1).sum()) / float(M)


def class_prior_empirical(y, L=None):
    y = np.asarray(y, dtype=int)
    L = int(L or (np.max(y) + 1))
    counts = np.bincount(y, minlength=L).astype(np.float64)
    if counts.sum() == 0:
        raise ValueError("Empty labels when computing empirical prior.")
    return counts / counts.sum()


def PT_joint_pred_on_edges(A_tar, pred_tar, L):
    u, v = _upper_tri_edge_indices(A_tar)
    M = len(u)
    if M == 0:
        return np.full((L, L), np.nan, dtype=np.float64)
    i, j = pred_tar[u], pred_tar[v]
    out = np.zeros((L, L), dtype=np.float64)
    np.add.at(out, (i, j), 1.0)
    out /= M
    return out


def PS_joint_pred_label_on_edges(A_src, pred_src, y_src, L):
    u, v = _upper_tri_edge_indices(A_src)
    M = len(u)
    if M == 0:
        return np.full((L, L, L, L), np.nan, dtype=np.float64)
    i, j = pred_src[u], pred_src[v]
    k, l = y_src[u], y_src[v]
    T = np.zeros((L, L, L, L), dtype=np.float64)
    np.add.at(T, (i, j, k, l), 1.0)
    T /= M
    return T


def QS_labelpair_on_edges(A_src, y_src, L):
    u, v = _upper_tri_edge_indices(A_src)
    M = len(u)
    if M == 0:
        raise ValueError("No edges found in source graph.")
    k, l = y_src[u], y_src[v]
    Q = np.zeros((L, L), dtype=np.float64)
    np.add.at(Q, (k, l), 1.0)
    Q /= M
    return Q


def PS_cond_edgeprob_given_labelpair(A_src, y_src, L, laplace=0.0):
    y_src = np.asarray(y_src, dtype=int)
    iu, iv = np.triu_indices(A_src.shape[0], k=1)
    k = y_src[iu]
    l = y_src[iv]
    total = np.zeros((L, L), dtype=np.int64)
    np.add.at(total, (k, l), 1)

    is_edge = (A_src[iu, iv] == 1)
    k1 = k[is_edge]
    l1 = l[is_edge]
    edges = np.zeros((L, L), dtype=np.int64)
    np.add.at(edges, (k1, l1), 1)

    total_f = total.astype(np.float64)
    total_f = (total_f + total_f.T)/2
    edges_f = edges.astype(np.float64)
    edges_f = (edges_f + edges_f.T)/2

    P1 = (edges_f + float(laplace)) / (total_f + 2.0 * float(laplace))
    P1 = np.clip(P1, 1e-12, 1.0 - 1e-12)
    P0 = 1.0 - P1

    return P1, P0, total