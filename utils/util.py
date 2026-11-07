import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import scipy.sparse as sp
from torch_geometric.data import Data
from sklearn.metrics import f1_score


def to_tensors(X_np: np.ndarray, y_np: np.ndarray):
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.long)
    return X, y


def make_loader(X_tensor: torch.Tensor,
                y_tensor: torch.Tensor,
                batch_size: int = 128,
                shuffle: bool = True) -> DataLoader:
    
    ds = TensorDataset(X_tensor, y_tensor)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def csr_to_edge_index(A):
    coo = A.tocoo()
    row = torch.from_numpy(coo.row.astype(np.int64))
    col = torch.from_numpy(coo.col.astype(np.int64))
    return torch.stack([row, col], dim=0)


def prepare_graph_data(A, X, y, w_mat=None, lam=1.0, use_matrix_values_as_weight=False):
    if not sp.isspmatrix_csr(A):
        A = sp.csr_matrix(A)
    A = A.copy()
    A.setdiag(0); A.eliminate_zeros()
    if (A != A.T).nnz != 0:
        A = A.maximum(A.T)
        A.setdiag(0); A.eliminate_zeros()

    edge_index = csr_to_edge_index(A)

    x = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).long()

    if use_matrix_values_as_weight:
        coo = A.tocoo()
        edge_weight = torch.from_numpy(coo.data.astype(np.float32))
        data = Data(x=x, edge_index=edge_index, y=y_t,
                    edge_weight=edge_weight)
    elif w_mat is not None:
        y_np = np.asarray(y, dtype=int)
        ei = edge_index.cpu().numpy()
        u, v = ei[0], ei[1]

        w_edge = w_mat[y_np[u], y_np[v]].astype(np.float32)

        gamma_edge = (1.0 - float(lam)) + float(lam) * w_edge

        edge_weight = torch.from_numpy(gamma_edge)
        data = Data(x=x, edge_index=edge_index, y=y_t,
                    edge_weight=edge_weight)
    else:
        data = Data(x=x, edge_index=edge_index, y=y_t)

    return data


def split_train_val(n, val_ratio=0.1):
    idx = np.random.permutation(n)
    cut = int(n * (1 - val_ratio))
    return torch.as_tensor(idx[:cut]), torch.as_tensor(idx[cut:])


def evaluate_metrics(logits, y_true):
    preds = logits.argmax(dim=-1).cpu().numpy()
    y_np  = y_true.cpu().numpy()
    acc = (preds == y_np).mean()
    f1m = f1_score(y_np, preds, average="macro", zero_division=0)
    return acc, f1m, preds