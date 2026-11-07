from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, to_undirected


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def csr_from_edge_index(edge_index, n, undirected=True):
    row = edge_index[0].cpu().numpy()
    col = edge_index[1].cpu().numpy()
    dat = np.ones_like(row, dtype=np.float32)
    A = sp.coo_matrix((dat, (row, col)), shape=(n, n)).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    if undirected:
        A = A.maximum(A.T)
        A.setdiag(0)
        A.eliminate_zeros()
    return A


def stratified_split(y_long, train_ratio=0.7):
    y = y_long.view(-1).cpu().numpy()
    idx_tr, idx_va = [], []
    for c in np.unique(y):
        idx_c = np.where(y == c)[0]
        np.random.shuffle(idx_c)
        k = max(1, int(len(idx_c) * train_ratio))
        idx_tr.append(idx_c[:k])
        idx_va.append(idx_c[k:])
    return (torch.as_tensor(np.concatenate(idx_tr)),
            torch.as_tensor(np.concatenate(idx_va)))


def pack_subset(data, idx, undirected=True):
    idx = idx.cpu().numpy()
    keep = np.zeros(data.num_nodes, dtype=bool)
    keep[idx] = True

    row = data.edge_index[0].cpu().numpy()
    col = data.edge_index[1].cpu().numpy()
    mask_e = keep[row] & keep[col]
    row_s = row[mask_e]
    col_s = col[mask_e]


    inv = -np.ones(data.num_nodes, dtype=np.int64)
    inv[idx] = np.arange(idx.size, dtype=np.int64)
    row_s = inv[row_s]
    col_s = inv[col_s]

    ei = torch.from_numpy(np.vstack([row_s, col_s]))
    X = data.x[idx].cpu().numpy().astype(np.float32)
    y = data.y[idx].cpu().numpy().astype(np.int64)
    A = csr_from_edge_index(ei, n=X.shape[0], undirected=undirected)
    return y, A, X


def summarize(name, X, y, A):
    deg = np.asarray(A.sum(axis=1)).ravel()
    print(
        f"{name:<10} | N={X.shape[0]:,} d={X.shape[1]} "
        f"nnz={A.nnz:,} avg_deg={deg.mean():.2f} "
        f"y_counts={np.bincount(y)}"
    )


def load_single_domain_acm_dblp(base, name, use_undirected=True):
    prefix = name
    docs_path   = os.path.join(base, name, "raw", f"{prefix}_docs.txt")
    edges_path  = os.path.join(base, name, "raw", f"{prefix}_edgelist.txt")
    labels_path = os.path.join(base, name, "raw", f"{prefix}_labels.txt")

    X = np.loadtxt(docs_path, delimiter=",", dtype=np.float32)
    n_nodes, feat_dim = X.shape

    y = np.loadtxt(labels_path, dtype=np.int64).reshape(-1)
    if y.shape[0] != n_nodes:
        raise ValueError(f"#labels ({y.shape[0]}) != #nodes ({n_nodes}) for {name}")

    edges = np.loadtxt(edges_path, delimiter=",", dtype=np.int64)
    if edges.ndim == 1:
        edges = edges.reshape(1, 2)
    edge_index = torch.from_numpy(edges.T)

    edge_index, _ = remove_self_loops(edge_index)
    if use_undirected:
        edge_index = to_undirected(edge_index, num_nodes=n_nodes)

    data = Data(
        x=torch.from_numpy(X),
        y=torch.from_numpy(y),
        edge_index=edge_index,
    )
    return data

@dataclass
class DomainData:
    y: np.ndarray
    A: sp.csr_matrix
    X: np.ndarray

@dataclass
class DomainBundle:
    src_train: DomainData
    src_val: DomainData
    src_test: DomainData
    target: DomainData
    num_features: int
    num_classes: int
    source_name: str
    target_name: str

def load_acm_dblp_data(
    base: str,
    src: str,
    tar: str,
    src_train_ratio: float = 0.8,
    use_undirected: bool = True,
) -> DomainBundle:

    if src == tar:
        raise ValueError("Source and target domains must be different.")

    data_src_full = load_single_domain_acm_dblp(base, src, use_undirected=use_undirected)
    data_tar_full = load_single_domain_acm_dblp(base, tar, use_undirected=use_undirected)

    idx_tr, idx_val = stratified_split(data_src_full.y, train_ratio=src_train_ratio)
    idx_te = idx_val.clone()

    y_src_train, A_src_train, X_src_train = pack_subset(data_src_full, idx_tr, undirected=use_undirected)
    y_src_val,   A_src_val,   X_src_val   = pack_subset(data_src_full, idx_val, undirected=use_undirected)
    y_src_test,  A_src_test,  X_src_test  = pack_subset(data_src_full, idx_te, undirected=use_undirected)

    idx_all_t = torch.arange(data_tar_full.num_nodes)
    y_tar, A_tar, X_tar = pack_subset(data_tar_full, idx_all_t, undirected=use_undirected)

    summarize("src_train", X_src_train, y_src_train, A_src_train)
    summarize("src_val",   X_src_val,   y_src_val,   A_src_val)
    summarize("src_test",  X_src_test,  y_src_test,  A_src_test)
    summarize("target",    X_tar,       y_tar,       A_tar)

    num_features = X_src_train.shape[1]
    num_classes = int(np.max(y_src_train) + 1)

    return DomainBundle(
        src_train=DomainData(y=y_src_train, A=A_src_train, X=X_src_train),
        src_val=DomainData(y=y_src_val, A=A_src_val, X=X_src_val),
        src_test=DomainData(y=y_src_test, A=A_src_test, X=X_src_test),
        target=DomainData(y=y_tar, A=A_tar, X=X_tar),
        num_features=num_features,
        num_classes=num_classes,
        source_name=src,
        target_name=tar,
    )

def _select_top_k_labels(y_src, k=6):
    y_src = np.asarray(y_src).reshape(-1)
    labels, counts = np.unique(y_src, return_counts=True)
    order = np.argsort(-counts)
    return labels[order[:k]].tolist()

def _filter_and_remap(X, y, A, selected_labels):
    y = np.asarray(y).reshape(-1)
    mask = np.isin(y, selected_labels)
    idx = np.where(mask)[0]

    label_to_new = {old: i for i, old in enumerate(selected_labels)}
    y_new = np.array([label_to_new[int(lbl)] for lbl in y[idx]], dtype=np.int64)
    X_new = X[idx]
    A_new = A[idx, :][:, idx]
    return X_new, y_new, A_new, idx


def load_cora_data(
    base_dir: str = "data/cora_data",
    prefix: str = "good_cora",
    train_ratio: float = 0.8,
    k_labels: int = 6,
) -> DomainBundle:


    src_data = np.load(os.path.join(base_dir, f"{prefix}_src_data.npz"))
    X_src = src_data["X"].astype(np.float32)
    y_src = src_data["y"]
    A_src = sp.load_npz(os.path.join(base_dir, f"{prefix}_src_adj.npz"))

    tar_data = np.load(os.path.join(base_dir, f"{prefix}_tar_data.npz"))
    X_tar = tar_data["X"].astype(np.float32)
    y_tar = tar_data["y"]
    A_tar = sp.load_npz(os.path.join(base_dir, f"{prefix}_tar_adj.npz"))

    selected_labels = _select_top_k_labels(y_src, k=k_labels)
    print(f"  Selected labels (original IDs): {selected_labels}")

    X_src_f, y_src_f, A_src_f, _ = _filter_and_remap(X_src, y_src, A_src, selected_labels)
    X_tar_f, y_tar_f, A_tar_f, _ = _filter_and_remap(X_tar, y_tar, A_tar, selected_labels)

    A_src_f = sp.csr_matrix(A_src_f)
    A_src_f.setdiag(0); A_src_f.eliminate_zeros()
    A_tar_f = sp.csr_matrix(A_tar_f)
    A_tar_f.setdiag(0); A_tar_f.eliminate_zeros()

    y_t = torch.from_numpy(y_src_f)
    idx_tr, idx_val = stratified_split(y_t, train_ratio=train_ratio)
    idx_te = idx_val.clone()

    def _extract(A, X, y, idx):
        idx = idx.cpu().numpy()
        return y[idx], sp.csr_matrix(A[idx, :][:, idx]), X[idx].astype(np.float32)

    y_src_train, A_src_train, X_src_train = _extract(A_src_f, X_src_f, y_src_f, idx_tr)
    y_src_val,   A_src_val,   X_src_val   = _extract(A_src_f, X_src_f, y_src_f, idx_val)
    y_src_test,  A_src_test,  X_src_test  = _extract(A_src_f, X_src_f, y_src_f, idx_te)

    for A in [A_src_train, A_src_val, A_src_test]:
        A.setdiag(0); A.eliminate_zeros()

    y_tar_f = y_tar_f.astype(np.int64)
    X_tar_f = X_tar_f.astype(np.float32)

    summarize("src_train", X_src_train, y_src_train, A_src_train)
    summarize("src_val",   X_src_val,   y_src_val,   A_src_val)
    summarize("target",    X_tar_f,     y_tar_f,     A_tar_f)

    num_features = X_src_train.shape[1]
    num_classes = int(np.max(y_src_train) + 1)

    return DomainBundle(
        src_train=DomainData(y=y_src_train, A=A_src_train, X=X_src_train),
        src_val=DomainData(y=y_src_val, A=A_src_val, X=X_src_val),
        src_test=DomainData(y=y_src_test, A=A_src_test, X=X_src_test),
        target=DomainData(y=y_tar_f, A=A_tar_f, X=X_tar_f),
        num_features=num_features,
        num_classes=num_classes,
        source_name="cora_src",
        target_name="cora_tgt",
    )


def _load_twitch_domain(base_dir, lang):
    import json, csv, glob

    d = os.path.join(base_dir, lang)

    edge_file = os.path.join(d, f"musae_{lang}_edges.csv")
    with open(edge_file) as f:
        reader = csv.DictReader(f)
        edges = [(int(r["from"]), int(r["to"])) for r in reader]

    target_file = os.path.join(d, f"musae_{lang}_target.csv")
    with open(target_file) as f:
        rows = list(csv.DictReader(f))
    n_nodes = len(rows)

    id_to_label = {}
    for r in rows:
        nid = int(r["new_id"])
        id_to_label[nid] = 1 if r["mature"] == "True" else 0
    y = np.array([id_to_label[i] for i in range(n_nodes)], dtype=np.int64)

    feat_files = sorted(glob.glob(os.path.join(d, "*.json")))
    feat_file = feat_files[0]
    with open(feat_file) as f:
        feats = json.load(f)
    all_ids = set()
    for v in feats.values():
        all_ids.update(v)
    feat_dim = max(all_ids) + 1

    X = np.zeros((n_nodes, feat_dim), dtype=np.float32)
    for node_id_str, feat_list in feats.items():
        node_id = int(node_id_str)
        if node_id < n_nodes:
            for fid in feat_list:
                X[node_id, fid] = 1.0

    row = np.array([e[0] for e in edges] + [e[1] for e in edges], dtype=np.int64)
    col = np.array([e[1] for e in edges] + [e[0] for e in edges], dtype=np.int64)
    data = np.ones(len(row), dtype=np.float32)
    A = sp.csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes))
    A.setdiag(0)
    A.eliminate_zeros()

    return X, y, A


def load_twitch_data(
    base_dir: str,
    src: str,
    tar: str,
    src_train_ratio: float = 0.8,
) -> DomainBundle:


    X_src, y_src, A_src = _load_twitch_domain(base_dir, src)
    X_tar, y_tar, A_tar = _load_twitch_domain(base_dir, tar)

    d_max = max(X_src.shape[1], X_tar.shape[1])
    if X_src.shape[1] < d_max:
        X_src = np.pad(X_src, ((0, 0), (0, d_max - X_src.shape[1])))
    if X_tar.shape[1] < d_max:
        X_tar = np.pad(X_tar, ((0, 0), (0, d_max - X_tar.shape[1])))

    y_t = torch.from_numpy(y_src)
    idx_tr, idx_val = stratified_split(y_t, train_ratio=src_train_ratio)
    idx_te = idx_val.clone()

    def _extract(A, X, y, idx):
        idx = idx.cpu().numpy()
        return y[idx], sp.csr_matrix(A[idx, :][:, idx]), X[idx].astype(np.float32)

    y_src_train, A_src_train, X_src_train = _extract(A_src, X_src, y_src, idx_tr)
    y_src_val,   A_src_val,   X_src_val   = _extract(A_src, X_src, y_src, idx_val)
    y_src_test,  A_src_test,  X_src_test  = _extract(A_src, X_src, y_src, idx_te)

    for A in [A_src_train, A_src_val, A_src_test]:
        A.setdiag(0); A.eliminate_zeros()

    summarize("src_train", X_src_train, y_src_train, A_src_train)
    summarize("src_val",   X_src_val,   y_src_val,   A_src_val)
    summarize("target",    X_tar,       y_tar,       A_tar)

    num_features = X_src_train.shape[1]
    num_classes = int(np.max(y_src_train) + 1)

    return DomainBundle(
        src_train=DomainData(y=y_src_train, A=A_src_train, X=X_src_train),
        src_val=DomainData(y=y_src_val, A=A_src_val, X=X_src_val),
        src_test=DomainData(y=y_src_test, A=A_src_test, X=X_src_test),
        target=DomainData(y=y_tar, A=A_tar, X=X_tar),
        num_features=num_features,
        num_classes=num_classes,
        source_name=f"twitch_{src}",
        target_name=f"twitch_{tar}",
    )


_AIRPORT_DEFAULT_FEAT_DIM    = 32
_AIRPORT_DEFAULT_CLASS_MEANS = {0: -0.5, 1: 0.0, 2: 0.5, 3: 1.0}

def _read_airport_edgelist(edge_path, label_path):
    edges = []
    with open(edge_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            a, b = s.split()[:2]
            edges.append((int(a), int(b)))

    labels = {}
    with open(label_path, "r") as f:
        first = f.readline().strip().split()
        try:
            a0, b0 = int(first[0]), int(first[1])
            labels[a0] = b0
        except Exception:
            pass
        for line in f:
            s = line.strip()
            if not s:
                continue
            nid, lab = s.split()[:2]
            labels[int(nid)] = int(lab)

    nodes = set()
    for a, b in edges:
        nodes.add(a); nodes.add(b)
    nodes |= set(labels.keys())
    id_map = {nid: i for i, nid in enumerate(sorted(nodes))}

    ei = np.array([[id_map[a] for a, _ in edges],
                   [id_map[b] for _, b in edges]], dtype=np.int64)
    n = len(id_map)
    y = np.zeros(n, dtype=np.int64)
    for old, lab in labels.items():
        y[id_map[old]] = lab
    return ei, y, n


def _make_gaussian_by_class(y_long, feat_dim, class_means_dict, std=1.0):
    y = y_long.view(-1).cpu()
    X = torch.empty(y.numel(), feat_dim)
    std_t = torch.as_tensor(std, dtype=torch.float32)
    std_t = std_t.view(1, 1) if std_t.dim() == 0 else std_t.view(1, -1)
    for c in torch.unique(y).tolist():
        idx = (y == c).nonzero(as_tuple=False).view(-1)
        z = torch.randn(idx.numel(), feat_dim)
        mu = torch.as_tensor(class_means_dict.get(c, 0.0), dtype=torch.float32)
        mu = mu.view(1, 1).expand(1, feat_dim) if mu.dim() == 0 else mu.view(1, -1)
        X[idx] = z * std_t + mu
    return X


def _load_airport_domain(base_dir, name, feat_dim, class_means, use_undirected=True):
    edge_path = os.path.join(base_dir, f"{name}-airports.edgelist")
    lab_path  = os.path.join(base_dir, f"labels-{name}-airports.txt")
    ei_np, y_np, n = _read_airport_edgelist(edge_path, lab_path)

    A = csr_from_edge_index(torch.from_numpy(ei_np), n, undirected=use_undirected)
    X = _make_gaussian_by_class(torch.from_numpy(y_np), feat_dim, class_means,
                                 std=1.0).cpu().numpy().astype(np.float32)
    return X, y_np, A


def load_airport_data(
    base_dir: str,
    src: str,
    tar: str,
    src_train_ratio: float = 0.8,
    feat_dim: int = _AIRPORT_DEFAULT_FEAT_DIM,
    class_means: dict = None,
    use_undirected: bool = True,
) -> DomainBundle:


    if class_means is None:
        class_means = dict(_AIRPORT_DEFAULT_CLASS_MEANS)

    X_src, y_src, A_src = _load_airport_domain(base_dir, src, feat_dim, class_means, use_undirected)
    X_tar, y_tar, A_tar = _load_airport_domain(base_dir, tar, feat_dim, class_means, use_undirected)

    idx_tr, idx_val = stratified_split(torch.from_numpy(y_src), train_ratio=src_train_ratio)
    idx_te = idx_val.clone()

    def _extract(A, X, y, idx):
        idx = idx.cpu().numpy()
        return y[idx], sp.csr_matrix(A[idx, :][:, idx]), X[idx].astype(np.float32)

    y_src_train, A_src_train, X_src_train = _extract(A_src, X_src, y_src, idx_tr)
    y_src_val,   A_src_val,   X_src_val   = _extract(A_src, X_src, y_src, idx_val)
    y_src_test,  A_src_test,  X_src_test  = _extract(A_src, X_src, y_src, idx_te)

    for A in (A_src_train, A_src_val, A_src_test):
        A.setdiag(0); A.eliminate_zeros()

    summarize("src_train", X_src_train, y_src_train, A_src_train)
    summarize("src_val",   X_src_val,   y_src_val,   A_src_val)
    summarize("target",    X_tar,       y_tar,       A_tar)

    num_features = X_src_train.shape[1]
    num_classes = int(max(np.max(y_src_train), np.max(y_tar)) + 1)

    return DomainBundle(
        src_train=DomainData(y=y_src_train, A=A_src_train, X=X_src_train),
        src_val=DomainData(y=y_src_val, A=A_src_val, X=X_src_val),
        src_test=DomainData(y=y_src_test, A=A_src_test, X=X_src_test),
        target=DomainData(y=y_tar, A=A_tar, X=X_tar),
        num_features=num_features,
        num_classes=num_classes,
        source_name=f"airport_{src}",
        target_name=f"airport_{tar}",
    )


def _load_blog_domain(filepath):
    import scipy.io as sio

    mat = sio.loadmat(filepath)
    X = np.array(mat["attrb"], dtype=np.float32)
    group = np.array(mat["group"])
    y = group.argmax(axis=1).astype(np.int64)
    A = mat["network"]
    A = sp.csr_matrix(A)
    A.setdiag(0)
    A.eliminate_zeros()

    A = A.maximum(A.T)
    A.setdiag(0)
    A.eliminate_zeros()

    return X, y, A


def load_blog_data(
    base_dir: str = "data/blog",
    src: str = "Blog1_30",
    tar: str = "Blog2_30",
    src_train_ratio: float = 0.8,
) -> DomainBundle:
    X_src, y_src, A_src = _load_blog_domain(os.path.join(base_dir, f"{src}.mat"))
    X_tar, y_tar, A_tar = _load_blog_domain(os.path.join(base_dir, f"{tar}.mat"))

    y_t = torch.from_numpy(y_src)
    idx_tr, idx_val = stratified_split(y_t, train_ratio=src_train_ratio)
    idx_te = idx_val.clone()

    def _extract(A, X, y, idx):
        idx = idx.cpu().numpy()
        return y[idx], sp.csr_matrix(A[idx, :][:, idx]), X[idx].astype(np.float32)

    y_src_train, A_src_train, X_src_train = _extract(A_src, X_src, y_src, idx_tr)
    y_src_val,   A_src_val,   X_src_val   = _extract(A_src, X_src, y_src, idx_val)
    y_src_test,  A_src_test,  X_src_test  = _extract(A_src, X_src, y_src, idx_te)

    for A in [A_src_train, A_src_val, A_src_test]:
        A.setdiag(0); A.eliminate_zeros()

    summarize("src_train", X_src_train, y_src_train, A_src_train)
    summarize("src_val",   X_src_val,   y_src_val,   A_src_val)
    summarize("target",    X_tar,       y_tar,       A_tar)

    num_features = X_src_train.shape[1]
    num_classes = int(np.max(y_src_train) + 1)

    return DomainBundle(
        src_train=DomainData(y=y_src_train, A=A_src_train, X=X_src_train),
        src_val=DomainData(y=y_src_val, A=A_src_val, X=X_src_val),
        src_test=DomainData(y=y_src_test, A=A_src_test, X=X_src_test),
        target=DomainData(y=y_tar, A=A_tar, X=X_tar),
        num_features=num_features,
        num_classes=num_classes,
        source_name=src,
        target_name=tar,
    )


def _csbm_validate(n, p, P_edge, d, sigma, mus):
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer.")
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("p must be a 1D probability vector of length L.")
    if not np.isclose(p.sum(), 1.0):
        raise ValueError("p must sum to 1.")
    if np.any(p < 0):
        raise ValueError("p must have nonnegative entries.")
    L = p.size
    P_edge = np.asarray(P_edge, dtype=float)
    if P_edge.shape != (L, L):
        raise ValueError(f"P_edge must be of shape ({L}, {L}).")
    if np.any(P_edge < 0) or np.any(P_edge > 1):
        raise ValueError("Entries of P_edge must be in [0, 1].")
    if not isinstance(d, int) or d <= 0:
        raise ValueError("d must be a positive integer.")
    if sigma < 0:
        raise ValueError("sigma must be nonnegative.")
    mus = np.asarray(mus, dtype=float)
    if mus.shape != (L, d):
        raise ValueError(f"mus must have shape (L, d)=({L}, {d}).")
    return L, d


def _csbm_labels(n, p, sorted_labels=True):
    p = np.asarray(p, dtype=float)
    L = p.size
    counts = np.random.multinomial(n, p)
    if sorted_labels:
        return np.concatenate([np.full(c, cls, dtype=int) for cls, c in enumerate(counts)])
    return np.random.choice(L, size=n, p=p).astype(int)


def _csbm_block_bounds(y, L):
    counts = np.bincount(y, minlength=L)
    starts = np.cumsum(np.r_[0, counts[:-1]])
    return counts, starts, starts + counts


def _csbm_sym_block(n_block, p_edge):
    if n_block <= 1:
        return np.zeros((n_block, n_block), dtype=np.uint8)
    upper = np.triu(np.random.binomial(1, p_edge, size=(n_block, n_block)), k=1)
    return (upper + upper.T).astype(np.uint8)


def _csbm_adj_sorted(y, P_edge_sym):
    y = np.asarray(y, dtype=int)
    n = y.size
    L = P_edge_sym.shape[0]
    counts, starts, ends = _csbm_block_bounds(y, L)
    A = np.zeros((n, n), dtype=np.uint8)
    for a in range(L):
        s, e = starts[a], ends[a]
        if counts[a] == 0:
            continue
        A[s:e, s:e] = _csbm_sym_block(counts[a], P_edge_sym[a, a])
    for a in range(L):
        for b in range(a + 1, L):
            s1, e1 = starts[a], ends[a]
            s2, e2 = starts[b], ends[b]
            if counts[a] == 0 or counts[b] == 0:
                continue
            block = np.random.binomial(1, P_edge_sym[a, b],
                                        size=(counts[a], counts[b])).astype(np.uint8)
            A[s1:e1, s2:e2] = block
            A[s2:e2, s1:e1] = block.T
    np.fill_diagonal(A, 0)
    return A


def _csbm_features_sorted(y, d, sigma, mus):
    y = np.asarray(y, dtype=int)
    L = mus.shape[0]
    counts, _, _ = _csbm_block_bounds(y, L)
    cov = (sigma ** 2) * np.eye(d)
    blocks = []
    for a in range(L):
        if counts[a] == 0:
            continue
        blocks.append(np.random.multivariate_normal(mean=mus[a], cov=cov, size=counts[a]))
    return np.vstack(blocks) if blocks else np.zeros((0, d), dtype=float)


def generate_graph_data(n, p, P_edge, d, sigma, mus,
                        sorted_labels=True, shuffle=False, sparse=False):
    L, d_checked = _csbm_validate(n, p, P_edge, d, sigma, mus)
    P_edge_sym = 0.5 * (np.asarray(P_edge, dtype=float)
                        + np.asarray(P_edge, dtype=float).T)
    y = _csbm_labels(n, np.asarray(p, dtype=float), sorted_labels=sorted_labels)
    if not sorted_labels:
        counts = np.bincount(y, minlength=L)
        sorted_y = np.concatenate([np.full(c, cls, dtype=int)
                                    for cls, c in enumerate(counts)])
        A_sorted = _csbm_adj_sorted(sorted_y, P_edge_sym)
        X_sorted = _csbm_features_sorted(sorted_y, d_checked, sigma, mus)
        idxs_per_class = [np.where(y == cls)[0] for cls in range(L)]
        starts = np.cumsum(np.r_[0, counts[:-1]])
        ends = starts + counts
        X = np.empty((n, d_checked), dtype=float)
        for cls in range(L):
            X[idxs_per_class[cls], :] = X_sorted[starts[cls]:ends[cls], :]
        perm = np.empty(n, dtype=int)
        for cls in range(L):
            perm[starts[cls]:ends[cls]] = idxs_per_class[cls][: ends[cls] - starts[cls]]
        inv_perm = np.empty(n, dtype=int)
        inv_perm[perm] = np.arange(n)
        A = A_sorted[inv_perm][:, inv_perm]
    else:
        y = np.asarray(y, dtype=int)
        A = _csbm_adj_sorted(y, P_edge_sym)
        X = _csbm_features_sorted(y, d_checked, sigma, mus)
    if shuffle:
        perm = np.random.permutation(n)
        y = y[perm]
        if hasattr(A, "toarray"):
            try:
                A = A[perm][:, perm]
            except Exception:
                A = np.asarray(A.todense())[perm][:, perm]
        else:
            A = A[perm][:, perm]
        X = X[perm, :]
    if sparse:
        try:
            A = sp.csr_matrix(np.asarray(A, dtype=np.uint8))
        except Exception:
            pass
    return y, A, X


def load_csbm_bundle(seed: int,
                     n: int,
                     L: int,
                     d: int,
                     sigma: float,
                     prior,
                     P_src,
                     P_tar) -> DomainBundle:

    set_seed(seed)
    mus = np.random.default_rng(seed).standard_normal((L, d)) / np.sqrt(d)
    P_src_a = np.asarray(P_src, dtype=float)
    P_tar_a = np.asarray(P_tar, dtype=float)
    prior_a = np.asarray(prior, dtype=float)

    def _gen(P):
        y, A, X = generate_graph_data(int(n), prior_a, P, int(d), float(sigma),
                                       mus, sparse=True)
        return (np.asarray(y, dtype=np.int64), A,
                np.asarray(X, dtype=np.float32))

    y_tr, A_tr, X_tr = _gen(P_src_a)
    y_va, A_va, X_va = _gen(P_src_a)
    y_te, A_te, X_te = _gen(P_src_a)
    y_t,  A_t,  X_t  = _gen(P_tar_a)

    bundle = DomainBundle(
        src_train=DomainData(y=y_tr, A=A_tr, X=X_tr),
        src_val=  DomainData(y=y_va, A=A_va, X=X_va),
        src_test= DomainData(y=y_te, A=A_te, X=X_te),
        target=   DomainData(y=y_t,  A=A_t,  X=X_t),
        num_features=int(d),
        num_classes=int(L),
        source_name='csbm_src',
        target_name='csbm_tar',
    )
    bundle.R_true = P_tar_a / np.clip(P_src_a, 1e-12, None)
    return bundle