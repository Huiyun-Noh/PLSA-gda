import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, issparse

def _label_pair_stats(A: csr_matrix, y: np.ndarray, L: int):
    idx_by = [np.flatnonzero(y == k) for k in range(L)]
    sizes = [len(x) for x in idx_by]

    coo = A.tocoo(copy=False)
    upper = coo.row < coo.col
    ru, rv = coo.row[upper], coo.col[upper]

    E = np.zeros((L, L), dtype=np.int64)
    for u, v in zip(ru, rv):
        ku, lv = int(y[u]), int(y[v])
        k, l = (ku, lv) if ku <= lv else (lv, ku)
        E[k, l] += 1

    N = np.zeros((L, L), dtype=np.int64)
    for k in range(L):
        nk = sizes[k]
        N[k, k] = nk * (nk - 1) // 2
        for l in range(k + 1, L):
            nl = sizes[l]
            N[k, l] = nk * nl

    p = np.zeros((L, L), dtype=np.float64)
    nz = N > 0
    p[nz] = E[nz] / N[nz]
    return N, E, p, idx_by


def sample_adjusted_B(A_src: csr_matrix, y_src: np.ndarray, w_mat: np.ndarray) -> csr_matrix:

    A = A_src.tocsr(copy=True)
    A.setdiag(0); A.eliminate_zeros()
    if (A != A.T).nnz != 0:
        A = ((A + A.T) > 0).astype(np.uint8).tocsr()

    L = int(w_mat.shape[0])
    N_pairs, E_pairs, p_src, idx_by = _label_pair_stats(A, y_src, L)
    p_tilde = np.clip(w_mat * p_src, 0.0, 1.0)

    coo = A.tocoo(copy=False)
    upper = coo.row < coo.col
    ru, rv = coo.row[upper], coo.col[upper]

    keep = np.ones(len(ru), dtype=bool)
    for i in range(len(ru)):
        ku, lv = int(y_src[ru[i]]), int(y_src[rv[i]])
        k, l = (ku, lv) if ku <= lv else (lv, ku)
        if w_mat[k, l] < 1.0:
            keep[i] = (np.random.rand() < w_mat[k, l])

    ru_new, rv_new = ru[keep], rv[keep]
    dat = np.ones_like(ru_new, dtype=np.uint8)

    row = np.concatenate([ru_new, rv_new])
    col = np.concatenate([rv_new, ru_new])
    data = np.concatenate([dat, dat])

    B = coo_matrix((data, (row, col)), shape=A.shape).tocsr()
    B.setdiag(0); B.eliminate_zeros()

    cooB = B.tocoo(copy=False)
    upperB = cooB.row < cooB.col
    bu, bv = cooB.row[upperB], cooB.col[upperB]
    E_curr = np.zeros((L, L), dtype=np.int64)
    for u, v in zip(bu, bv):
        ku, lv = int(y_src[u]), int(y_src[v])
        k, l = (ku, lv) if ku <= lv else (lv, ku)
        E_curr[k, l] += 1

    B = B.tolil(copy=True)
    for k in range(L):
        for l in range(k, L):
            if N_pairs[k, l] == 0 or w_mat[k, l] <= 1.0:
                continue
            if p_src[k, l] >= 1.0:
                continue
            q = (p_tilde[k, l] - p_src[k, l]) / (1.0 - p_src[k, l])
            q = float(np.clip(q, 0.0, 1.0))

            N_non = int(N_pairs[k, l] - E_curr[k, l])
            if N_non <= 0 or q == 0.0:
                continue
            M = np.random.binomial(N_non, q)

            idx_k, idx_l = idx_by[k], idx_by[l]
            added = 0
            attempts = 0
            max_attempts = M * 20 + 1000

            if k == l:
                nk = len(idx_k)
                if nk >= 2:
                    while added < M and attempts < max_attempts:
                        u, v = np.random.choice(idx_k, size=2, replace=False)
                        if u == v:
                            continue
                        uu, vv = (u, v) if u < v else (v, u)
                        if B[uu, vv] == 0:
                            B[uu, vv] = 1; B[vv, uu] = 1
                            added += 1
                        attempts += 1
            else:
                nk, nl = len(idx_k), len(idx_l)
                if nk > 0 and nl > 0:
                    while added < M and attempts < max_attempts:
                        u = np.random.choice(idx_k); v = np.random.choice(idx_l)
                        uu, vv = (u, v) if u < v else (v, u)
                        if B[uu, vv] == 0:
                            B[uu, vv] = 1; B[vv, uu] = 1
                            added += 1
                        attempts += 1

    B = B.tocsr()
    B.setdiag(0); B.eliminate_zeros()
    return B


def average_adjusted_B(
    A_src: csr_matrix,
    y_src: np.ndarray,
    w_mat: np.ndarray,
    num_samples: int = 1,
) -> csr_matrix:
    B_sum = None
    for _ in range(num_samples):
        B_t = sample_adjusted_B_simple(A_src, y_src, w_mat).astype(np.float32)
        B_sum = B_t if B_sum is None else (B_sum + B_t)
    B_avg = (B_sum / float(num_samples)).tocsr()
    B_avg.eliminate_zeros()
    return B_avg


def sample_adjusted_B_simple(
    A_src: csr_matrix,
    y_src: np.ndarray,
    w_mat: np.ndarray,
    eps_w: float = 0.05,
) -> csr_matrix:

    if issparse(A_src):
        A = A_src.toarray()
    else:
        A = np.asarray(A_src)

    A = (A > 0).astype(np.uint8)
    n = A.shape[0]

    np.fill_diagonal(A, 0)
    A = ((A + A.T) > 0).astype(np.uint8)

    y = np.asarray(y_src, dtype=int)
    L = w_mat.shape[0]
    idx_by = [np.flatnonzero(y == k) for k in range(L)]

    B = np.zeros_like(A, dtype=np.uint8)

    for k in range(L):
        idx_k = idx_by[k]
        nk = len(idx_k)

        for l in range(k, L):
            idx_l = idx_by[l]
            nl = len(idx_l)

            if nk == 0 or nl == 0:
                continue

            w = float(w_mat[k, l])

            if k == l:
                if nk < 2:
                    continue
                rows, cols = np.triu_indices(nk, k=1)
                u = idx_k[rows]
                v = idx_k[cols]
            else:
                u = np.repeat(idx_k, nl)
                v = np.tile(idx_l, nk)

            if u.size == 0:
                continue

            a_block = A[u, v]
            M = a_block.size
            if M == 0:
                continue

            p_src = a_block.mean()
            p_tilde = np.clip(w * p_src, 0.0, 1.0)

            if abs(w - 1.0) <= eps_w or p_src == p_tilde:
                b_block = a_block.copy()
            else:
                if w < 1.0 - eps_w:
                    b_block = np.zeros_like(a_block, dtype=np.uint8)
                    is_edge = (a_block == 1)
                    n_edge = is_edge.sum()
                    if n_edge > 0:
                        keep = (np.random.rand(n_edge) < w).astype(np.uint8)
                        b_block[is_edge] = keep
                else:
                    b_block = a_block.copy()
                    if p_src < 1.0:
                        is_non = (a_block == 0)
                        n_non = is_non.sum()
                        if n_non > 0:
                            q = (p_tilde - p_src) / (1.0 - p_src)
                            q = float(np.clip(q, 0.0, 1.0))
                            add = (np.random.rand(n_non) < q).astype(np.uint8)
                            b_block[is_non] = add

            B[u, v] = b_block
            B[v, u] = b_block

    np.fill_diagonal(B, 0)
    return csr_matrix(B)