import numpy as np
import cvxpy as cp

from utils.edge_stats import (
    predict_labels, PT_joint_pred_on_edges, PS_joint_pred_label_on_edges,
    QS_labelpair_on_edges, edge_rate_uppertri
)


class PairAlign:

    def __init__(self, lam=1.0, solver="SCS", symmetrize=True, pred_batch_size=256, verbose=False):
        self.lam = float(lam)
        self.solver = solver
        self.symmetrize = bool(symmetrize)
        self.pred_batch_size = int(pred_batch_size)
        self.verbose = bool(verbose)

        self.L = None
        self.status = None
        self.p_vec = None
        self.M_mat = None
        self.q_vec = None
        self.W_hat = None
        self.R_joint = None
        self.p_edge_S = None
        self.p_edge_T = None

    @staticmethod
    def _vec_from_LxL(M):
        L = M.shape[0]
        assert M.shape == (L, L)
        return M.reshape(L * L)

    @staticmethod
    def _mat_from_LxLxLxL(T):
        L = T.shape[0]
        assert T.shape == (L, L, L, L)
        return T.reshape(L * L, L * L)

    def _solve_w_cvx(self, M, p, q):
        
        M = np.asarray(M, dtype=float)
        p = np.asarray(p, dtype=float).reshape(-1)
        q = np.asarray(q, dtype=float).reshape(-1)

        n = M.shape[1]
        assert M.shape == (n, n) and p.size == n and q.size == n

        w = cp.Variable(n)
        obj = cp.sum_squares(M @ w - p) + self.lam * cp.sum_squares(w - 1)
        cons = [w >= 0, q @ w == 1]

        prob = cp.Problem(cp.Minimize(obj), cons)
        prob.solve(solver=self.solver, verbose=self.verbose)

        if w.value is None:
            raise RuntimeError(f"CVX solve failed: status={prob.status}")

        return np.array(w.value, dtype=float), prob.status

    def _build_system(self, A_src_train, y_src_train, A_tar, pred_src, pred_tar, L):
        PT_edge_pred = PT_joint_pred_on_edges(A_tar, pred_tar, L)
        PS_edge_pred_label = PS_joint_pred_label_on_edges(
            A_src_train, pred_src, y_src_train.astype(np.int64), L
        )
        Q_S = QS_labelpair_on_edges(A_src_train, y_src_train.astype(np.int64), L)

        p_vec = self._vec_from_LxL(PT_edge_pred)
        M_mat = self._mat_from_LxLxLxL(PS_edge_pred_label)
        q_vec = self._vec_from_LxL(Q_S)
        return p_vec, M_mat, q_vec, Q_S

    def fit(self,
            A_src_train, y_src_train,
            A_tar, X_src_train, X_tar,
            model, device):
        
        self.L = int(np.max(y_src_train) + 1)

        pred_src = predict_labels(model, X_src_train, device, batch_size=self.pred_batch_size)
        pred_tar = predict_labels(model, X_tar,        device, batch_size=self.pred_batch_size)

        self.p_vec, self.M_mat, self.q_vec, Q_S = self._build_system(
            A_src_train, y_src_train, A_tar, pred_src, pred_tar, self.L
        )

        w_hat, self.status = self._solve_w_cvx(self.M_mat, self.p_vec, self.q_vec)
        W_hat = w_hat.reshape(self.L, self.L)
        if self.symmetrize:
            W_hat = 0.5 * (W_hat + W_hat.T)
        self.W_hat = W_hat

        self.p_edge_S = edge_rate_uppertri(A_src_train)
        self.p_edge_T = edge_rate_uppertri(A_tar)
        scale = self.p_edge_T / (self.p_edge_S + 1e-12)
        self.R_joint = self.W_hat * scale

        return self

    def get_weights(self):

        return self.W_hat, self.R_joint