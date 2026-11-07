import numpy as np

from utils.edge_stats import predict_labels


class StruRW:

    def __init__(self, symmetrize=True, pred_batch_size=256, verbose=False):
        self.symmetrize = bool(symmetrize)
        self.pred_batch_size = int(pred_batch_size)
        self.verbose = bool(verbose)

        self.L = None
        self.B_S = None
        self.B_T = None
        self.R_joint = None

    @staticmethod
    def _estimate_B(A, y, L):
        
        y = np.asarray(y, dtype=np.int64)
        counts = np.bincount(y, minlength=L).astype(np.float64)

        if hasattr(A, "tocoo"):
            coo = A.tocoo()
            rows, cols = coo.row, coo.col
        else:
            rows, cols = np.nonzero(np.asarray(A))

        B = np.zeros((L, L), dtype=np.float64)
        np.add.at(B, (y[rows], y[cols]), 1)

        denom = np.outer(counts, counts)
        denom = np.maximum(denom, 1e-12)
        B = B / denom
        return B

    def fit(self, A_src_train, y_src_train, A_tar, X_src_train, X_tar, model, device):
        
        self.L = int(np.max(y_src_train) + 1)

        self.B_S = self._estimate_B(A_src_train, y_src_train, self.L)

        pred_tar = predict_labels(model, X_tar, device, batch_size=self.pred_batch_size)
        self.B_T = self._estimate_B(A_tar, pred_tar, self.L)

        B_S_safe = np.maximum(self.B_S, 1e-12)
        self.R_joint = self.B_T / B_S_safe
        self.R_joint[~np.isfinite(self.R_joint)] = 1.0

        if self.symmetrize:
            self.R_joint = 0.5 * (self.R_joint + self.R_joint.T)

        if self.verbose:
            print(f"[StruRW] B^S:\n{self.B_S}")
            print(f"[StruRW] B^T:\n{self.B_T}")
            print(f"[StruRW] R_joint:\n{self.R_joint}")

        return self

    def get_weights(self):

        return self.R_joint