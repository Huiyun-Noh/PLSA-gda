import numpy as np
import torch
import torch.nn as nn

from utils.edge_stats import edge_rate_uppertri, PS_cond_edgeprob_given_labelpair


@torch.no_grad()
def predict_probs_calibrated(model, X_np, device, batch_size=256, T=1.0, bias=None):
    """Return calibrated class probabilities F on TARGET as torch tensor (n,L) on device."""
    model.eval()
    outs = []
    n = X_np.shape[0]
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        Xb = torch.as_tensor(X_np[s:e], dtype=torch.float32, device=device)
        logits = model(Xb) / float(T)
        if bias is not None:
            logits = logits + torch.as_tensor(bias, dtype=logits.dtype, device=logits.device)
        outs.append(torch.softmax(logits, dim=1))
    return torch.cat(outs, dim=0)


class PLSA:

    def __init__(self,
                 max_steps=2000,
                 lr=5e-2,
                 init_noise=1e-3,
                 pred_batch_size=256,
                 progress_every=None,
                 es_check_every=None,
                 es_tol=1e-12,
                 min_steps=50,
                 verbose=False,
                 es_patience=999,
                 laplace=1.0,
                 lam=0.0,
                 c0_eps=1e-12,
                 exp_clip=50.0):
        self.max_steps = int(max_steps)
        self.lr = float(lr)
        self.init_noise = float(init_noise)
        self.pred_batch_size = int(pred_batch_size)
        self.progress_every = int(progress_every) if progress_every else None
        self.es_check_every = int(es_check_every) if es_check_every else None
        self.es_tol = float(es_tol)
        self.min_steps = int(min_steps)
        self.verbose = bool(verbose)
        self.es_patience = es_patience
        self.laplace = float(laplace)
        self.lam = float(lam)
        self.c0_eps = float(c0_eps)
        self.exp_clip = float(exp_clip)

        self.L = None
        self.theta_hat = None
        self.c0_edge_hat = None
        self.W_hat = None
        self.loss_hist = []
        self.n_epochs_ = 0

    @staticmethod
    def _estimate_p0_y_from_y(y_src, L, device, dtype):
        y_np = np.asarray(y_src, dtype=int)
        counts = np.bincount(y_np, minlength=L)
        p_y = counts.astype(np.float64) / counts.sum()
        p_y = np.clip(p_y, 1e-12, None)
        p_y = p_y / p_y.sum()
        p0_y = torch.as_tensor(p_y, dtype=dtype, device=device)
        return p0_y

    @staticmethod
    def _estimate_c0_edge_from_edges(A_src, y_src, L, laplace=1.0):

        y_np = np.asarray(y_src, dtype=int)
        A_np = A_src.toarray() if hasattr(A_src, "toarray") else np.asarray(A_src)
        A_np = np.asarray(A_np)
        A_bool = (A_np != 0)
        A_bool = np.logical_or(A_bool, A_bool.T)
        np.fill_diagonal(A_bool, False)

        iu, iv = np.triu_indices(A_bool.shape[0], k=1)
        mask = A_bool[iu, iv]
        u = iu[mask]
        v = iv[mask]

        counts = np.zeros((L, L), dtype=np.float64)
        if u.size > 0:
            yu = y_np[u]
            yv = y_np[v]
            np.add.at(counts, (yu, yv), 1.0)
            np.add.at(counts, (yv, yu), 1.0)

        if laplace > 0:
            counts = counts + float(laplace)

        total = counts.sum()
        if total <= 0:
            raise ValueError("No source edges found and laplace <= 0; cannot estimate c0_edge.")

        c0 = counts / total
        c0 = 0.5 * (c0 + c0.T)
        c0 = np.clip(c0, 1e-12, None)
        c0 = c0 / c0.sum()
        return c0

    @staticmethod
    def _loss_edges(F_tgt, edge_u, edge_v, theta_over_p0, theta,
                    c0_edge_safe, lam, eps=1e-9):

        Fu = F_tgt[edge_u]
        Fv = F_tgt[edge_v]
        S = ((Fu @ theta_over_p0) * Fv).sum(dim=1)
        S = torch.clamp(S, min=eps)
        loss = -torch.log(S).mean()
        if lam > 0.0:
            W_edge = theta / c0_edge_safe
            loss = loss + float(lam) * ((W_edge - 1.0) ** 2).sum()
        return loss

    def fit(self, A_src, y_src, A_tar, X_tar, model, device,
            T=1.0, bias=None, F_precomputed=None):

        if F_precomputed is not None:
            F_tgt = torch.as_tensor(F_precomputed, dtype=torch.float32, device=device)
        else:
            F_tgt = predict_probs_calibrated(model, X_tar, device,
                                             batch_size=self.pred_batch_size,
                                             T=T, bias=bias).to(torch.float32)
        self.L = F_tgt.shape[1]

        A = torch.as_tensor(A_tar, dtype=torch.bool, device=F_tgt.device)
        A = A | A.T
        A.fill_diagonal_(False)
        iu, iv = torch.where(torch.triu(A, diagonal=1))
        n_edges = iu.size(0)
        if n_edges == 0:
            raise ValueError("No target edges found; PLSA requires edge-conditioned samples.")
        if self.verbose:
            print(f"[PLSA] Target edges: {n_edges}")

        p0_y = self._estimate_p0_y_from_y(y_src, self.L,
                                          device=F_tgt.device, dtype=F_tgt.dtype)
        p0_yy = torch.outer(p0_y, p0_y).clamp(min=1e-12)

        c0_edge_np = self._estimate_c0_edge_from_edges(
            A_src, y_src, self.L, laplace=self.laplace
        )
        c0_edge = torch.as_tensor(c0_edge_np, dtype=F_tgt.dtype, device=F_tgt.device)
        c0_edge_safe = torch.clamp(c0_edge, min=self.c0_eps)

        theta = torch.ones(self.L, self.L, dtype=F_tgt.dtype, device=F_tgt.device)
        if self.init_noise > 0:
            theta = theta + self.init_noise * torch.randn_like(theta)
            theta = torch.clamp(theta, min=1e-12)
        theta = theta / theta.sum()

        self.loss_hist = []
        best_loss = np.inf
        best_theta = None
        no_improve_checks = 0
        patience = (self.es_patience if self.es_patience is not None
                    else max(3, (self.es_check_every or 1)))
        self.n_epochs_ = self.max_steps

        for epoch in range(1, self.max_steps + 1):

            with torch.no_grad():
                theta.copy_(0.5 * (theta + theta.T))
                theta.clamp_(min=1e-18)
                theta.div_(theta.sum())

            theta.requires_grad_(True)

            theta_over_p0 = theta / p0_yy
            loss = self._loss_edges(
                F_tgt, iu, iv, theta_over_p0, theta,
                c0_edge_safe, self.lam
            )

            cur_loss = float(loss.item())
            self.loss_hist.append(cur_loss)

            if cur_loss < best_loss - 1e-12 * max(1.0, abs(best_loss)):
                best_loss = cur_loss
                best_theta = theta.detach().cpu().clone()

            loss.backward()
            with torch.no_grad():
                eg_step = torch.clamp(-self.lr * theta.grad,
                                      min=-self.exp_clip, max=self.exp_clip)
                theta.mul_(torch.exp(eg_step))
                theta.clamp_(min=1e-18)
                theta.div_(theta.sum())
                theta.grad.zero_()

            if self.verbose and self.progress_every is not None:
                if (epoch % self.progress_every == 0) or epoch == 1 or epoch == self.max_steps:
                    print(f"[PLSA] Epoch {epoch}/{self.max_steps}  loss={cur_loss:.6f}")

            if (epoch >= self.min_steps) and (self.es_check_every is not None):
                if (epoch % self.es_check_every == 0) and len(self.loss_hist) > self.es_check_every:
                    prev = self.loss_hist[-1 - self.es_check_every]
                    rel_diff = abs(prev - cur_loss) / (abs(prev) + 1e-12)
                    if rel_diff < self.es_tol:
                        no_improve_checks += 1
                    else:
                        no_improve_checks = 0
                    if no_improve_checks >= patience:
                        if self.verbose:
                            print(f"[PLSA] Early stop at epoch {epoch}")
                        self.n_epochs_ = epoch
                        break

        if best_theta is not None:
            theta = best_theta.to(F_tgt.device)

        with torch.no_grad():
            theta = 0.5 * (theta + theta.T)
            theta = torch.clamp(theta, min=1e-18)
            theta = theta / theta.sum()

            W = theta / c0_edge_safe
            W = 0.5 * (W + W.T)

            theta_np = theta.detach().cpu().numpy()
            c0_np = c0_edge.detach().cpu().numpy()
            W_np = W.detach().cpu().numpy()

            if not np.allclose(theta_np.sum(), 1.0, atol=1e-5):
                raise RuntimeError("PLSA sanity check failed: theta_hat does not sum to 1.")
            if not np.allclose(theta_np, theta_np.T, atol=1e-6):
                raise RuntimeError("PLSA sanity check failed: theta_hat is not symmetric.")
            if np.any(W_np < -1e-8):
                raise RuntimeError("PLSA sanity check failed: W_hat has negative entries.")
            if not np.allclose((c0_np * W_np).sum(), 1.0, atol=1e-5):
                raise RuntimeError("PLSA sanity check failed: <c0_edge_hat, W_hat> != 1.")

            self.theta_hat = theta_np
            self.c0_edge_hat = c0_np
            self.W_hat = W_np

        return self

    def get_weights(self):
        
        return self.W_hat
    

@torch.no_grad()
def estimate_importance_weights_mlls(
    model,
    X_tar_np,
    y_src_np,
    device,
    batch_size=256,
    T=1.0,
    bias=None,
    max_iters=200,
    tol=1e-8,
    eps=1e-12,
    verbose=False,
    F_precomputed=None,
    lam_w=0.0,
    lr_w=1e-2,
):

    if F_precomputed is not None:
        F = torch.as_tensor(F_precomputed, dtype=torch.float64)
    else:
        F = predict_probs_calibrated(
            model, X_tar_np, device,
            batch_size=batch_size, T=T, bias=bias,
        ).to(torch.float64)
    m, L = F.shape

    y_src_np = np.asarray(y_src_np)
    counts = np.bincount(y_src_np, minlength=L).astype(np.float64)
    if counts.sum() == 0:
        raise ValueError("No source samples found when computing p_S.")
    p_s_np = counts / counts.sum()
    p_s = torch.as_tensor(p_s_np, dtype=F.dtype, device=F.device)
    p_s_safe = torch.clamp(p_s, min=eps)

    w = torch.ones(L, dtype=F.dtype, device=F.device)
    best_w = w.clone()
    best_obj = -np.inf
    lr = float(lr_w)
    no_improve = 0
    min_lr = lr * 1e-4

    for it in range(max_iters):

        num = F * w.view(1, -1)
        denom = num.sum(dim=1, keepdim=True)
        denom = torch.clamp(denom, min=eps)
        r = num / denom
        r_mean = r.mean(dim=0)

        if lam_w == 0.0:

            w_new = r_mean / p_s_safe
        else:


            w_safe = torch.clamp(w, min=eps)
            grad = r_mean / w_safe - lam_w * (w - 1.0)
            w_new = w * torch.exp(lr * grad)

        w_new = torch.clamp(w_new, min=eps)
        norm = torch.sum(p_s_safe * w_new)
        norm = torch.clamp(norm, min=eps)
        w_new = w_new / norm


        if lam_w > 0.0:
            log_lik = torch.log(torch.clamp(F @ w_new, min=eps)).mean()
            reg = -0.5 * lam_w * ((w_new - 1.0) ** 2).sum()
            obj = (log_lik + reg).item()

            if obj > best_obj:
                best_obj = obj
                best_w = w_new.clone()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 3 and lr > min_lr:
                    lr *= 0.5
                    w = best_w.clone()
                    no_improve = 0
                    if verbose:
                        print(f"[MLLS-EG] iter {it+1:03d}  lr halved -> {lr:.2e}, revert to best")
                    continue

        delta = torch.sum(torch.abs(w_new - w)).item()
        if verbose:
            tag = "EG" if lam_w > 0.0 else "EM"
            extra = f"  obj={obj:.6f}  lr={lr:.2e}" if lam_w > 0.0 else ""
            print(f"[MLLS-{tag}] iter {it+1:03d}  |Δw|_1 = {delta:.3e}{extra}")
        w = w_new
        if delta < tol:
            if verbose:
                tag = "EG" if lam_w > 0.0 else "EM"
                print(f"[MLLS-{tag}] converged at iter {it+1} with |Δw|_1={delta:.3e}")
            break


    if lam_w > 0.0:
        w = best_w

    q = p_s_safe * w
    q = q / q.sum()

    w_np = w.detach().cpu().numpy().astype(np.float64)
    q_np = q.detach().cpu().numpy().astype(np.float64)
    p_s_np = p_s.detach().cpu().numpy().astype(np.float64)

    return w_np, q_np, p_s_np