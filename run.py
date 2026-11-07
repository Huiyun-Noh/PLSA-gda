from __future__ import annotations

import os
import time

import numpy as np


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from scipy.sparse import csr_matrix

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from utils.data import set_seed
from utils.util import to_tensors, make_loader, prepare_graph_data
from utils.calibration import Calibrator
from utils.graph_reweighting import average_adjusted_B
from utils.train import Trainer

from models.mlp import SimpleMLP
from models.plsa import predict_probs_calibrated, estimate_importance_weights_mlls
from models.plsa import PLSA
from models.pairalign import PairAlign
from models.strurw import StruRW
from utils.edge_stats import edge_rate_uppertri
from models.feature_align import train_feature_aligner, transform_features

from methods.a2gnn import train_a2gnn
from methods.dgsda import train_dgsda
from methods.specreg import train_specreg
from methods.sagda import train_sagda


def _generate_labels(n, p, sorted_labels=True):
    p = np.asarray(p, dtype=float)
    L = p.size
    counts = np.random.multinomial(n, p)
    if sorted_labels:
        return np.concatenate([np.full(c, cls, dtype=int)
                                for cls, c in enumerate(counts)])
    return np.random.choice(L, size=n, p=p).astype(int)


def _block_boundaries_from_sorted_y(y, L):
    counts = np.bincount(y, minlength=L)
    starts = np.cumsum(np.r_[0, counts[:-1]])
    ends = starts + counts
    return counts, starts, ends


def _symmetric_block(n_block, p_edge):
    if n_block <= 1:
        return np.zeros((n_block, n_block), dtype=np.uint8)
    upper = np.triu(np.random.binomial(1, p_edge, size=(n_block, n_block)), k=1)
    return (upper + upper.T).astype(np.uint8)


def _generate_A_sorted(y, P_edge_sym):
    y = np.asarray(y, dtype=int)
    n = y.size
    L = P_edge_sym.shape[0]
    counts, starts, ends = _block_boundaries_from_sorted_y(y, L)
    A = np.zeros((n, n), dtype=np.uint8)

    for a in range(L):
        s, e = starts[a], ends[a]
        if counts[a] == 0:
            continue
        A[s:e, s:e] = _symmetric_block(counts[a], P_edge_sym[a, a])

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


def _generate_X_sorted(y, d, sigma, mus):
    y = np.asarray(y, dtype=int)
    L = mus.shape[0]
    counts, starts, ends = _block_boundaries_from_sorted_y(y, L)

    cov = (sigma ** 2) * np.eye(d)
    blocks = []
    for a in range(L):
        if counts[a] == 0:
            continue
        Xa = np.random.multivariate_normal(mean=mus[a], cov=cov, size=counts[a])
        blocks.append(Xa)
    return np.vstack(blocks) if blocks else np.zeros((0, d), dtype=float)


def generate_graph_data(n, p, P_edge, d, sigma, mus, sorted_labels=True):
    
    p = np.asarray(p, dtype=float)
    P_edge_sym = 0.5 * (np.asarray(P_edge, dtype=float)
                         + np.asarray(P_edge, dtype=float).T)
    y = _generate_labels(n, p, sorted_labels=sorted_labels)
    A = _generate_A_sorted(y, P_edge_sym)
    X = _generate_X_sorted(y, d, sigma, mus)
    return y, A, X

MLP_DEFAULTS  = dict(depth=2, hidden1=16, hidden2=16)
MLP_TRAIN     = dict(batch_size=64, n_epochs=50, lr=1e-3)
MMD_DEFAULTS  = dict(out_dim=128, hidden_dims=(128,), lr=1e-3, epochs=200, batch_size=512)
PLSA_DEFAULTS = dict(max_steps=3000, lr=5e-2, init_noise=1e-3,
                       progress_every=None, es_check_every=None, es_tol=1e-12,
                       es_patience=999, verbose=False, laplace=1.0)
CALIB_DEFAULTS = dict(max_iter=500, lr=1e-3)
MLLS_DEFAULTS  = dict(batch_size=256, max_iters=400, tol=1e-8, lr_w=1e-3)
MMD_LS_DEFAULTS = dict(lam_w=0.1, n_iters=0)

R_CLIP_MAX = 10.0
R_CLIP_MIN = 1.0 / R_CLIP_MAX


def apply_prob_temperature(F, smooth_T=1.0):
    if smooth_T == 1.0:
        return F
    F_safe = np.clip(F, 1e-12, 1.0) ** (1.0 / float(smooth_T))
    F_safe = F_safe / F_safe.sum(axis=1, keepdims=True)
    return F_safe

GDA_METHODS = {
    'a2gnn':   train_a2gnn,
    'dgsda':   train_dgsda,
    'specreg': train_specreg,
    'sagda':   train_sagda,
}
GDA_DEFAULTS = {
    'a2gnn':   dict(hidden=32, lr=1e-3, epochs=50, adv_weight=1e-3),
    'dgsda':   dict(hidden=32, lr=1e-3, epochs=50, alpha_coeff=0.05, beta_coeff=0.5,
                    gamma_coeff=0.05, bern_K=3),
    'specreg': dict(hidden=32, lr=1e-3, epochs=50, adv_weight=0.1, spec_weight=0.01),
    'sagda':   dict(hidden=32, lr=4e-3, epochs=50, weight_decay=3e-3,
                    adv_dim=40, ppmi_path_len=10),
}


def _fit_feature_aligner(X_src, y_src, X_tar, bundle, num_classes, device, seed,
                          lam, src_weights):
    
    set_seed(seed)
    g = train_feature_aligner(
        X_src=X_src, y_src=y_src, X_tgt=X_tar,
        num_classes=num_classes, device=device, lam=float(lam),
        src_weights=src_weights, **MMD_DEFAULTS, verbose=False,
    )
    return (
        transform_features(g, X_src, device),
        transform_features(g, bundle.src_val.X, device),
        transform_features(g, bundle.src_test.X, device),
        transform_features(g, X_tar, device),
    )


def _estimate_mmd_label_shift_weights(X_src, y_src, X_src_val, y_src_val, X_tar,
                                       num_classes, device, seed, lam_w):
    
    set_seed(seed)
    d = X_src.shape[1]
    model = SimpleMLP(input_dim=d, n_classes=num_classes, **MLP_DEFAULTS)
    Xtr, ytr = to_tensors(X_src, y_src)
    loader = make_loader(Xtr, ytr, batch_size=MLP_TRAIN['batch_size'], shuffle=True)
    Trainer(model, device=device, lr=MLP_TRAIN['lr']).train(
        loader, n_epochs=MLP_TRAIN['n_epochs'], progress_every=None)

    cal = Calibrator(model, device=device)
    Xval, yval = to_tensors(X_src_val, y_src_val)
    val_loader = make_loader(Xval, yval, batch_size=256, shuffle=False)
    logits, labels = cal.collect_logits(val_loader)
    T, bias = cal.fit_bcts_lbfgs(logits, labels, **CALIB_DEFAULTS)

    F_tar = predict_probs_calibrated(model, X_tar, device, batch_size=256,
                                      T=T, bias=bias).cpu().numpy()
    w_imp, _, _ = estimate_importance_weights_mlls(
        model=model, X_tar_np=X_tar, y_src_np=y_src, device=device,
        T=T, bias=bias, F_precomputed=F_tar, lam_w=lam_w, **MLLS_DEFAULTS,
    )
    return w_imp


def mmd_align(bundle, lam, device, *, seed=0,
              lam_w=MMD_LS_DEFAULTS['lam_w'],
              n_iters=MMD_LS_DEFAULTS['n_iters']):

    if lam is None or lam < 0:
        return bundle

    L = bundle.num_classes
    X_src, y_src = bundle.src_train.X, bundle.src_train.y
    X_tar = bundle.target.X

    X_src_al, X_src_val_al, X_src_test_al, X_tar_al = _fit_feature_aligner(
        X_src, y_src, X_tar, bundle, L, device, seed, lam, src_weights=None)

    for _ in range(n_iters):
        w_imp = _estimate_mmd_label_shift_weights(
            X_src_al, y_src, X_src_val_al, bundle.src_val.y, X_tar_al,
            L, device, seed, lam_w=lam_w)
        src_weights = w_imp[y_src]
        X_src_al, X_src_val_al, X_src_test_al, X_tar_al = _fit_feature_aligner(
            X_src, y_src, X_tar, bundle, L, device, seed, lam, src_weights=src_weights)

    bundle.src_train.X = X_src_al
    bundle.src_val.X   = X_src_val_al
    bundle.src_test.X  = X_src_test_al
    bundle.target.X    = X_tar_al
    bundle.num_features = bundle.src_train.X.shape[1]
    return bundle


def build_confusion_matrix_from_argmax(y_hats, y_true, L, alpha=1.0):
    C = np.zeros((L, L), dtype=np.float64)
    np.add.at(C, (np.asarray(y_hats, dtype=np.int64),
                   np.asarray(y_true, dtype=np.int64)), 1.0)
    C += float(alpha)
    return C / C.sum(axis=1, keepdims=True)


def train_source_pipeline(bundle, seed, device, *, calib='bcts',
                           mlp_config=None, lam_w=0.0):

    L = bundle.num_classes
    d = bundle.num_features
    cfg = {**MLP_DEFAULTS, **(mlp_config or {})}

    set_seed(seed)
    model = SimpleMLP(input_dim=d, n_classes=L, **cfg)
    Xtr, ytr = to_tensors(bundle.src_train.X, bundle.src_train.y)
    train_loader = make_loader(Xtr, ytr, batch_size=MLP_TRAIN['batch_size'], shuffle=True)
    Trainer(model, device=device, lr=MLP_TRAIN['lr']).train(
        train_loader, n_epochs=MLP_TRAIN['n_epochs'], progress_every=None)

    eval_loader = make_loader(Xtr, ytr, batch_size=MLP_TRAIN['batch_size'], shuffle=False)
    src_train_acc = Trainer(model, device=device, lr=MLP_TRAIN['lr']).evaluate(eval_loader)

    Xval, yval = to_tensors(bundle.src_val.X, bundle.src_val.y)
    val_loader = make_loader(Xval, yval, batch_size=256, shuffle=False)
    cal = Calibrator(model, device=device)
    logits, labels = cal.collect_logits(val_loader)

    if calib == 'bcts':
        T, bias = cal.fit_bcts_lbfgs(logits, labels, **CALIB_DEFAULTS)
    elif calib == 'isotonic':
        from sklearn.isotonic import IsotonicRegression
        probs_val = torch.softmax(logits, dim=1).numpy()
        labels_np = labels.numpy()
        iso_models = []
        for c in range(L):
            ir = IsotonicRegression(out_of_bounds='clip')
            ir.fit(probs_val[:, c], (labels_np == c).astype(float))
            iso_models.append(ir)
        T, bias = 1.0, None
        cal._iso_models = iso_models
    else:
        raise ValueError(f"Unknown calib: {calib}")

    Xte, yte = to_tensors(bundle.src_test.X, bundle.src_test.y)
    test_loader = make_loader(Xte, yte, batch_size=256, shuffle=False)
    if calib == 'bcts':
        ece = cal.evaluate_loader(test_loader, T=T, bias=bias)[2]
    else:
        ece = float('nan')

    def _calibrate_probs(X_np):
        if calib == 'isotonic':
            model.eval()
            with torch.no_grad():
                logits = []
                for s in range(0, len(X_np), 256):
                    e = min(s+256, len(X_np))
                    Xb = torch.as_tensor(X_np[s:e], dtype=torch.float32, device=device)
                    logits.append(model(Xb).cpu())
                logits = torch.cat(logits, dim=0)
                probs = torch.softmax(logits, dim=1).numpy()
                F = np.zeros_like(probs)
                for c in range(L):
                    F[:, c] = cal._iso_models[c].transform(probs[:, c])
                return F / F.sum(axis=1, keepdims=True).clip(1e-12)
        return predict_probs_calibrated(model, X_np, device,
                                         batch_size=256, T=T, bias=bias).cpu().numpy()

    F_pre = _calibrate_probs(bundle.target.X)
    F_val = _calibrate_probs(bundle.src_val.X)

    w_imp, _, _ = estimate_importance_weights_mlls(
        model=model, X_tar_np=bundle.target.X, y_src_np=bundle.src_train.y,
        device=device, T=T, bias=bias,
        F_precomputed=F_pre, lam_w=lam_w,
        **MLLS_DEFAULTS,
    )

    y_hat_val = F_val.argmax(axis=1)
    C_hat = build_confusion_matrix_from_argmax(y_hat_val, bundle.src_val.y, L, alpha=1.0)
    y_hat_tar = F_pre.argmax(axis=1)
    F_CM = C_hat[y_hat_tar]

    pseudo_acc = float((y_hat_tar == bundle.target.y).mean())

    return dict(
        model=model, T=T, bias=bias,
        F_pre=F_pre, F_CM=F_CM, C_hat=C_hat, w_imp=w_imp,
        src_train_acc=src_train_acc, ece=ece, pseudo_acc=pseudo_acc,
    )


def _stratified_target_val_test_mask(y_long, val_ratio=0.2, seed=0):
    
    y = y_long.view(-1).cpu().numpy()
    rng = np.random.RandomState(int(seed))
    val_idx, test_idx = [], []
    for c in np.unique(y):
        idx_c = np.where(y == c)[0].copy()
        rng.shuffle(idx_c)
        k = max(1, int(len(idx_c) * val_ratio))
        val_idx.append(idx_c[:k])
        test_idx.append(idx_c[k:])
    val_idx = np.concatenate(val_idx)
    test_idx = np.concatenate(test_idx)
    n = len(y)
    val_mask = torch.zeros(n, dtype=torch.bool, device=y_long.device)
    test_mask = torch.zeros(n, dtype=torch.bool, device=y_long.device)
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return val_mask, test_mask


def run_downstream_gda(bundle, R, gda_method, seed, device, *, num_samples=1,
                        target_val_ratio=0.2, class_weights=None):
    A_src_csr = csr_matrix(bundle.src_train.A)

    if R is not None:
        set_seed(seed)
        B = average_adjusted_B(A_src_csr, bundle.src_train.y, w_mat=R, num_samples=num_samples)
        B_csr = csr_matrix(B)
        if num_samples > 1:
            source_data = prepare_graph_data(B_csr, bundle.src_train.X, bundle.src_train.y,
                                              use_matrix_values_as_weight=True).to(device)
        else:
            source_data = prepare_graph_data(B_csr, bundle.src_train.X, bundle.src_train.y).to(device)
    else:
        source_data = prepare_graph_data(A_src_csr, bundle.src_train.X, bundle.src_train.y).to(device)

    target_data = prepare_graph_data(csr_matrix(bundle.target.A),
                                      bundle.target.X, bundle.target.y).to(device)
    
    src_val_data = prepare_graph_data(csr_matrix(bundle.src_val.A),
                                       bundle.src_val.X, bundle.src_val.y).to(device)

    if class_weights is not None:
        class_weights = torch.as_tensor(class_weights, dtype=torch.float32, device=device)

    set_seed(seed)
    train_fn = GDA_METHODS[gda_method]
    result = train_fn(source_data=source_data, target_data=target_data,
                      num_features=bundle.num_features, num_classes=bundle.num_classes,
                      device=device, src_val_data=src_val_data, class_weights=class_weights,
                      **GDA_DEFAULTS[gda_method])
    target_all = dict(result['target_all'])

    if 'source_val' in result:
        for k, v in result['source_val'].items():
            target_all[f'srcval_{k}'] = v

    if 'target_logits' in result:
        from utils.metrics import compute_metrics as _cm
        tl = result['target_logits']
        ty = target_data.y
        val_mask, test_mask = _stratified_target_val_test_mask(
            ty, val_ratio=target_val_ratio, seed=seed)
        val_m = _cm(tl[val_mask], ty[val_mask])
        test_m = _cm(tl[test_mask], ty[test_mask])
        for k, v in val_m.items():
            target_all[f'val_{k}'] = v
        for k, v in test_m.items():
            target_all[f'test_{k}'] = v
    return target_all


CSBM_MLP_DEFAULTS    = dict(depth=2, hidden1=32)
CSBM_TRAIN_DEFAULTS  = dict(batch_size=256, lr=1e-3, n_epochs=60, progress_every=None)
CSBM_CALIB_DEFAULTS  = dict(max_iter=1000, lr=0.01)

CSBM_PLSA_HP = dict(
    max_steps=3000, lr=5e-2, init_noise=1e-3,
    pred_batch_size=256, progress_every=None, es_check_every=20,
    es_tol=1e-8, min_steps=50, verbose=False, es_patience=5,
    laplace=1.0, lam=0.0,
)
CSBM_PAIR_HP = dict(lam=0.0, solver="SCS", symmetrize=True,
                    pred_batch_size=256, verbose=False)


def csbm_set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def csbm_rel_error_upper_tri(A, B):
    num = np.linalg.norm(np.triu(A - B))
    den = np.linalg.norm(np.triu(A))
    return float(num / (den + 1e-12))


def csbm_build_data(L, n, prior, P_edge_source, P_edge_target, mus, d, sigma):
    
    out = {}
    for tag in ("src_tr", "src_te", "src_va"):
        y, A, X = generate_graph_data(n, prior, P_edge_source, d, sigma, mus)
        out[tag] = dict(y=y, A=A, X=X)
    y, A, X = generate_graph_data(n, prior, P_edge_target, d, sigma, mus)
    out["tar"] = dict(y=y, A=A, X=X)
    return out


def csbm_train_source_model(X_tr, y_tr, L, d, device):
    model = SimpleMLP(input_dim=d, n_classes=L, **CSBM_MLP_DEFAULTS).to(device)
    Xt, yt = to_tensors(X_tr, y_tr)
    loader = make_loader(Xt.to(device), yt.to(device),
                         batch_size=CSBM_TRAIN_DEFAULTS["batch_size"], shuffle=True)
    Trainer(model, device=device, lr=CSBM_TRAIN_DEFAULTS["lr"]).train(
        loader, n_epochs=CSBM_TRAIN_DEFAULTS["n_epochs"],
        progress_every=CSBM_TRAIN_DEFAULTS["progress_every"],
    )
    return model


def csbm_calibrate_bcts(model, X_va, y_va, device):
    Xv, yv = to_tensors(X_va, y_va)
    loader = make_loader(Xv, yv, batch_size=256, shuffle=False)
    calib = Calibrator(model, device=device)
    logits, labels = calib.collect_logits(loader)
    T_bct, b_bct = calib.fit_bcts_lbfgs(logits, labels, **CSBM_CALIB_DEFAULTS)
    return T_bct, b_bct


def csbm_fit_plsa(data, model, device, T, bias):
    pe = PLSA(**CSBM_PLSA_HP)
    pe.fit(data["src_tr"]["A"], data["src_tr"]["y"],
           data["tar"]["A"], data["tar"]["X"],
           model, device, T=T, bias=bias)
    W_E = pe.get_weights()
    e_S = edge_rate_uppertri(data["src_tr"]["A"])
    e_T = edge_rate_uppertri(data["tar"]["A"])
    return W_E * (float(e_T) / max(float(e_S), 1e-12))


def csbm_fit_pairalign(data, model, device):
    pa = PairAlign(**CSBM_PAIR_HP)
    pa.fit(data["src_tr"]["A"], data["src_tr"]["y"],
           data["tar"]["A"], data["src_tr"]["X"], data["tar"]["X"],
           model, device)
    _, R_joint = pa.get_weights()
    return R_joint, pa.status


def csbm_run_one(L, n, seed, prior, P_edge_source, P_edge_target,
                 methods, device, *, d=20, sigma=1.0, verbose=True):
    csbm_set_seed(seed)
    rng_mus = np.random.default_rng(seed)
    mus = rng_mus.standard_normal((L, d)) / np.sqrt(d)

    t_start = time.time()
    if verbose:
        print(f"  [{time.time()-t_start:5.1f}s] generating data n={n}...", flush=True)
    data = csbm_build_data(L, n, prior, P_edge_source, P_edge_target,
                           mus, d, sigma)
    if verbose:
        print(f"  [{time.time()-t_start:5.1f}s] training source MLP...", flush=True)
    model = csbm_train_source_model(data["src_tr"]["X"], data["src_tr"]["y"],
                                    L, d, device)
    if verbose:
        print(f"  [{time.time()-t_start:5.1f}s] calibrating...", flush=True)
    T_bct, b_bct = csbm_calibrate_bcts(model, data["src_va"]["X"],
                                       data["src_va"]["y"], device)
    if verbose:
        print(f"  [{time.time()-t_start:5.1f}s] calibration done "
              f"(T={float(T_bct):.3f})", flush=True)

    R_true = np.asarray(P_edge_target, dtype=float) / np.clip(
        np.asarray(P_edge_source, dtype=float), 1e-12, None
    )

    records = []
    for mname in methods:
        t0 = time.time()
        try:
            if mname in ("plsa", "plsa_cal"):
                R_est = csbm_fit_plsa(data, model, device,
                                      T=float(T_bct), bias=b_bct)
                status = "ok"
            elif mname == "plsa_uncal":
                R_est = csbm_fit_plsa(data, model, device, T=1.0, bias=None)
                status = "ok"
            elif mname == "pairalign":
                R_est, status = csbm_fit_pairalign(data, model, device)
            else:
                continue
            err = csbm_rel_error_upper_tri(R_true, R_est)
        except Exception as e:
            err = None
            status = f"error: {type(e).__name__}: {e}"
        rec = dict(seed=int(seed), method=mname, rel_error=err,
                   status=status, runtime_s=round(time.time() - t0, 3))
        if verbose:
            err_str = "None" if err is None else f"{err:.4e}"
            print(f"  {mname:11s} rel_err={err_str}  "
                  f"({rec['runtime_s']}s, {status})")
        records.append(rec)
    return records


CSBM_SOURCE_PRIORS = {2: [0.5, 0.5], 3: [1/3, 1/3, 1/3]}
CSBM_SOURCE_P_EDGE = {
    2: [[0.02, 0.005], [0.005, 0.02]],
    3: [[0.02, 0.005, 0.005], [0.005, 0.02, 0.005], [0.005, 0.005, 0.02]],
}
CSBM_CALIB_Q_SRC   = [[0.02, 0.005], [0.005, 0.02]]
CSBM_CALIB_X_VALS  = [0.01, 0.015, 0.02, 0.025]
CSBM_ROMAN_UNIFORM = ["I", "II", "III", "IV"]
CSBM_ROMAN_IMBAL   = ["V", "VI", "VII", "VIII"]


def csbm_fig1_grid(L=2):
    if L == 2:
        x_values = [0.010, 0.015, 0.020, 0.025, 0.030]
        panels = [
            ("a", 0.0025, 0.010),
            ("b", 0.0025, 0.015),
            ("c", 0.0075, 0.010),
            ("d", 0.0075, 0.015),
        ]
        out = []
        for label, q, p2 in panels:
            for x in x_values:
                Pt = [[x, q], [q, p2]]
                out.append((f"{label}_x{x}", Pt))
        return out
    if L == 3:
        return [("a_x0.020",
                 [[0.020, 0.005, 0.005],
                  [0.005, 0.020, 0.005],
                  [0.005, 0.005, 0.020]])]
    raise ValueError(f"L={L} not supported")


def csbm_fig2_grid(L=2):
    if L == 2:
        Pt = [[0.03, 0.0075], [0.0075, 0.01]]
    else:
        Pt = [[0.03, 0.005, 0.0025],
              [0.005, 0.015, 0.001],
              [0.0025, 0.001, 0.01]]
    ns = [3000, 5000, 7000, 10000, 13000, 15000]
    return Pt, ns


def csbm_calib_settings():
    out = []
    for r, x in zip(CSBM_ROMAN_UNIFORM, CSBM_CALIB_X_VALS):
        out.append(dict(label=r, prior_label="uniform", prior=[0.5, 0.5],
                        x=x, Q_tar=[[x, 0.0025], [0.0025, 0.01]]))
    for r, x in zip(CSBM_ROMAN_IMBAL, CSBM_CALIB_X_VALS):
        out.append(dict(label=r, prior_label="imbalanced", prior=[0.2, 0.8],
                        x=x, Q_tar=[[x, 0.0025], [0.0025, 0.01]]))
    return out


def _cli_csbm(args, device):
    import argparse, json
    from pathlib import Path

    prior = CSBM_SOURCE_PRIORS[args.L]
    P_edge_source = CSBM_SOURCE_P_EDGE[args.L]

    out_dir = Path(__file__).resolve().parent / "csbm_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.out is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.out = str(out_dir / f"{args.mode}_L{args.L}_{ts}.jsonl")

    print(f"Device: {device}")
    print(f"Mode: {args.mode}  L={args.L}  seeds={args.seeds}  methods={args.methods}")
    print(f"Output: {args.out}")

    with open(args.out, "w") as fout:
        if args.mode == "fig1":
            for panel, Pt in csbm_fig1_grid(args.L):
                setting_id = f"fig1_L{args.L}_panel{panel}"
                print(f"\n=== {setting_id}  P_edge_target={Pt}  n={args.n} ===")
                for seed in args.seeds:
                    print(f"\n[seed={seed}]")
                    recs = csbm_run_one(args.L, args.n, seed, prior,
                                        P_edge_source, Pt, args.methods, device)
                    for r in recs:
                        r.update(setting_id=setting_id, panel=panel,
                                 L=args.L, n=int(args.n))
                        fout.write(json.dumps(r) + "\n")
                    fout.flush()
        else:
            Pt, ns = csbm_fig2_grid(args.L)
            for n in ns:
                setting_id = f"fig2_L{args.L}_n{n}"
                print(f"\n=== {setting_id}  P_edge_target={Pt} ===")
                for seed in args.seeds:
                    print(f"\n[seed={seed}]")
                    recs = csbm_run_one(args.L, n, seed, prior,
                                        P_edge_source, Pt, args.methods, device)
                    for r in recs:
                        r.update(setting_id=setting_id, panel="fig2",
                                 L=args.L, n=int(n))
                        fout.write(json.dumps(r) + "\n")
                    fout.flush()
    print(f"\nDone. Wrote: {args.out}")


def _cli_csbm_calib(args, device):
    import json
    from pathlib import Path

    settings = csbm_calib_settings()
    if args.settings:
        wanted = set(args.settings)
        settings = [s for s in settings if s["label"] in wanted]
        if not settings:
            raise SystemExit(
                f"No settings match {args.settings}; choose from I..VIII")

    out_dir = Path(__file__).resolve().parent / "csbm_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.out is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.out = str(out_dir / f"calib_ablation_{ts}.jsonl")

    print(f"Device:   {device}")
    print(f"Settings: {[s['label'] for s in settings]}")
    print(f"Methods:  {args.methods}")
    print(f"Seeds:    {args.seeds}")
    print(f"Output:   {args.out}")

    N_NODES = 5000
    with open(args.out, "w") as fout:
        for s in settings:
            print(f"\n=== Setting {s['label']}  prior={s['prior']}  "
                  f"x={s['x']}  Q_tar={s['Q_tar']} ===")
            for seed in args.seeds:
                print(f"\n[setting {s['label']}, seed={seed}]")
                recs = csbm_run_one(L=2, n=N_NODES, seed=seed,
                                    prior=s["prior"],
                                    P_edge_source=CSBM_CALIB_Q_SRC,
                                    P_edge_target=s["Q_tar"],
                                    methods=args.methods, device=device)
                for r in recs:
                    r.update(setting_id=f"setting_{s['label']}",
                             setting_label=s["label"],
                             prior_label=s["prior_label"],
                             x=s["x"], n=N_NODES, L=2)
                    fout.write(json.dumps(r) + "\n")
                fout.flush()
    print(f"\nDone. Wrote: {args.out}")


def _data_root():
    from pathlib import Path
    return str(Path(__file__).resolve().parent / "data")


REAL_DATASETS = {

    "acm_dblp": ("load_acm_dblp_data",
                 lambda: dict(base=_data_root(), src_train_ratio=0.7), True),
    "cora":     ("load_cora_data",
                 lambda: dict(base_dir=f"{_data_root()}/cora_data",
                              prefix="good_cora", train_ratio=0.7), False),
    "twitch":   ("load_twitch_data",
                 lambda: dict(base_dir=f"{_data_root()}/twitch",
                              src_train_ratio=0.7), True),
    "airport":  ("load_airport_data",
                 lambda: dict(base_dir=f"{_data_root()}/airport_data",
                              src_train_ratio=0.7), True),
}


def _load_real_bundle(args):
    from utils import data as data_mod
    if args.dataset not in REAL_DATASETS:
        raise SystemExit(f"Unknown dataset {args.dataset}; "
                         f"choose from {list(REAL_DATASETS)}")
    loader_name, defaults_fn, supports_src_tar = REAL_DATASETS[args.dataset]
    loader = getattr(data_mod, loader_name)

    kwargs = defaults_fn()
    if supports_src_tar:
        if not args.src or not args.tar:
            raise SystemExit(f"--src and --tar required for {args.dataset}")
        kwargs.update(src=args.src, tar=args.tar)
        label = f"{args.dataset}:{args.src}->{args.tar}"
    else:
        if args.k_labels is not None:
            kwargs["k_labels"] = args.k_labels
        label = f"cora_L{kwargs.get('k_labels', 6)}"

    bundle = loader(**kwargs)
    return bundle, label


def _hp_from_args(args):
    hp = dict(
        mmd_lam=args.mmd_lam,
        mmd_n_iters=args.mmd_n_iters,
        calib=args.calib,
        lam_w=args.lam_w,
        num_samples=args.num_samples,

        lam_plsa=args.lam_plsa,
        smooth_T=args.smooth_T,

        lam_pair=args.lam_pair,
        solver=args.solver,

        lam_blend=args.lam_blend,
    )
    return {k: v for k, v in hp.items() if v is not None}


def _cli_real(args, device):
    import gc
    import json
    from pathlib import Path

    bundle, label = _load_real_bundle(args)
    hp = _hp_from_args(args)

    out_dir = Path(__file__).resolve().parent / "real_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.out is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.out = str(out_dir / f"real_{args.dataset}_{ts}.jsonl")

    print(f"Device:   {device}")
    print(f"Dataset:  {label}  (n_classes={bundle.num_classes}, "
          f"d={bundle.num_features})")
    print(f"Methods:  {args.methods}")
    print(f"Models:   {args.models}")
    print(f"Seeds:    {args.seeds}")
    print(f"HP:       {hp}")
    print(f"IW-ERM:   {bool(args.iw_erm)}")
    print(f"Output:   {args.out}")

    del bundle
    with open(args.out, "w") as fout:
        for seed in args.seeds:
            cache = _RunCache(lambda: _load_real_bundle(args)[0], seed, device)
            for method in args.methods:
                for model in args.models:
                    print(f"\n=== {label}  method={method}  model={model}  "
                          f"seed={seed} ===")
                    t0 = time.time()
                    try:
                        result = _run_one(cache, method=method, model=model,
                                          hp=hp, seed=seed, device=device,
                                          iw_erm=args.iw_erm)
                        acc = float(result.get("acc", float("nan")))
                        val_acc = float(result.get("val_acc", float("nan")))
                        test_acc = float(result.get("test_acc", float("nan")))
                        status = "ok"
                    except Exception as e:
                        acc = val_acc = test_acc = float("nan")
                        status = f"error: {type(e).__name__}: {e}"
                        result = {}
                    rec = dict(dataset=args.dataset, label=label,
                               src=args.src, tar=args.tar,
                               method=method, model=model, seed=int(seed),
                               acc=acc, val_acc=val_acc, test_acc=test_acc,
                               status=status,
                               runtime_s=round(time.time() - t0, 3),
                               hp=hp, iw_erm=bool(args.iw_erm))
                    fout.write(json.dumps(rec) + "\n")
                    fout.flush()
                    print(f"  all={acc:.4f}  val={val_acc:.4f}  test={test_acc:.4f}  "
                          f"({rec['runtime_s']}s, {status})")
                    del result
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    print(f"\nDone. Wrote: {args.out}")


class _RunCache:
    def __init__(self, loader_fn, seed, device):
        self._loader = loader_fn
        self.seed = seed
        self.device = device
        self._bundle, self._pipe = {}, {}
        self._plsa, self._pair = {}, {}
        self._strurw = {}

    def get_bundle(self, mmd_lam, n_iters=0):
        key = (mmd_lam, n_iters)
        if key not in self._bundle:
            set_seed(self.seed)
            b = self._loader()
            b = mmd_align(b, mmd_lam, self.device, seed=self.seed, n_iters=n_iters)
            self._bundle[key] = b
        return self._bundle[key]

    def get_pipe(self, mmd_lam, n_iters, calib, lam_w):
        key = (mmd_lam, n_iters, calib, lam_w)
        if key not in self._pipe:
            b = self.get_bundle(mmd_lam, n_iters)
            self._pipe[key] = train_source_pipeline(
                b, self.seed, self.device, calib=calib, lam_w=lam_w)
        return self._pipe[key]

    def get_plsa_R_raw(self, mmd_lam, n_iters, calib, lam_w, lam_plsa, smooth_T):
        key = (mmd_lam, n_iters, calib, lam_w, lam_plsa, smooth_T)
        if key not in self._plsa:
            b = self.get_bundle(mmd_lam, n_iters)
            pipe = self.get_pipe(mmd_lam, n_iters, calib, lam_w)
            F_soft = apply_prob_temperature(pipe["F_pre"], smooth_T)
            F_mix = F_soft
            A_s = b.src_train.A.toarray()
            A_t = b.target.A.toarray()
            set_seed(self.seed)
            plsa = PLSA(**PLSA_DEFAULTS, lam=lam_plsa)
            plsa.fit(A_s, b.src_train.y, A_t, b.target.X, pipe["model"],
                     self.device, T=pipe["T"], bias=pipe["bias"],
                     F_precomputed=F_mix)
            W = plsa.get_weights()
            p_S = edge_rate_uppertri(A_s)
            p_T = edge_rate_uppertri(A_t)
            R_joint = W * (float(p_T) / max(float(p_S), 1e-12))
            self._plsa[key] = R_joint / np.outer(pipe["w_imp"], pipe["w_imp"])
        return self._plsa[key]

    def get_pair_R_raw(self, mmd_lam, n_iters, calib, lam_w, lam_pair,
                       solver="CLARABEL"):
        key = (mmd_lam, n_iters, calib, lam_w, lam_pair, solver)
        if key not in self._pair:
            b = self.get_bundle(mmd_lam, n_iters)
            pipe = self.get_pipe(mmd_lam, n_iters, calib, lam_w)
            A_s = b.src_train.A.toarray()
            A_t = b.target.A.toarray()
            pair = PairAlign(lam=lam_pair, solver=solver,
                             symmetrize=True, pred_batch_size=256)
            pair.fit(A_s, b.src_train.y, A_t,
                     b.src_train.X, b.target.X, pipe["model"], self.device)
            _, R_raw = pair.get_weights()
            self._pair[key] = R_raw / np.outer(pipe["w_imp"], pipe["w_imp"])
        return self._pair[key]

    def get_strurw_R_raw(self, mmd_lam, n_iters, calib, lam_w):
        key = (mmd_lam, n_iters)
        if key not in self._strurw:
            b = self.get_bundle(mmd_lam, n_iters)
            pipe = self.get_pipe(mmd_lam, n_iters, calib, lam_w)
            A_s = b.src_train.A.toarray()
            A_t = b.target.A.toarray()
            sw = StruRW(symmetrize=True, pred_batch_size=256)
            sw.fit(A_s, b.src_train.y, A_t,
                   b.src_train.X, b.target.X, pipe["model"], self.device)
            self._strurw[key] = sw.get_weights()
        return self._strurw[key]


def _run_one(cache, *, method, model, hp, seed, device, iw_erm=False):
    mmd_lam = hp.get("mmd_lam")
    n_iters = int(hp.get("mmd_n_iters", 0))
    b = cache.get_bundle(mmd_lam, n_iters)

    calib = hp.get("calib", "bcts")
    lam_w = hp.get("lam_w", 0.0)
    class_weights = None
    if iw_erm:
        # Share the final MLLS estimate across all backbones and graph variants.
        class_weights = cache.get_pipe(mmd_lam, n_iters, calib, lam_w)["w_imp"]

    if model == "plain":
        return run_downstream_gda(b, R=None, gda_method=method, seed=seed,
                                  device=device, class_weights=class_weights,
                                  num_samples=hp.get("num_samples", 1))

    if model == "plsa":
        R_raw = cache.get_plsa_R_raw(mmd_lam, n_iters, calib, lam_w,
                                     hp.get("lam_plsa", 0.0),
                                     hp.get("smooth_T", 1.0))
    elif model == "pairalign":
        R_raw = cache.get_pair_R_raw(mmd_lam, n_iters, calib, lam_w,
                                     hp.get("lam_pair", 0.0),
                                     hp.get("solver", "CLARABEL"))
    elif model == "strurw":
        R_raw = cache.get_strurw_R_raw(mmd_lam, n_iters, calib, lam_w)
    else:
        raise ValueError(f"Unknown model: {model}")

    lam_blend = hp.get("lam_blend", 1.0)
    R = (1.0 - lam_blend) + lam_blend * R_raw
    R = np.clip(R, R_CLIP_MIN, R_CLIP_MAX)
    return run_downstream_gda(b, R, gda_method=method, seed=seed,
                              device=device, class_weights=class_weights,
                              num_samples=hp.get("num_samples", 1))


def _build_cli():
    import argparse
    p = argparse.ArgumentParser(
        description="Unified pipeline CLI: csbm, csbm-calib, real.")
    p.add_argument("--device", default="cuda", help="cuda / cpu")
    sub = p.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("csbm",
                        help="CSBM Fig 1 (q-sweep) or Fig 2 (n-sweep)")
    p1.add_argument("--mode", choices=["fig1", "fig2"], required=True)
    p1.add_argument("--L", type=int, default=2)
    p1.add_argument("--n", type=int, default=5000,
                    help="node count for fig1 mode")
    p1.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p1.add_argument("--methods", nargs="+", default=["pairalign", "plsa"])
    p1.add_argument("--out", default=None, help="output JSONL path")

    p2 = sub.add_parser("csbm-calib",
                        help="CSBM Table 5 calibration ablation")
    p2.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    p2.add_argument("--methods", nargs="+",
                    default=["plsa_cal", "plsa_uncal", "pairalign"])
    p2.add_argument("--settings", nargs="+", default=None,
                    help="Subsets")
    p2.add_argument("--out", default=None, help="Output JSONL path.")

    p3 = sub.add_parser("real",
                        help="Real-data benchmark (ACM-DBLP, Cora, Twitch, Airport)")
    p3.add_argument("--dataset", required=True,
                    choices=list(REAL_DATASETS),
                    help="acm_dblp / cora / twitch / airport")
    p3.add_argument("--src", default=None,
                    help="Source domain (acm/dblp, DE/ENGB, usa/brazil/europe)")
    p3.add_argument("--tar", default=None, help="Target domain")
    p3.add_argument("--k-labels", type=int, default=None,
                    help="cora only: top-k label filter (6 or 8)")
    p3.add_argument("--methods", nargs="+",
                    default=["a2gnn", "specreg", "dgsda", "sagda"],
                    help="GDA backbones to run")
    p3.add_argument("--models", nargs="+",
                    default=["plain", "plsa", "pairalign", "strurw"],
                    help="reweighting models to run")
    p3.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p3.add_argument("--out", default=None, help="Output JSONL path.")

    p3.add_argument("--mmd-lam", type=float, default=None)
    p3.add_argument("--mmd-n-iters", type=int, default=None,
                    help="Number of iterative feature-alignment and label-shift refinement rounds.")
    p3.add_argument("--calib", default=None, choices=["bcts", "isotonic"])
    p3.add_argument("--lam-w", type=float, default=None)
    p3.add_argument("--lam-plsa", type=float, default=None)
    p3.add_argument("--smooth-T", type=float, default=None)
    p3.add_argument("--lam-pair", type=float, default=None)
    p3.add_argument("--solver", default=None,
                    help="CVXPY solver for PairAlign (CLARABEL / SCS)")
    p3.add_argument("--lam-blend", type=float, default=None)
    p3.add_argument("--num-samples", type=int, default=None)
    p3.add_argument("--iw-erm", action="store_true",
                    help="Weight the source classification loss of the GDA backbone "
                         "by the MLLS label-shift weights, for all models including plain.")

    return p


def main():
    args = _build_cli().parse_args()
    device = torch.device(args.device if (args.device == "cuda" and
                                           torch.cuda.is_available())
                          else "cpu")
    if args.cmd == "csbm":
        _cli_csbm(args, device)
    elif args.cmd == "csbm-calib":
        _cli_csbm_calib(args, device)
    elif args.cmd == "real":
        _cli_real(args, device)
    else:
        raise SystemExit(f"Unknown subcommand: {args.cmd}")


if __name__ == "__main__":
    main()
