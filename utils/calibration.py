import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class BCTS(nn.Module):
    def __init__(self, num_classes, init_T=1.0):
        super().__init__()
        self.log_T = nn.Parameter(torch.tensor([np.log(np.exp(init_T) - 1.0)], dtype=torch.float32))
        self.bias  = nn.Parameter(torch.zeros(num_classes, dtype=torch.float32))

    def temperature(self):
        return torch.nn.functional.softplus(self.log_T) + 1e-6

    def forward(self, logits):
        T = self.temperature()
        return logits / T + self.bias


class Calibrator:
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or torch.device("cpu")

    @torch.no_grad()
    def collect_logits(self, loader):
        self.model.eval()
        all_logits, all_labels = [], []
        for Xb, yb in loader:
            Xb = Xb.to(self.device)
            logits = self.model(Xb).detach().cpu()
            all_logits.append(logits)
            all_labels.append(yb.detach().cpu())
        logits_cpu = torch.cat(all_logits, dim=0)
        labels_cpu = torch.cat(all_labels, dim=0)
        return logits_cpu, labels_cpu

    def fit_ts_lbfgs(self, logits_cpu, labels_cpu, max_iter=50, lr=0.01):
        class _TS(nn.Module):
            def __init__(self):
                super().__init__()
                self.log_T = nn.Parameter(torch.tensor([np.log(np.exp(1.0) - 1.0)], dtype=torch.float32))
            def temperature(self):
                return torch.nn.functional.softplus(self.log_T) + 1e-6
            def forward(self, logits):
                return logits / self.temperature()

        ts = _TS()
        nll = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([ts.log_T], lr=lr, max_iter=max_iter)

        logits_cpu = logits_cpu.clone().detach()
        labels_cpu = labels_cpu.clone().detach()

        def closure():
            optimizer.zero_grad()
            scaled = ts(logits_cpu)
            loss = nll(scaled, labels_cpu)
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            T_opt = float(ts.temperature().item())
        return T_opt

    def fit_bcts_lbfgs(self, logits_cpu, labels_cpu, max_iter=100, lr=0.05):
        N, C = logits_cpu.shape
        bcts = BCTS(num_classes=C, init_T=1.0)
        nll = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS(list(bcts.parameters()), lr=lr, max_iter=max_iter)

        logits_cpu = logits_cpu.clone().detach()
        labels_cpu = labels_cpu.clone().detach()

        def closure():
            optimizer.zero_grad()
            scaled = bcts(logits_cpu)
            loss = nll(scaled, labels_cpu)
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            T_opt = float(bcts.temperature().item())
            b_opt = bcts.bias.detach().cpu().numpy()
        return T_opt, b_opt

    @torch.no_grad()
    def apply_ts(self, logits, T):
        return torch.softmax(logits / T, dim=1)

    @torch.no_grad()
    def apply_bcts(self, logits, T, bias):
        bias_t = torch.as_tensor(bias, dtype=logits.dtype, device=logits.device)
        return torch.softmax(logits / T + bias_t, dim=1)

    @staticmethod
    def _ece_streaming_update(bin_totals, bin_conf_sums, bin_corr_sums, confidences, correct, n_bins):
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        for b in range(n_bins):
            lo, hi = bins[b], bins[b + 1]
            in_bin = (confidences > lo) & (confidences <= hi)
            cnt = in_bin.sum()
            if cnt > 0:
                bin_totals[b]    += cnt
                bin_conf_sums[b] += confidences[in_bin].sum()
                bin_corr_sums[b] += correct[in_bin].sum()

    @staticmethod
    def _ece_streaming_finalize(bin_totals, bin_conf_sums, bin_corr_sums):
        N = bin_totals.sum()
        if N == 0:
            return 0.0
        ece = 0.0
        for b in range(len(bin_totals)):
            if bin_totals[b] > 0:
                avg_conf = bin_conf_sums[b] / bin_totals[b]
                avg_acc  = bin_corr_sums[b] / bin_totals[b]
                ece += (bin_totals[b] / N) * abs(avg_conf - avg_acc)
        return float(ece)

    @torch.no_grad()
    def evaluate_loader(self, loader, T=1.0, bias=None, n_bins=15):
        self.model.eval()
        ce = nn.CrossEntropyLoss(reduction='sum')

        bin_totals    = np.zeros(n_bins, dtype=np.int64)
        bin_conf_sums = np.zeros(n_bins, dtype=np.float64)
        bin_corr_sums = np.zeros(n_bins, dtype=np.float64)

        bias_t = None
        if bias is not None:
            bias_t = torch.tensor(bias, dtype=torch.float32, device=self.device)

        total = 0
        nll_sum = 0.0
        correct_sum = 0

        for Xb, yb in loader:
            Xb = Xb.to(self.device)
            yb = yb.to(self.device)

            logits = self.model(Xb) / T
            if bias_t is not None:
                logits = logits + bias_t

            probs  = torch.softmax(logits, dim=1)
            preds  = probs.argmax(dim=1)

            nll_sum     += ce(logits, yb).item()
            correct_sum += (preds == yb).sum().item()
            total       += yb.size(0)

            conf = probs.max(dim=1).values.detach().cpu().numpy()
            corr = (preds == yb).detach().cpu().numpy().astype(np.float32)
            self._ece_streaming_update(bin_totals, bin_conf_sums, bin_corr_sums, conf, corr, n_bins)

        acc = correct_sum / total
        nll = nll_sum / total
        ece = self._ece_streaming_finalize(bin_totals, bin_conf_sums, bin_corr_sums)
        return acc, nll, ece