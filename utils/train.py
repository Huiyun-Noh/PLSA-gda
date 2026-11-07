import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from tqdm import tqdm

class Trainer:
    def __init__(self, model, device=None, lr=1e-3):
        self.model = model
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_history = []

    def train(self, train_loader, n_epochs=50, progress_every=10):
        for epoch in range(n_epochs):
            self.model.train()
            epoch_loss = 0.0

            use_bar = (progress_every and ((epoch + 1) % progress_every == 0))
            iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}") if use_bar else train_loader

            for Xb, yb in iterator:
                Xb = Xb.to(self.device)
                yb = yb.to(self.device)

                logits = self.model(Xb)
                loss = self.criterion(logits, yb)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * Xb.size(0)
                if use_bar:
                    iterator.set_postfix(loss=loss.item())

            avg = epoch_loss / len(train_loader.dataset)
            self.loss_history.append(avg)

        return self.loss_history

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        correct, total = 0, 0
        for Xb, yb in loader:
            Xb = Xb.to(self.device)
            yb = yb.to(self.device)
            preds = self.model(Xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total   += yb.size(0)
        return correct / total

class GNNTrainer:
    def __init__(self, model, device, lr=1e-3):
        self.model = model
        self.device = device
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.loss_history = []

    def _metrics(self, logits, y):
        preds = logits.argmax(dim=-1).detach().cpu().numpy()
        y_np  = y.detach().cpu().numpy()
        acc = (preds == y_np).mean()
        f1m = f1_score(y_np, preds, average="macro", zero_division=0)
        return acc, f1m

    def train(self, data, n_epochs=300, progress_every=20):
        data = data.to(self.device)
        for ep in range(1, n_epochs+1):
            self.model.train()
            self.optimizer.zero_grad()

            edge_weight = getattr(data, "edge_weight", None)
            logits = self.model(data.x, data.edge_index, edge_weight=edge_weight)

            loss = self.criterion(logits, data.y)
            loss.backward()
            self.optimizer.step()
            self.loss_history.append(loss.item())

            if progress_every and ep % progress_every == 0:
                acc_tr, f1_tr = self._metrics(logits, data.y)
                print(f"[{ep:03d}] loss={loss.item():.4f} | acc_tr={acc_tr:.3f} f1_tr={f1_tr:.3f}")

        return self.loss_history

    @torch.no_grad()
    def evaluate(self, data):
        data = data.to(self.device)
        self.model.eval()

        edge_weight = getattr(data, "edge_weight", None)
        logits = self.model(data.x, data.edge_index, edge_weight=edge_weight)

        acc, f1m = self._metrics(logits, data.y)
        return acc, f1m, logits