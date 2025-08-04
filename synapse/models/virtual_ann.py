"""PyTorch implementation of the virtual ANN used by SynapseX.

This version augments the basic network with utility methods to visualise
training progress and weight distributions.  The additional functionality
is used by the assembly programs shipped with the project which expect
windows to pop up showing the training curves.
"""

from typing import List
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from config import hyperparameters as hp


class VirtualANN(nn.Module):
    def __init__(self, layer_sizes: List[int], dropout_rate: float = 0.2):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.dropout_rate = dropout_rate
        self._build_network()

    def _build_network(self) -> None:
        layers: List[nn.Module] = []
        for i in range(len(self.layer_sizes) - 1):
            in_dim = self.layer_sizes[i]
            out_dim = self.layer_sizes[i + 1]
            layers.append(nn.Linear(in_dim, out_dim))
            if i < len(self.layer_sizes) - 2:
                layers.append(nn.LeakyReLU())
                layers.append(nn.Dropout(self.dropout_rate))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def train_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        lr: float,
        batch_size: int,
        min_accuracy: float = hp.TARGET_ACCURACY,
        min_precision: float = hp.TARGET_PRECISION,
        min_recall: float = hp.TARGET_RECALL,
        min_f1: float = hp.TARGET_F1,
    ):
        """Train the ANN and return matplotlib figures.

        The original implementation popped up matplotlib windows via
        ``plt.show``.  For GUI integration we instead return the figure
        objects so callers can decide how to present them (e.g. embed in
        a notebook tab).  When used in non-GUI contexts the caller may
        simply call ``fig.show()`` on the returned figures.
        """
        dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_hist, acc_hist, prec_hist, rec_hist, f1_hist = [], [], [], [], []
        num_classes = self.layer_sizes[-1]
        self.train()
        epoch = 0
        best_f1 = 0.0
        stagnant = 0
        while True:
            epoch += 1
            epoch_loss = 0.0
            correct = 0
            total = 0
            all_preds: list[int] = []
            all_true: list[int] = []
            for xb, yb in loader:
                optimizer.zero_grad()
                preds = self(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
                pred_labels = preds.argmax(dim=1)
                correct += (pred_labels == yb).sum().item()
                total += xb.size(0)
                all_preds.extend(pred_labels.cpu().numpy().tolist())
                all_true.extend(yb.cpu().numpy().tolist())
            loss_hist.append(epoch_loss / total)
            acc = correct / total if total else 0.0
            acc_hist.append(acc)
            cm = np.zeros((num_classes, num_classes), dtype=int)
            for t, p in zip(all_true, all_preds):
                cm[t, p] += 1
            precs, recs, f1s = [], [], []
            for i in range(num_classes):
                tp = cm[i, i]
                fp = cm[:, i].sum() - tp
                fn = cm[i, :].sum() - tp
                prec = tp / (tp + fp) if (tp + fp) else 0.0
                rec = tp / (tp + fn) if (tp + fn) else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
                precs.append(prec)
                recs.append(rec)
                f1s.append(f1)
            prec = float(np.mean(precs))
            rec = float(np.mean(recs))
            f1_val = float(np.mean(f1s))
            prec_hist.append(prec)
            rec_hist.append(rec)
            f1_hist.append(f1_val)

            if f1_val > best_f1:
                best_f1 = f1_val
                stagnant = 0
            else:
                stagnant += 1
                if stagnant >= hp.MUTATE_PATIENCE:
                    self.mutate()
                    optimizer = torch.optim.Adam(self.parameters(), lr=lr)
                    stagnant = 0

            if (
                epoch >= epochs
                and acc >= min_accuracy
                and prec >= min_precision
                and rec >= min_recall
                and f1_val >= min_f1
            ):
                break
        preds_full = self.predict(X)

        figs = []
        fig = self.visualize_training(
            loss_hist,
            acc_hist,
            prec_hist,
            rec_hist,
            f1_hist,
        )
        if fig is not None:
            figs.append(fig)
        fig = self.visualize_weights()
        if fig is not None:
            figs.append(fig)
        fig = self.visualize_confusion_matrix(y, preds_full)
        if fig is not None:
            figs.append(fig)

        return figs

    def predict(self, X: np.ndarray):
        self.eval()
        with torch.no_grad():
            logits = self(torch.from_numpy(X).float())
            return torch.argmax(logits, dim=1).cpu().numpy()

    def predict_with_uncertainty(self, X: np.ndarray, mc_passes: int = 10):
        self.train()  # enable dropout at inference
        outputs = []
        inp = torch.from_numpy(X).float()
        for _ in range(mc_passes):
            outputs.append(self(inp).detach().cpu().numpy())
        mean = np.mean(outputs, axis=0)
        var = np.var(outputs, axis=0)
        return mean, var

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------
    def visualize_training(self, loss_hist, acc_hist, prec_hist, rec_hist, f1_hist):
        """Plot training loss and evaluation metrics in separate subplots."""
        if not loss_hist:
            return None
        epochs = range(1, len(loss_hist) + 1)
        fig, axes = plt.subplots(5, 1, figsize=(8, 12), sharex=True)

        axes[0].plot(epochs, loss_hist, color="tab:red")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Progress")

        metrics = [
            (acc_hist, "Accuracy", "tab:blue"),
            (prec_hist, "Precision", "tab:orange"),
            (rec_hist, "Recall", "tab:green"),
            (f1_hist, "F1 Score", "tab:purple"),
        ]
        for ax, (hist, label, color) in zip(axes[1:], metrics):
            ax.plot(epochs, hist, color=color)
            ax.set_ylabel(label)

        axes[-1].set_xlabel("Epoch")
        fig.tight_layout()
        return fig

    def visualize_weights(self):
        """Visualise all linear layer weights."""
        linears = [mod for mod in self.net if isinstance(mod, nn.Linear)]
        if not linears:
            return None

        cols = len(linears)
        fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
        if cols == 1:
            axes = [axes]

        for idx, (layer, ax) in enumerate(zip(linears, axes)):
            with torch.no_grad():
                weights = layer.weight.cpu().numpy()
            ax.imshow(weights, cmap="seismic")
            ax.set_title(f"Layer {idx + 1} Weights")
            ax.set_xticks([])
            ax.set_yticks([])

        fig.tight_layout()
        return fig

    def visualize_confusion_matrix(self, y_true, y_pred):
        """Visualise binary confusion matrix labelled with TN/FP/FN/TP."""
        cm = np.zeros((2, 2), dtype=int)
        for t, p in zip(y_true, y_pred):
            if t == 1 and p == 1:
                cm[1, 1] += 1  # TP
            elif t == 1 and p == 0:
                cm[1, 0] += 1  # FN
            elif t == 0 and p == 1:
                cm[0, 1] += 1  # FP
            else:
                cm[0, 0] += 1  # TN
        print("Confusion matrix:\n", cm)
        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap=plt.cm.Blues)
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Negative", "Positive"])
        ax.set_yticklabels(["Negative", "Positive"])
        ax.set_title("Confusion Matrix")
        labels = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{labels[i][j]}: {cm[i, j]}", ha="center", va="center", color="black")
        plt.tight_layout()
        return fig

    # Persistence helpers -------------------------------------------------
    def save(self, path: str):
        torch.save(
            {
                "state_dict": self.state_dict(),
                "layer_sizes": self.layer_sizes,
                "dropout": self.dropout_rate,
            },
            path,
        )

    def load(self, path: str):
        data = torch.load(path)
        if isinstance(data, dict):
            self.layer_sizes = data.get("layer_sizes", self.layer_sizes)
            self.dropout_rate = data.get("dropout", self.dropout_rate)
            self._build_network()
            self.load_state_dict(data["state_dict"], strict=False)
        else:
            self.load_state_dict(data)

    def mutate(self) -> None:
        """Randomly alter network structure and weights."""
        if len(self.layer_sizes) > 2 and random.random() < 0.5:
            idx = random.randint(1, len(self.layer_sizes) - 2)
            change = max(1, int(self.layer_sizes[idx] * 0.1))
            self.layer_sizes[idx] = max(1, self.layer_sizes[idx] + random.choice([-change, change]))
        else:
            new_units = max(1, int(self.layer_sizes[-2] * 0.5))
            self.layer_sizes.insert(-1, new_units)
        self._build_network()
        with torch.no_grad():
            for p in self.parameters():
                p.add_(torch.randn_like(p) * hp.MUTATION_STD)


class TransformerClassifier(nn.Module):
    """Simple transformer-based classifier with adjustable embedding dimension."""

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.1, nhead: int = 4):
        super().__init__()
        embed_dim = ((input_dim + nhead - 1) // nhead) * nhead
        self.proj = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.unsqueeze(1)  # (batch, 1, embed_dim)
        x = self.encoder(x)
        x = x.squeeze(1)
        return self.classifier(x)


class PyTorchANN:
    """Wrapper around ``TransformerClassifier`` providing training and inference helpers."""

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.1, nhead: int = 4):
        self.model = TransformerClassifier(input_dim, num_classes, dropout, nhead)

    def train_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        lr: float,
        batch_size: int,
        min_accuracy: float = hp.TARGET_ACCURACY,
        min_precision: float = hp.TARGET_PRECISION,
        min_recall: float = hp.TARGET_RECALL,
        min_f1: float = hp.TARGET_F1,
    ):
        dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        num_classes = self.model.classifier.out_features
        self.model.train()
        epoch = 0
        best_f1 = 0.0
        stagnant = 0
        while True:
            epoch += 1
            epoch_loss = 0.0
            correct = 0
            total = 0
            all_preds: list[int] = []
            all_true: list[int] = []
            for xb, yb in loader:
                optimizer.zero_grad()
                preds = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
                pred_labels = preds.argmax(dim=1)
                correct += (pred_labels == yb).sum().item()
                total += xb.size(0)
                all_preds.extend(pred_labels.cpu().numpy().tolist())
                all_true.extend(yb.cpu().numpy().tolist())
            acc = correct / total if total else 0.0
            cm = np.zeros((num_classes, num_classes), dtype=int)
            for t, p in zip(all_true, all_preds):
                cm[t, p] += 1
            precs, recs, f1s = [], [], []
            for i in range(num_classes):
                tp = cm[i, i]
                fp = cm[:, i].sum() - tp
                fn = cm[i, :].sum() - tp
                prec = tp / (tp + fp) if (tp + fp) else 0.0
                rec = tp / (tp + fn) if (tp + fn) else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
                precs.append(prec)
                recs.append(rec)
                f1s.append(f1)
            prec = float(np.mean(precs))
            rec = float(np.mean(recs))
            f1_val = float(np.mean(f1s))

            if f1_val > best_f1:
                best_f1 = f1_val
                stagnant = 0
            else:
                stagnant += 1
                if stagnant >= hp.MUTATE_PATIENCE:
                    self.mutate()
                    optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
                    stagnant = 0

            if (
                epoch >= epochs
                and acc >= min_accuracy
                and prec >= min_precision
                and rec >= min_recall
                and f1_val >= min_f1
            ):
                break

    def predict(self, X: np.ndarray):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(X).float())
            return torch.argmax(logits, dim=1).cpu().numpy()

    def predict_with_uncertainty(self, X: np.ndarray, mc_passes: int = 10):
        self.model.train()  # enable dropout at inference
        outputs = []
        inp = torch.from_numpy(X).float()
        for _ in range(mc_passes):
            outputs.append(self.model(inp).detach().cpu().numpy())
        mean = np.mean(outputs, axis=0)
        var = np.var(outputs, axis=0)
        return mean, var

    def save(self, path: str):
        cfg = {
            "input_dim": self.model.proj.in_features,
            "num_classes": self.model.classifier.out_features,
            "dropout": self.model.encoder.layers[0].dropout.p,
            "nhead": self.model.encoder.layers[0].self_attn.num_heads,
        }
        torch.save({"state_dict": self.model.state_dict(), "config": cfg}, path)

    def load(self, path: str):
        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict) and "config" in data:
            cfg = data["config"]
            self.model = TransformerClassifier(
                cfg["input_dim"], cfg["num_classes"], cfg.get("dropout", 0.1), cfg.get("nhead", 4)
            )
            self.model.load_state_dict(data["state_dict"], strict=False)
        else:
            self.model.load_state_dict(data)

    def mutate(self) -> None:
        """Perturb weights and dropout to explore new structures."""
        with torch.no_grad():
            for p in self.model.parameters():
                p.add_(torch.randn_like(p) * hp.MUTATION_STD)
        # adjust dropout of encoder layers
        for layer in self.model.encoder.layers:
            new_p = min(0.5, max(0.0, layer.dropout.p + random.uniform(-0.05, 0.05)))
            layer.dropout.p = new_p
