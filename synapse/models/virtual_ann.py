# Copyright (C) 2025 Miguel Marina
# Author: Miguel Marina <karel.capek.robotics@gmail.com>
# LinkedIn: https://www.linkedin.com/in/progman32/
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""PyTorch implementation of the virtual ANN used by SynapseX.

This version augments the basic network with utility methods to visualise
training progress and weight distributions.  The additional functionality
is used by the assembly programs shipped with the project which expect
windows to pop up showing the training curves.
"""

from typing import List
import logging

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


class VirtualANN(nn.Module):
    def __init__(self, layer_sizes: List[int], dropout_rate: float = 0.2):
        super().__init__()
        self.layer_sizes = layer_sizes
        layers = []
        for i in range(len(layer_sizes) - 1):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i + 1]
            layers.append(nn.Linear(in_dim, out_dim))
            if i < len(layer_sizes) - 2:
                layers.append(nn.LeakyReLU())
                layers.append(nn.Dropout(dropout_rate))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = nn.Sequential(*layers).to(self.device)

    def forward(self, x):
        return self.net(x)

    def train_model(self, X: np.ndarray, y: np.ndarray, epochs: int, lr: float, batch_size: int):
        """Train the ANN and return matplotlib figures.

        The original implementation popped up matplotlib windows via
        ``plt.show``.  For GUI integration we instead return the figure
        objects so callers can decide how to present them (e.g. embed in
        a notebook tab).  When used in non-GUI contexts the caller may
        simply call ``fig.show()`` on the returned figures.
        """
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(level=logging.INFO)
        dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=self.device.type == "cuda",
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_hist, acc_hist, prec_hist, rec_hist, f1_hist = [], [], [], [], []
        num_classes = self.layer_sizes[-1]
        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            all_preds: list[int] = []
            all_true: list[int] = []
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
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
            acc_hist.append(correct / total if total else 0)
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
            prec_hist.append(float(np.mean(precs)))
            rec_hist.append(float(np.mean(recs)))
            f1_hist.append(float(np.mean(f1s)))
            logger.info(
                "Epoch %d/%d - loss: %.4f - acc: %.4f - precision: %.4f - recall: %.4f - f1: %.4f",
                epoch + 1,
                epochs,
                loss_hist[-1],
                acc_hist[-1],
                prec_hist[-1],
                rec_hist[-1],
                f1_hist[-1],
            )
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
            logits = self(torch.from_numpy(X).float().to(self.device))
            return torch.argmax(logits, dim=1).cpu().numpy()

    def predict_with_uncertainty(self, X: np.ndarray, mc_passes: int = 10):
        self.train()  # enable dropout at inference
        outputs = []
        inp = torch.from_numpy(X).float().to(self.device)
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
        """Visualise a confusion matrix for arbitrary class counts."""
        num_classes = int(max(y_true.max(), y_pred.max()) + 1)
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[int(t), int(p)] += 1
        print("Confusion matrix:\n", cm)
        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap=plt.cm.Blues)
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_title("Confusion Matrix")
        if num_classes == 2:
            labels = np.array([["TN", "FP"], ["FN", "TP"]])
            for i in range(num_classes):
                for j in range(num_classes):
                    ax.text(
                        j,
                        i,
                        f"{labels[i, j]}: {cm[i, j]}",
                        ha="center",
                        va="center",
                        color="black",
                    )
        else:
            for i in range(num_classes):
                for j in range(num_classes):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        plt.tight_layout()
        return fig

    # Persistence helpers -------------------------------------------------
    def save(self, path: str):
        torch.save({k: v.cpu() for k, v in self.state_dict().items()}, path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location=self.device))


class TransformerClassifier(nn.Module):
    """Simple transformer-based classifier with adjustable embedding dimension."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        nhead: int = 4,
        num_layers: int = 1,
    ):
        super().__init__()
        embed_dim = ((input_dim + nhead - 1) // nhead) * nhead
        self.proj = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.unsqueeze(1)  # (batch, 1, embed_dim)
        x = self.encoder(x)
        x = x.squeeze(1)
        return self.classifier(x)


class PyTorchANN:
    """Wrapper around ``TransformerClassifier`` providing training and inference helpers."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        nhead: int = 4,
        num_layers: int = 1,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TransformerClassifier(input_dim, num_classes, dropout, nhead, num_layers).to(self.device)

    def train_model(self, X: np.ndarray, y: np.ndarray, epochs: int, lr: float, batch_size: int):
        dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=self.device.type == "cuda")
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                preds = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

    def predict(self, X: np.ndarray):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(X).float().to(self.device))
            return torch.argmax(logits, dim=1).cpu().numpy()

    def predict_with_uncertainty(self, X: np.ndarray, mc_passes: int = 10):
        self.model.train()  # enable dropout at inference
        outputs = []
        inp = torch.from_numpy(X).float().to(self.device)
        for _ in range(mc_passes):
            outputs.append(self.model(inp).detach().cpu().numpy())
        mean = np.mean(outputs, axis=0)
        var = np.var(outputs, axis=0)
        return mean, var

    def save(self, path: str):
        torch.save({k: v.cpu() for k, v in self.model.state_dict().items()}, path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
