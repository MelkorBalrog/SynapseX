"""PyTorch implementation of the virtual ANN used by SynapseX.

This version augments the basic network with utility methods to visualise
training progress and weight distributions.  The additional functionality
is used by the assembly programs shipped with the project which expect
windows to pop up showing the training curves.
"""

from typing import List
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


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
        return_figures: bool = False,
    ):
        """Train the ANN and optionally return matplotlib figures.

        Parameters
        ----------
        X, y: np.ndarray
            Training data and labels.
        epochs: int
            Number of epochs to train for.
        lr: float
            Learning rate for Adam optimiser.
        batch_size: int
            Mini-batch size used during training.
        return_figures: bool, optional
            When ``True`` the method returns a list of matplotlib ``Figure``
            objects instead of displaying them directly.
        """
        dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_hist, acc_hist = [], []
        self.train()
        for _ in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                preds = self(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
                correct += (preds.argmax(dim=1) == yb).sum().item()
                total += xb.size(0)
            loss_hist.append(epoch_loss / total)
            acc_hist.append(correct / total if total else 0)
        preds_full = self.predict(X)
        figs = []
        f = self.visualize_training(loss_hist, acc_hist)
        if f is not None:
            figs.append(f)
        f = self.visualize_weights()
        if f is not None:
            figs.append(f)
        f = self.visualize_confusion_matrix(y, preds_full, self.layer_sizes[-1])
        if f is not None:
            figs.append(f)
        if return_figures:
            return figs
        for fig in figs:
            fig.show()
        return []

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
    def visualize_training(self, loss_hist, acc_hist):
        """Plot loss and accuracy curves and return the figure."""
        if not loss_hist:
            return None
        epochs = range(1, len(loss_hist) + 1)
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss", color="tab:red")
        ax1.plot(epochs, loss_hist, color="tab:red", label="Loss")
        ax1.tick_params(axis="y", labelcolor="tab:red")
        ax2 = ax1.twinx()
        ax2.set_ylabel("Accuracy", color="tab:blue")
        ax2.plot(epochs, acc_hist, color="tab:blue", label="Accuracy")
        ax2.tick_params(axis="y", labelcolor="tab:blue")
        fig.tight_layout()
        ax1.set_title("Training Progress")
        return fig

    def visualize_weights(self):
        """Visualise the first layer weights as an image."""
        first_layer = None
        for mod in self.net:
            if isinstance(mod, nn.Linear):
                first_layer = mod
                break
        if first_layer is None:
            return None
        with torch.no_grad():
            weights = first_layer.weight.cpu().numpy()
        avg_w = weights.mean(axis=0)
        side = int(np.sqrt(avg_w.size))
        if side * side != avg_w.size:
            # pad to square for display
            pad = np.zeros(side * side)
            pad[: avg_w.size] = avg_w
            avg_w = pad
        img = avg_w.reshape(side, side)
        fig, ax = plt.subplots()
        ax.imshow(img, cmap="seismic")
        ax.set_title("First Layer Avg Weights")
        ax.axis("off")
        fig.tight_layout()
        return fig

    def visualize_confusion_matrix(self, y_true, y_pred, num_classes):
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        print("Confusion matrix:\n", cm)
        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap=plt.cm.Blues)
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_title("Confusion Matrix")
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
        fig.tight_layout()
        return fig

    # Persistence helpers -------------------------------------------------
    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))


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

    def train_model(self, X: np.ndarray, y: np.ndarray, epochs: int, lr: float, batch_size: int):
        dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                preds = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

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
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path))
