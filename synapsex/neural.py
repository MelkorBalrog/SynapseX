import os
import sys
from dataclasses import replace
from typing import Dict, Tuple, List, Optional

import matplotlib

# Only fall back to a non-interactive backend when running headless on
# platforms that honour the ``DISPLAY`` variable (i.e. Unix).  Windows users
# typically have a display even when ``DISPLAY`` is unset, so we avoid forcing
# the ``Agg`` backend there.
if os.environ.get("DISPLAY", "") == "" and sys.platform != "win32":
    matplotlib.use("Agg")
matplotlib.rcParams["figure.max_open_warning"] = 0
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import hp, HyperParameters
from .models import TransformerClassifier


class PyTorchANN:
    """Wrapper around a PyTorch model with training and inference helpers.

    Parameters
    ----------
    hp_override:
        Optional ``HyperParameters`` instance.  When provided it overrides the
        global configuration and allows techniques such as genetic algorithms to
        propose new hyper-parameter sets.
    """

    def __init__(self, hp_override: Optional[HyperParameters] = None):
        self.hp = hp_override or hp
        patch_size = max(self.hp.image_size // 4, 1)
        embed_dim = patch_size * patch_size
        if embed_dim % self.hp.nhead != 0:
            for candidate in range(self.hp.nhead, 0, -1):
                if embed_dim % candidate == 0:
                    self.hp = replace(self.hp, nhead=candidate)
                    break
        self.model = TransformerClassifier(
            self.hp.image_size,
            num_classes=3,
            dropout=self.hp.dropout,
            num_layers=self.hp.num_layers,
            nhead=self.hp.nhead,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _format_input(self, X: torch.Tensor) -> torch.Tensor:
        """Reshape tensors to ``(N, 1, image_size, image_size)``.

        Training and inference code often provides flattened images of shape
        ``(N, image_size * image_size)``.  The transformer-based classifier,
        however, expects 4D image tensors.  This helper centralises the
        conversion so all call sites consistently feed the network correctly
        shaped inputs.
        """

        if X.dim() == 1:
            X = X.unsqueeze(0)
        if X.dim() == 2:
            return X.view(-1, 1, self.hp.image_size, self.hp.image_size)
        if X.dim() == 3 and X.size(1) == 1:
            return X.view(-1, 1, self.hp.image_size, self.hp.image_size)
        if X.dim() == 4:
            return X
        raise ValueError(f"Unexpected input shape {tuple(X.shape)}")

    def train(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        *,
        patience: int = 3,
        min_epochs: int = 5,
        val_split: float = 0.2,
    ) -> Tuple[Dict[str, float], List[plt.Figure]]:
        """Train the network and return evaluation metrics and figures.

        The function always generates training curves, weight heatmaps and a
        confusion matrix for consumers such as the GUI.  Callers can decide
        how to display or embed the returned figures."""

        # Split the data into a deterministic training/validation partition so
        # early stopping decisions are based on unseen samples.
        n = len(X)
        val_size = int(n * val_split)
        train_X, val_X = X[:-val_size], X[-val_size:]
        train_y, val_y = y[:-val_size], y[-val_size:]

        train_ds = TensorDataset(self._format_input(train_X), train_y)
        train_loader = DataLoader(
            train_ds, batch_size=self.hp.batch_size, shuffle=True
        )

        opt = torch.optim.Adam(self.model.parameters(), lr=self.hp.learning_rate)
        criterion = nn.CrossEntropyLoss()
        self.model.train()

        loss_hist: List[float] = []
        acc_hist: List[float] = []
        prec_hist: List[float] = []
        rec_hist: List[float] = []
        f1_hist: List[float] = []

        best_f1 = -1.0
        best_state: Optional[dict] = None
        stale_epochs = 0

        for _ in range(self.hp.epochs):
            epoch_loss = 0.0
            total = 0
            for xb, yb in train_loader:
                opt.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                opt.step()
                epoch_loss += float(loss.item()) * xb.size(0)
                total += xb.size(0)

            loss_hist.append(epoch_loss / total if total else 0.0)
            train_metrics = self.evaluate(train_X, train_y)
            acc_hist.append(train_metrics["accuracy"])
            prec_hist.append(train_metrics["precision"])
            rec_hist.append(train_metrics["recall"])
            f1_hist.append(train_metrics["f1"])

            val_metrics = self.evaluate(val_X, val_y)
            if val_metrics["f1"] > best_f1 + 1e-4:
                best_f1 = val_metrics["f1"]
                best_state = self.model.state_dict()
                stale_epochs = 0
            else:
                stale_epochs += 1
                if len(loss_hist) >= min_epochs and stale_epochs >= patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        final_metrics = self.evaluate(X, y)

        # Generate figures for consumers such as the GUI.
        figs = [
            self._plot_training(loss_hist, acc_hist, prec_hist, rec_hist, f1_hist),
            self._plot_weights(),
        ]
        preds = self.predict(X).argmax(dim=1).cpu().numpy()
        figs.append(self._plot_confusion_matrix(y.cpu().numpy(), preds))
        figs = [fig for fig in figs if fig is not None]

        return final_metrics, figs

    def predict(self, X: torch.Tensor, mc_dropout: bool = False) -> torch.Tensor:
        X = self._format_input(X)
        if mc_dropout:
            self.model.train()  # enable dropout
            preds = []
            for _ in range(self.hp.mc_dropout_passes):
                preds.append(self.model(X))
            mean = torch.stack(preds).mean(0)
            return nn.functional.softmax(mean, dim=1)
        else:
            self.model.eval()
            with torch.no_grad():
                logits = self.model(X)
            return nn.functional.softmax(logits, dim=1)

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """Return accuracy, precision, recall and F1 for the given dataset."""

        self.model.eval()
        with torch.no_grad():
            logits = self.model(self._format_input(X))
            preds = logits.argmax(dim=1)

        accuracy = float((preds == y).float().mean().item())
        num_classes = logits.shape[1]
        precision_list = []
        recall_list = []
        for cls in range(num_classes):
            tp = ((preds == cls) & (y == cls)).sum().item()
            fp = ((preds == cls) & (y != cls)).sum().item()
            fn = ((preds != cls) & (y == cls)).sum().item()
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            precision_list.append(precision)
            recall_list.append(recall)
        precision = float(sum(precision_list) / num_classes)
        recall = float(sum(recall_list) / num_classes)
        f1 = float(2 * precision * recall / (precision + recall + 1e-8))
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    def architecture(self) -> Dict[str, int]:
        """Return a dictionary describing the ANN structure."""

        return {"num_layers": self.hp.num_layers, "nhead": self.hp.nhead}

    def tune_hyperparameters_ga(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        *,
        generations: int = 5,
        population_size: int = 8,
    ) -> None:
        """Use a genetic algorithm and adopt the best performing network.

        The search returns both the strongest ``PyTorchANN`` instance and its
        hyper-parameters so that the selected model—with the lowest false
        positives and false negatives—becomes this wrapper's active network."""

        from .genetic import genetic_search

        best_hp, best_ann = genetic_search(
            X, y, generations=generations, population_size=population_size
        )
        self.hp = best_hp
        self.model = best_ann.model

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.model.state_dict(), "hp": self.hp.__dict__}, path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            hp_dict = state.get("hp")
            if hp_dict:
                self.hp = HyperParameters(**hp_dict)
                patch_size = max(self.hp.image_size // 4, 1)
                embed_dim = patch_size * patch_size
                if embed_dim % self.hp.nhead != 0:
                    for candidate in range(self.hp.nhead, 0, -1):
                        if embed_dim % candidate == 0:
                            self.hp = replace(self.hp, nhead=candidate)
                            break
            self.model = TransformerClassifier(
                self.hp.image_size,
                num_classes=3,
                dropout=self.hp.dropout,
                num_layers=self.hp.num_layers,
                nhead=self.hp.nhead,
            )
            # Allow loading models saved without positional embeddings
            self.model.load_state_dict(state["state_dict"], strict=False)
        else:
            patch_size = max(self.hp.image_size // 4, 1)
            embed_dim = patch_size * patch_size
            if embed_dim % self.hp.nhead != 0:
                for candidate in range(self.hp.nhead, 0, -1):
                    if embed_dim % candidate == 0:
                        self.hp = replace(self.hp, nhead=candidate)
                        break
            self.model = TransformerClassifier(
                self.hp.image_size,
                num_classes=3,
                dropout=self.hp.dropout,
                num_layers=self.hp.num_layers,
                nhead=self.hp.nhead,
            )
            self.model.load_state_dict(state, strict=False)


    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------
    def _plot_training(
        self,
        loss_hist: List[float],
        acc_hist: List[float],
        prec_hist: List[float],
        rec_hist: List[float],
        f1_hist: List[float],
    ):
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

    def _plot_weights(self):
        linears = [m for m in self.model.modules() if isinstance(m, nn.Linear)]
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

    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray):
        num_classes = int(max(y_true.max(), y_pred.max()) + 1)
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[int(t), int(p)] += 1
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
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        plt.tight_layout()
        return fig
