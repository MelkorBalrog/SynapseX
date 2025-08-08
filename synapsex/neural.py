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

from dataclasses import replace
from typing import Dict, Tuple, List, Optional

import logging
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["figure.max_open_warning"] = 0
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .config import hp, HyperParameters
from .models import TransformerClassifier
from .image_processing import load_annotated_dataset


logger = logging.getLogger(__name__)


class PyTorchANN:
    """Wrapper around a PyTorch model with training and inference helpers.

    Parameters
    ----------
    hp_override:
        Optional ``HyperParameters`` instance.  When provided it overrides the
        global configuration and allows techniques such as genetic algorithms to
        propose new hyper-parameter sets.
    device:
        Explicit device on which the model should run.  Defaults to ``cuda`` when
        available.
    """

    def __init__(
        self,
        hp_override: Optional[HyperParameters] = None,
        *,
        device: torch.device | str | None = None,
    ):
        self.hp = hp_override or hp
        patch_size = max(self.hp.image_size // 4, 1)
        embed_dim = patch_size * patch_size
        if embed_dim % self.hp.nhead != 0:
            for candidate in range(self.hp.nhead, 0, -1):
                if embed_dim % candidate == 0:
                    self.hp = replace(self.hp, nhead=candidate)
                    break
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = TransformerClassifier(
            self.hp.image_size,
            num_classes=self.hp.num_classes,
            dropout=self.hp.dropout,
            num_layers=self.hp.num_layers,
            nhead=self.hp.nhead,
            in_channels=self.hp.image_channels,
        ).to(self.device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _format_input(self, X: torch.Tensor) -> torch.Tensor:
        """Reshape tensors to ``(N, image_channels, image_size, image_size)``.

        Training and inference code often provides flattened images of shape
        ``(N, image_channels * image_size * image_size)``.  The transformer-based classifier,
        however, expects 4D image tensors.  This helper centralises the
        conversion so all call sites consistently feed the network correctly
        shaped inputs.
        """

        if X.dim() == 1:
            X = X.unsqueeze(0)
        if X.dim() == 2:
            return X.view(-1, self.hp.image_channels, self.hp.image_size, self.hp.image_size)
        if X.dim() == 3 and X.size(1) == self.hp.image_channels:
            return X.view(-1, self.hp.image_channels, self.hp.image_size, self.hp.image_size)
        if X.dim() == 4:
            return X
        raise ValueError(f"Unexpected input shape {tuple(X.shape)}")

    def _generalized_box_iou(
        self, boxes1: torch.Tensor, boxes2: torch.Tensor
    ) -> torch.Tensor:
        """Return the pairwise generalized IoU for corresponding boxes.

        Both inputs must be ``(N, 4)`` tensors in ``(x1, y1, x2, y2)`` format.
        ``torchvision`` is intentionally avoided so this helper works in
        environments where the dependency is not installed.
        """

        if boxes1.numel() == 0 or boxes2.numel() == 0:
            return torch.zeros((boxes1.size(0),), device=boxes1.device)

        x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
        y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
        x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
        y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

        area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (
            boxes1[:, 3] - boxes1[:, 1]
        ).clamp(min=0)
        area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (
            boxes2[:, 3] - boxes2[:, 1]
        ).clamp(min=0)
        union = area1 + area2 - inter + 1e-7
        iou = inter / union

        cx1 = torch.min(boxes1[:, 0], boxes2[:, 0])
        cy1 = torch.min(boxes1[:, 1], boxes2[:, 1])
        cx2 = torch.max(boxes1[:, 2], boxes2[:, 2])
        cy2 = torch.max(boxes1[:, 3], boxes2[:, 3])
        c_area = (cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0) + 1e-7
        return iou - (c_area - union) / c_area

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

        if not logging.getLogger().hasHandlers():
            logging.basicConfig(level=logging.INFO)
        # Split the data into a deterministic training/validation partition so
        # early stopping decisions are based on unseen samples.  Datasets used
        # in the GUI can be tiny, so guard against edge cases where the split
        # would yield empty training or validation sets.
        n = len(X)
        if n == 0:
            raise ValueError("Empty training set")
        if n < 2:
            train_X = val_X = X
            train_y = val_y = y
        else:
            val_size = int(n * val_split)
            if val_size <= 0 or val_size >= n:
                val_size = max(1, n - 1)
            train_X, val_X = X[:-val_size], X[-val_size:]
            train_y, val_y = y[:-val_size], y[-val_size:]

        train_ds = TensorDataset(self._format_input(train_X), train_y)
        train_loader = DataLoader(
            train_ds,
            batch_size=self.hp.batch_size,
            shuffle=True,
            pin_memory=self.device.type == "cuda",
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

        for epoch in range(self.hp.epochs):
            epoch_loss = 0.0
            total = 0
            for xb, yb in train_loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
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
            logger.info(
                "Epoch %d/%d - loss: %.4f - train_f1: %.4f - val_f1: %.4f",
                epoch + 1,
                self.hp.epochs,
                loss_hist[-1],
                f1_hist[-1],
                val_metrics["f1"],
            )
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

    def train_detector(
        self,
        images: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        *,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, float]:
        """Train an object detector for one step and return loss values.

        Parameters
        ----------
        images:
            Input batch of images.
        targets:
            Dictionary with ``boxes`` (``N x 4``) and ``labels`` (``N``)
            tensors containing the ground truth annotations.
        optimizer:
            Optional optimizer.  A default Adam instance is created when
            ``None``.
        """

        self.model.train()
        images = self._format_input(images).to(self.device)
        gt_boxes = targets["boxes"].to(self.device)
        gt_labels = targets["labels"].to(self.device)
        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hp.learning_rate)

        optimizer.zero_grad()
        outputs = self.model(images)
        if not isinstance(outputs, dict) or "logits" not in outputs or "boxes" not in outputs:
            raise ValueError("Model must return a dict with 'logits' and 'boxes' keys")

        cls_loss = F.cross_entropy(outputs["logits"], gt_labels)
        giou = self._generalized_box_iou(outputs["boxes"], gt_boxes)
        reg_loss = 1 - giou.mean()
        loss = cls_loss + reg_loss
        loss.backward()
        optimizer.step()

        return {
            "classification_loss": float(cls_loss.item()),
            "regression_loss": float(reg_loss.item()),
        }

    def predict(self, X: torch.Tensor, mc_dropout: bool = False) -> torch.Tensor:
        X = self._format_input(X).to(self.device)
        if mc_dropout:
            self.model.train()  # enable dropout
            preds = []
            with torch.no_grad():
                for _ in range(self.hp.mc_dropout_passes):
                    preds.append(self.model(X))
            self.model.eval()  # restore evaluation mode
            mean = torch.stack(preds).mean(0)
            return nn.functional.softmax(mean, dim=1).cpu()
        else:
            self.model.eval()
            with torch.no_grad():
                logits = self.model(X)
            return nn.functional.softmax(logits, dim=1).cpu()

    def detect_objects(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run the underlying model and return raw detection outputs.

        The default implementation simply raises ``NotImplementedError`` as
        ``PyTorchANN`` ships with a classifier.  Subclasses or external code can
        override this method with an object detection model returning a
        dictionary containing ``boxes`` (``N x 4``), ``scores`` and optional
        ``labels`` tensors.
        """

        raise NotImplementedError("Object detection model not implemented")

    def infer_objects(
        self, X: torch.Tensor, *, conf_thresh: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """Detect objects and filter out low confidence predictions."""

        thresh = self.hp.conf_thresh if conf_thresh is None else conf_thresh
        outputs = self.detect_objects(X)
        scores = outputs.get("scores")
        if scores is None:
            raise ValueError("Detection output must contain 'scores'")
        keep = scores >= thresh
        filtered = {
            k: v[keep] if isinstance(v, torch.Tensor) and v.shape[0] == scores.shape[0] else v
            for k, v in outputs.items()
        }
        return filtered

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """Return accuracy, precision, recall and F1 for the given dataset."""
        if len(X) == 0:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

        self.model.eval()
        with torch.no_grad():
            logits = self.model(self._format_input(X).to(self.device))
            preds = logits.argmax(dim=1).cpu()
        logits = logits.cpu()

        y = y.cpu()
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
            X,
            y,
            generations=generations,
            population_size=population_size,
            device=self.device,
        )
        self.hp = best_hp
        self.model = best_ann.model

    def save(self, path: str) -> None:
        state = {k: v.cpu() for k, v in self.model.state_dict().items()}
        torch.save({"state_dict": state, "hp": self.hp.__dict__}, path)

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
                num_classes=self.hp.num_classes,
                dropout=self.hp.dropout,
                num_layers=self.hp.num_layers,
                nhead=self.hp.nhead,
                in_channels=self.hp.image_channels,
            ).to(self.device)
            # Allow loading models saved without positional embeddings
            result = self.model.load_state_dict(state["state_dict"], strict=False)
            if result.missing_keys or result.unexpected_keys:
                logging.warning(
                    "load_state_dict missing keys: %s, unexpected keys: %s",
                    result.missing_keys,
                    result.unexpected_keys,
                )
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
                num_classes=self.hp.num_classes,
                dropout=self.hp.dropout,
                num_layers=self.hp.num_layers,
                nhead=self.hp.nhead,
                in_channels=self.hp.image_channels,
            ).to(self.device)
            result = self.model.load_state_dict(state, strict=False)
            if result.missing_keys or result.unexpected_keys:
                logging.warning(
                    "load_state_dict missing keys: %s, unexpected keys: %s",
                    result.missing_keys,
                    result.unexpected_keys,
                )


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
        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap=plt.cm.Blues)
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_title("Confusion Matrix")
        labels = np.array([["TN", "FP"], ["FN", "TP"]])
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{labels[i, j]}: {cm[i, j]}",
                    ha="center",
                    va="center",
                    color="black",
                )
        plt.tight_layout()
        return fig


def train_object_detector(
    root_dir: str, num_classes: int, epochs: int = 10
) -> "torch.nn.Module":
    """Train a simple object detector on an annotated dataset.

    The function loads a dataset via :func:`load_annotated_dataset` and trains a
    ``FasterRCNN`` model from ``torchvision``.  Images are expected to be stored
    under ``root_dir`` following either COCO or YOLO conventions.
    """

    import torchvision

    samples = load_annotated_dataset(root_dir)

    class DetectionDataset(torch.utils.data.Dataset):
        def __init__(self, samples):
            self.samples = samples

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx):
            img, boxes, labels = self.samples[idx]
            target = {"boxes": boxes, "labels": labels}
            return img, target

    dataset = DetectionDataset(samples)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x))
    )

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None, num_classes=num_classes
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optim = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

    model.train()
    for _ in range(epochs):
        for images, targets in loader:
            imgs = [img.to(device) for img in images]
            targs = [{k: v.to(device) for k, v in t.items()} for t in targets]
            losses = model(imgs, targs)
            loss = sum(losses.values())
            optim.zero_grad()
            loss.backward()
            optim.step()

    return model
