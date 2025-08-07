import os
from dataclasses import replace
from typing import Dict, Tuple, List, Optional

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
    ) -> Dict[str, float]:
        """Train the network and return evaluation metrics.

        Uses a small validation split for early stopping based on the F1 score
        and restores the best observed model weights.  ``min_epochs`` guarantees
        that a few epochs are always run so the network can start learning."""

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

        best_f1 = -1.0
        best_state: Optional[dict] = None
        stale_epochs = 0

        for epoch in range(self.hp.epochs):
            for xb, yb in train_loader:
                opt.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                opt.step()

            metrics = self.evaluate(val_X, val_y)
            if metrics["f1"] > best_f1 + 1e-4:
                best_f1 = metrics["f1"]
                best_state = self.model.state_dict()
                stale_epochs = 0
            else:
                stale_epochs += 1
                if epoch + 1 >= min_epochs and stale_epochs >= patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self.evaluate(X, y)

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


class RedundantNeuralIP:
    """Manages multiple ANN instances and majority voting."""

    def __init__(self):
        self.ann_map: Dict[int, PyTorchANN] = {}

    def create_ann(self, ann_id: int) -> None:
        self.ann_map[ann_id] = PyTorchANN()

    def train_ann(self, ann_id: int, X: torch.Tensor, y: torch.Tensor) -> None:
        self.ann_map[ann_id].train(X, y)

    def predict_ann(self, ann_id: int, X: torch.Tensor, mc_dropout: bool = True) -> torch.Tensor:
        return self.ann_map[ann_id].predict(X, mc_dropout=mc_dropout)

    def save_all(self, prefix: str) -> None:
        os.makedirs(prefix, exist_ok=True)
        for ann_id, ann in self.ann_map.items():
            ann.save(os.path.join(prefix, f"ann_{ann_id}.pt"))

    def load_all(self, prefix: str) -> None:
        for ann_id, ann in self.ann_map.items():
            path = os.path.join(prefix, f"ann_{ann_id}.pt")
            if os.path.exists(path):
                ann.load(path)

    def majority_vote(self, X: torch.Tensor) -> Tuple[int, torch.Tensor]:
        probs: List[torch.Tensor] = []
        for ann in self.ann_map.values():
            probs.append(ann.predict(X, mc_dropout=True))
        mean_prob = torch.stack(probs).mean(0)
        pred = int(mean_prob.argmax(dim=1)[0])
        return pred, mean_prob[0]
