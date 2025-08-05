import os
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
        self.model = TransformerClassifier(self.hp.image_size, num_classes=3, dropout=self.hp.dropout)

    def train(self, X: torch.Tensor, y: torch.Tensor, *, patience: int = 2) -> Dict[str, float]:
        """Train the network and return evaluation metrics.

        Implements a small form of early stopping based on the F1 score to help
        improve accuracy, precision and recall."""

        dataset = TensorDataset(X.unsqueeze(1), y)
        loader = DataLoader(dataset, batch_size=self.hp.batch_size, shuffle=True)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.hp.learning_rate)
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        best_f1 = 0.0
        stale_epochs = 0
        for _ in range(self.hp.epochs):
            for xb, yb in loader:
                opt.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                opt.step()

            metrics = self.evaluate(X, y)
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break

        return self.evaluate(X, y)

    def predict(self, X: torch.Tensor, mc_dropout: bool = False) -> torch.Tensor:
        X = X.unsqueeze(1)
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
            logits = self.model(X.unsqueeze(1))
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

    def tune_hyperparameters_ga(self, X: torch.Tensor, y: torch.Tensor, *, generations: int = 5,
                                population_size: int = 8) -> None:
        """Use a genetic algorithm to tune the underlying hyper-parameters.

        The best configuration according to F1 score is loaded into this
        instance's model."""

        from .genetic import genetic_search

        best_hp = genetic_search(X, y, generations=generations, population_size=population_size)
        self.hp = best_hp
        self.model = TransformerClassifier(self.hp.image_size, num_classes=3, dropout=self.hp.dropout)

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location="cpu")
        # Allow loading models saved without positional embeddings
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
