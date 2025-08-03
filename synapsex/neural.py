import os
from typing import Dict, Tuple, List

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import hp
from .models import TransformerClassifier


class PyTorchANN:
    """Wrapper around a PyTorch model with training and inference helpers."""

    def __init__(self):
        self.model = TransformerClassifier(hp.image_size, num_classes=3, dropout=hp.dropout)

    def train(self, X: torch.Tensor, y: torch.Tensor) -> None:
        dataset = TensorDataset(X.unsqueeze(1), y)
        loader = DataLoader(dataset, batch_size=hp.batch_size, shuffle=True)
        opt = torch.optim.Adam(self.model.parameters(), lr=hp.learning_rate)
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for _ in range(hp.epochs):
            for xb, yb in loader:
                opt.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                opt.step()

    def predict(self, X: torch.Tensor, mc_dropout: bool = False) -> torch.Tensor:
        X = X.unsqueeze(1)
        if mc_dropout:
            self.model.train()  # enable dropout
            preds = []
            for _ in range(hp.mc_dropout_passes):
                preds.append(self.model(X))
            mean = torch.stack(preds).mean(0)
            return nn.functional.softmax(mean, dim=1)
        else:
            self.model.eval()
            with torch.no_grad():
                logits = self.model(X)
            return nn.functional.softmax(logits, dim=1)

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path, map_location="cpu"))


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
