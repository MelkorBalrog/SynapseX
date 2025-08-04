import os
from typing import Dict, Tuple, List
import random

import numpy as np
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=hp.lr_decay_factor, patience=hp.lr_decay_patience
        )
        criterion = nn.CrossEntropyLoss()
        num_classes = self.model.classifier.out_features
        self.model.train()
        epoch = 0
        best_f1 = 0.0
        stagnant = 0
        while epoch < hp.max_epochs:
            epoch += 1
            correct = 0
            total = 0
            all_preds: List[int] = []
            all_true: List[int] = []
            for xb, yb in loader:
                opt.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                opt.step()
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += xb.size(0)
                all_preds.extend(preds.cpu().numpy().tolist())
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

            scheduler.step(f1_val)

            if f1_val > best_f1:
                best_f1 = f1_val
                stagnant = 0
            else:
                stagnant += 1
                if stagnant >= hp.mutate_patience:
                    self.mutate()
                    opt = torch.optim.Adam(self.model.parameters(), lr=hp.learning_rate)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        opt,
                        mode="max",
                        factor=hp.lr_decay_factor,
                        patience=hp.lr_decay_patience,
                    )
                    stagnant = 0

            if (
                epoch >= hp.epochs
                and acc >= hp.target_accuracy
                and prec >= hp.target_precision
                and rec >= hp.target_recall
                and f1_val >= hp.target_f1
            ):
                break

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
        cfg = {
            "image_size": hp.image_size,
            "num_classes": self.model.classifier.out_features,
            "dropout": self.model.transformer.layers[0].dropout.p,
        }
        torch.save({"state_dict": self.model.state_dict(), "config": cfg}, path)

    def load(self, path: str) -> None:
        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict) and "config" in data:
            cfg = data["config"]
            self.model = TransformerClassifier(cfg["image_size"], cfg.get("num_classes", 3), cfg.get("dropout", 0.2))
            self.model.load_state_dict(data["state_dict"], strict=False)
        else:
            # Allow loading models saved without positional embeddings
            self.model.load_state_dict(data, strict=False)

    def mutate(self) -> None:
        with torch.no_grad():
            for p in self.model.parameters():
                p.add_(torch.randn_like(p) * hp.mutation_std)
        for layer in self.model.transformer.layers:
            new_p = min(0.5, max(0.0, layer.dropout.p + random.uniform(-0.05, 0.05)))
            layer.dropout.p = new_p


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
        if not os.path.exists(prefix):
            return
        for fname in os.listdir(prefix):
            if fname.startswith("ann_") and fname.endswith(".pt"):
                ann_id = int(fname.split("_")[1].split(".")[0])
                ann = PyTorchANN()
                ann.load(os.path.join(prefix, fname))
                self.ann_map[ann_id] = ann

    def majority_vote(self, X: torch.Tensor) -> Tuple[int, torch.Tensor]:
        probs: List[torch.Tensor] = []
        for ann in self.ann_map.values():
            probs.append(ann.predict(X, mc_dropout=True))
        mean_prob = torch.stack(probs).mean(0)
        pred = int(mean_prob.argmax(dim=1)[0])
        return pred, mean_prob[0]
