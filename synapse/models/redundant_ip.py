"""Management of multiple ANNs with majority voting."""
from collections import Counter
import numpy as np


class RedundantNeuralIP:
    def __init__(self):
        self.ann_map = {}

    def add_ann(self, ann_id: int, ann):
        self.ann_map[ann_id] = ann

    def train_ann(self, ann_id: int, X: np.ndarray, y: np.ndarray, epochs: int, lr: float, batch_size: int):
        self.ann_map[ann_id].train_model(X, y, epochs, lr, batch_size)

    def predict_majority(self, X: np.ndarray, mc_passes: int = 10):
        preds = {}
        for ann_id, ann in self.ann_map.items():
            pred = ann.predict(X)
            preds[ann_id] = pred
        # majority voting across ANNs for first sample
        votes = [preds[ann_id][0] for ann_id in self.ann_map]
        majority = Counter(votes).most_common(1)[0][0]
        return majority, preds
