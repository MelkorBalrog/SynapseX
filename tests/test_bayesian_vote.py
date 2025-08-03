import os
import sys
import numpy as np
import torch

sys.path.append(os.getcwd())
from synapsex.neural import RedundantNeuralIP as TorchRedundantIP
from synapse.models.redundant_ip import RedundantNeuralIP as NumpyRedundantIP


class DummyTorchANN:
    def __init__(self, probs):
        self._probs = torch.tensor([probs], dtype=torch.float32)

    def predict(self, X, mc_dropout=True):
        return self._probs


def test_bayesian_vote_torch():
    ip = TorchRedundantIP()
    ip.ann_map = {
        0: DummyTorchANN([0.8, 0.2]),
        1: DummyTorchANN([0.6, 0.4]),
        2: DummyTorchANN([0.9, 0.1]),
    }
    pred, posterior = ip.majority_vote(torch.zeros(1, 1, 8, 8))
    expected = torch.tensor([0.8 * 0.6 * 0.9, 0.2 * 0.4 * 0.1])
    expected = expected / expected.sum()
    assert pred == int(expected.argmax())
    assert torch.allclose(posterior, expected, atol=1e-6)


class DummyVirtualANN:
    def __init__(self, probs):
        self._probs = np.array([probs], dtype=float)

    def predict_proba(self, X):
        return self._probs


def test_bayesian_vote_numpy():
    ip = NumpyRedundantIP()
    ip.ann_map = {
        0: DummyVirtualANN([0.51, 0.49]),
        1: DummyVirtualANN([0.51, 0.49]),
        2: DummyVirtualANN([0.01, 0.99]),
    }
    pred, _ = ip.predict_majority(np.zeros((1, 2)))
    expected = np.array([0.51 * 0.51 * 0.01, 0.49 * 0.49 * 0.99])
    assert pred == int(expected.argmax())
