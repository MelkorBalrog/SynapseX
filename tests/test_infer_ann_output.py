import io
from contextlib import redirect_stdout
from types import SimpleNamespace

import numpy as np
import torch

from synapse.models.redundant_ip import RedundantNeuralIP


class DummyMem:
    def read(self, addr):
        return np.frombuffer(np.float32(0).tobytes(), dtype=np.uint32)[0]


class DummyANN:
    def __init__(self):
        self.hp = SimpleNamespace(image_size=1)

    def predict(self, X, mc_dropout: bool = False):
        probs = torch.zeros((1, 2))
        probs[0, 1] = 1.0
        return probs

    def save(self, path):
        pass

    def load(self, path):
        pass


def test_infer_ann_prints_class_name():
    ip = RedundantNeuralIP()
    ip.class_names = ["a", "b"]
    ip.ann_map[0] = DummyANN()
    mem = DummyMem()
    buf = io.StringIO()
    with redirect_stdout(buf):
        ip.run_instruction("INFER_ANN 0", mem)
    assert "ANN 0 prediction: b" in buf.getvalue()
