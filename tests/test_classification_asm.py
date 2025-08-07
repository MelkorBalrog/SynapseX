import os
import pathlib
import sys
from types import SimpleNamespace

import torch

sys.path.append(os.getcwd())

from synapse.soc import SoC
from synapse.models.redundant_ip import RedundantNeuralIP
from synapsex.config import hp


class MinimalNeuralIP(RedundantNeuralIP):
    class DummyANN:
        def __init__(self, pred: int, num_classes: int):
            self.pred = pred
            self.hp = SimpleNamespace(image_size=1, num_classes=num_classes)

        def predict(self, X, mc_dropout: bool = False):
            probs = torch.zeros((1, self.hp.num_classes))
            probs[0, self.pred] = 1.0
            return probs

        def save(self, path):
            pass

        def load(self, path):
            pass

    def __init__(self, num_classes: int = 3, stub_preds=None):
        super().__init__()
        hp.num_classes = num_classes
        self.class_names = [str(i) for i in range(num_classes)]
        self.stub_preds = stub_preds or {0: 0, 1: 1, 2: 1}

    def _config_ann(self, tokens):
        if len(tokens) < 2:
            return
        ann_id = int(tokens[0])
        if tokens[1] == "FINALIZE":
            pred = self.stub_preds[ann_id]
            self.ann_map[ann_id] = self.DummyANN(pred, hp.num_classes)


def test_classification_asm_majority(tmp_path):
    # Load assembly program
    asm_path = pathlib.Path("asm/classification.asm")
    lines = asm_path.read_text().splitlines()

    soc = SoC()
    soc.neural_ip = MinimalNeuralIP()
    soc.cpu.neural_ip = soc.neural_ip
    soc.load_assembly(lines)

    soc.run(max_steps=500)

    # GET_NUM_CLASSES result stored in $s0
    assert soc.cpu.get_reg("$s0") == 3

    # INFER_ANN + GET_ARGMAX cache predictions
    assert soc.neural_ip._argmax == {0: 0, 1: 1, 2: 1}
    base = soc.data_map["ann_preds"] // 4
    preds = [soc.memory.read(base + i) for i in range(3)]
    assert preds == [0, 1, 1]

    # Final majority vote computed by neural IP
    assert soc.cpu.get_reg("$t9") == 1


def test_classification_majority_over_16(tmp_path):
    asm_path = pathlib.Path("asm/classification.asm")
    lines = asm_path.read_text().splitlines()

    stub_preds = {0: 5, 1: 17, 2: 17}
    soc = SoC()
    soc.neural_ip = MinimalNeuralIP(num_classes=20, stub_preds=stub_preds)
    soc.cpu.neural_ip = soc.neural_ip
    soc.load_assembly(lines)
    soc.run(max_steps=500)

    assert soc.cpu.get_reg("$s0") == 20
    base = soc.data_map["ann_preds"] // 4
    preds = [soc.memory.read(base + i) for i in range(3)]
    assert preds == [5, 17, 17]
    assert soc.cpu.get_reg("$t9") == 17

