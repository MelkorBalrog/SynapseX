import os
import pathlib
import sys

sys.path.append(os.getcwd())

from synapse.soc import SoC


class DummyNeuralIP:
    """Minimal neural IP stub returning preset predictions."""

    def __init__(self):
        # predictions for three ANNs
        self.preds = {0: 0, 1: 1, 2: 1}
        self.last_result = None
        self.class_names = []

    def run_instruction(self, subcmd: str, memory=None) -> None:
        tokens = subcmd.strip().split()
        op = tokens[0].upper()
        if op == "GET_NUM_CLASSES":
            self.last_result = 3
        elif op == "GET_ARGMAX":
            ann_id = int(tokens[1])
            self.last_result = self.preds[ann_id]
        elif op == "INFER_ANN":
            self.last_result = None
        else:
            self.last_result = None


def test_classification_asm_majority(tmp_path):
    # Load assembly program
    asm_path = pathlib.Path("asm/classification.asm")
    lines = asm_path.read_text().splitlines()

    soc = SoC()
    soc.neural_ip = DummyNeuralIP()
    soc.cpu.neural_ip = soc.neural_ip
    soc.load_assembly(lines)

    soc.run(max_steps=500)

    assert soc.cpu.get_reg("$t9") == 1
