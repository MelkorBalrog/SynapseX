import os
import sys

import pytest

sys.path.append(os.getcwd())
from synapse.models.redundant_ip import RedundantNeuralIP
from synapse.hardware.cpu import CPU


def test_get_num_classes(tmp_path):
    (tmp_path / "class0").mkdir()
    (tmp_path / "class1").mkdir()
    ip = RedundantNeuralIP(train_data_dir=str(tmp_path))
    ip.run_instruction("GET_NUM_CLASSES")
    assert ip.last_result == 2


def test_cpu_halt_uses_class_names(capsys):
    ip = RedundantNeuralIP()
    ip.class_names = ["class0", "class1"]
    ip.last_result = 1
    cpu = CPU("CPU1", None, ip, None, None)
    cpu.set_reg("$t9", 1)
    cpu.step(["HALT"])
    captured = capsys.readouterr().out
    assert "Final classification: class1" in captured
