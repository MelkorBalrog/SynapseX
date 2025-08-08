import os
import sys

sys.path.insert(0, os.getcwd())

from synapse.models.redundant_ip import RedundantNeuralIP
from synapsex.config import hp


def test_class_names_from_training_dir(tmp_path):
    (tmp_path / "car").mkdir()
    (tmp_path / "bus").mkdir()
    hp.num_classes = 0
    ip = RedundantNeuralIP(train_data_dir=str(tmp_path))
    ip.run_instruction("CONFIG_ANN 0 FINALIZE 0.2")
    assert hp.num_classes == 2
    assert ip.class_names == sorted(["car", "bus"])
