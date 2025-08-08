import json
import os

from synapse.models.redundant_ip import RedundantNeuralIP
from synapsex.config import hp
from synapsex.neural import PyTorchANN


def test_save_all_writes_metadata(tmp_path):
    orig = hp.__dict__.copy()
    try:
        hp.num_classes = 5
        ip = RedundantNeuralIP()
        ip.ann_map[0] = PyTorchANN()
        prefix = tmp_path / "weights"
        ip.run_instruction(f"SAVE_ALL {prefix}")
        meta_path = tmp_path / "weights_meta.json"
        assert meta_path.exists()
        data = json.loads(meta_path.read_text())
        assert data["num_classes"] == 5
        assert data["class_names"] == []
    finally:
        for k, v in orig.items():
            setattr(hp, k, v)


def test_save_all_includes_class_names(tmp_path):
    (tmp_path / "car").mkdir()
    (tmp_path / "bus").mkdir()
    orig = hp.__dict__.copy()
    try:
        hp.num_classes = 0
        ip = RedundantNeuralIP(train_data_dir=str(tmp_path))
        ip.ann_map[0] = PyTorchANN()
        prefix = tmp_path / "weights"
        ip.run_instruction(f"SAVE_ALL {prefix}")
        meta_path = tmp_path / "weights_meta.json"
        data = json.loads(meta_path.read_text())
        assert data["class_names"] == sorted(["car", "bus"])
        assert data["num_classes"] == 2
    finally:
        for k, v in orig.items():
            setattr(hp, k, v)
