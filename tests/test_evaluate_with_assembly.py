import json
import os
import pathlib
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from synapse.soc import SoC
from synapse.models.redundant_ip import RedundantNeuralIP
from synapsex.config import hp
from SynapseX import evaluate_with_assembly


class MinimalNeuralIP(RedundantNeuralIP):
    class DummyANN:
        def __init__(self, pred, num_classes):
            self.pred = pred
            self.hp = SimpleNamespace(image_size=hp.image_size, num_classes=num_classes)

        def predict(self, X, mc_dropout: bool = False):
            probs = torch.zeros((1, self.hp.num_classes))
            probs[0, self.pred] = 1.0
            return probs

        def predict_class(self, X, mc_dropout: bool = False):
            return torch.tensor([self.pred])

        def save(self, path):
            pass

        def load(self, path):
            pass

    def __init__(self):
        super().__init__()
        hp.num_classes = 0
        self.class_names = ["a", "b"]
        self.stub_preds = {0: 0, 1: 0, 2: 1}

    def _config_ann(self, tokens):
        if len(tokens) < 2:
            return
        ann_id = int(tokens[0])
        if tokens[1] == "FINALIZE":
            meta_prefix = tokens[3] if len(tokens) >= 4 else ""
            if meta_prefix:
                meta_path = pathlib.Path(f"{meta_prefix}_meta.json")
                if meta_path.exists():
                    data = json.loads(meta_path.read_text())
                    hp.num_classes = data.get("num_classes", hp.num_classes)
            pred = self.stub_preds[ann_id]
            self.ann_map[ann_id] = self.DummyANN(pred, hp.num_classes)


def create_image(path, color):
    Image.new("L", (10, 10), color=color).save(path)


def test_evaluate_with_assembly(tmp_path):
    # create dataset
    data_root = tmp_path / "data"
    data_root.mkdir()
    for cls, color in [("a", 0), ("b", 255)]:
        d = data_root / cls
        d.mkdir()
        create_image(d / "img.png", color)

    # dummy weights and metadata
    meta = tmp_path / "trained_weights_meta.json"
    meta.write_text(json.dumps({"num_classes": 2, "class_names": ["a", "b"]}))
    for i in range(3):
        (tmp_path / f"trained_weights_{i}.pt").write_text("dummy")

    asm_src = pathlib.Path(__file__).resolve().parents[1] / "asm" / "classification.asm"
    asm_dir = tmp_path / "asm"
    asm_dir.mkdir()
    (asm_dir / "classification.asm").write_text(asm_src.read_text())

    soc = SoC()
    soc.neural_ip = MinimalNeuralIP()
    soc.cpu.neural_ip = soc.neural_ip

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        metrics, cm = evaluate_with_assembly(str(data_root), rotate=False, soc=soc)
    finally:
        os.chdir(cwd)

    assert cm.shape == (2, 2)
    assert metrics["accuracy"] == pytest.approx(0.5)
