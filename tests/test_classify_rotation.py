# Copyright (C) 2025 Miguel Marina
# Author: Miguel Marina <karel.capek.robotics@gmail.com>
# LinkedIn: https://www.linkedin.com/in/progman32/
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import io
import json
import os
import pathlib
import sys
from types import SimpleNamespace
from contextlib import redirect_stdout

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.getcwd())

from SynapseX import classify_with_assembly  # noqa: E402
from synapse.soc import SoC  # noqa: E402
from synapse.models.redundant_ip import RedundantNeuralIP  # noqa: E402
from synapsex.config import hp  # noqa: E402


class SeqNeuralIP(RedundantNeuralIP):
    class DummyANN:
        def __init__(self, pred: int, num_classes: int = 2):
            self.pred = pred
            self.hp = SimpleNamespace(
                image_size=1, image_channels=2, num_classes=num_classes
            )

        def predict(self, _X, mc_dropout: bool = False):
            probs = torch.zeros((1, self.hp.num_classes))
            probs[0, self.pred] = 1.0
            return probs

        def save(self, _path):
            pass

        def load(self, _path):
            pass

    def __init__(self) -> None:
        super().__init__()
        hp.num_classes = 2
        self.class_names = ["a", "b"]
        # Predictions for three successive runs (3 ANNs each)
        self.run_preds = [[0, 0, 1], [1, 1, 0], [1, 1, 1]]
        self.run_idx = -1

    def _config_ann(self, tokens):
        if len(tokens) < 2:
            return
        ann_id = int(tokens[0])
        if tokens[1] == "FINALIZE":
            if ann_id == 0:
                self.run_idx += 1
            pred = self.run_preds[self.run_idx][ann_id]
            self.ann_map[ann_id] = self.DummyANN(pred, hp.num_classes)


def test_classify_with_rotation(tmp_path):
    img = Image.fromarray(np.zeros((10, 10), dtype=np.uint8))
    img_path = tmp_path / "img.png"
    img.save(img_path)

    meta = tmp_path / "trained_weights_meta.json"
    meta.write_text(json.dumps({"num_classes": 2, "class_names": ["a", "b"]}))
    for i in range(3):
        (tmp_path / f"trained_weights_{i}.pt").write_text("dummy")

    asm_src = pathlib.Path(__file__).resolve().parents[1] / "asm" / "classification.asm"
    asm_dir = tmp_path / "asm"
    asm_dir.mkdir()
    (asm_dir / "classification.asm").write_text(asm_src.read_text())

    soc = SoC()
    soc.neural_ip = SeqNeuralIP()
    soc.cpu.neural_ip = soc.neural_ip

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            idx, label = classify_with_assembly(str(img_path), angles=[0, 90, 180], soc=soc)
    finally:
        os.chdir(cwd)

    assert idx == 1
    assert label == "b"
    # Ensure assembly was executed three times (nine ANN predictions)
    assert soc.neural_ip.run_idx == 2

