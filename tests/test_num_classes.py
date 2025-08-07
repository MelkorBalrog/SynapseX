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

"""Regression tests for dynamic class counts."""

import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.append(os.getcwd())
from synapse.models.redundant_ip import RedundantNeuralIP
from synapsex.config import hp


def _prepare_dataset(tmp_path):
    (tmp_path / "class0").mkdir()
    (tmp_path / "class1").mkdir()
    Image.fromarray(np.zeros((10, 10), dtype=np.uint8)).save(tmp_path / "class0" / "a.png")
    Image.fromarray(np.full((10, 10), 255, dtype=np.uint8)).save(tmp_path / "class1" / "b.png")


def test_num_classes_updates_and_predictions_in_range(tmp_path):
    orig = hp.__dict__.copy()
    try:
        hp.image_size = 8
        _prepare_dataset(tmp_path)
        ip = RedundantNeuralIP(str(tmp_path))
        ip.run_instruction("CONFIG_ANN 0 FINALIZE")
        ip.run_instruction("TRAIN_ANN 0 1")
        assert hp.num_classes == 2
        X, _, _ = ip._load_dataset()
        probs = ip.ann_map[0].predict(X)
        assert probs.shape[1] == 2
        assert int(probs.argmax(dim=1).max().item()) <= 1
    finally:
        for k, v in orig.items():
            setattr(hp, k, v)
