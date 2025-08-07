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

import os
import sys

import numpy as np
import torch

sys.path.append(os.getcwd())
from synapse.models.redundant_ip import RedundantNeuralIP


class DummyANN:
    def __init__(self, cls: int):
        self.cls = cls

    def predict(self, X, mc_dropout: bool = False):
        probs = torch.zeros((1, 3))
        probs[0, self.cls] = 1.0
        return probs


def test_predict_majority_picks_mode_class():
    ip = RedundantNeuralIP()
    ip.ann_map = {0: DummyANN(0), 1: DummyANN(1), 2: DummyANN(1)}
    X = np.zeros((1, 1), dtype=np.float32)
    majority, preds = ip.predict_majority(X)
    assert majority == 1
    assert preds[0][0] == 0
    assert preds[1][0] == 1
