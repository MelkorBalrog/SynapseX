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
from PIL import Image

sys.path.append(os.getcwd())
from synapsex.image_processing import load_vehicle_dataset


def _prepare_dataset(tmp_path):
    (tmp_path / "car").mkdir()
    (tmp_path / "truck").mkdir()
    Image.fromarray(np.zeros((10, 10), dtype=np.uint8)).save(tmp_path / "car" / "a.png")
    Image.fromarray(np.full((10, 10), 255, dtype=np.uint8)).save(tmp_path / "truck" / "b.png")


def test_load_vehicle_dataset(tmp_path):
    _prepare_dataset(tmp_path)
    X, y, names = load_vehicle_dataset(tmp_path, target_size=8)
    assert X.shape == (144, 64)
    assert torch.bincount(y).tolist() == [72, 72]
    assert names == ["car", "truck"]


def test_load_vehicle_dataset_no_rotate(tmp_path):
    _prepare_dataset(tmp_path)
    X, y, names = load_vehicle_dataset(tmp_path, target_size=8, rotate=False)
    assert X.shape == (2, 64)
    assert y.tolist() == [0, 1]
    assert names == ["car", "truck"]
