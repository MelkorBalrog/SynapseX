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
from synapsex.image_processing import load_process_shape_image, load_process_vehicle_image

def test_load_process_shape_image_angle_control(tmp_path):
    img = Image.fromarray(np.zeros((10, 10), dtype=np.uint8))
    path = tmp_path / "test.png"
    img.save(path)
    multi = load_process_shape_image(str(path), angles=[0, 90])
    single = load_process_shape_image(str(path), angles=[0])
    assert len(multi) == 2
    assert len(single) == 1


def test_load_process_vehicle_image_shape(tmp_path):
    img = Image.fromarray(np.zeros((20, 20, 3), dtype=np.uint8))
    path = tmp_path / "vehicle.png"
    img.save(path)
    tensor = load_process_vehicle_image(str(path), target_size=32, augment=False)
    assert tensor.shape == (3, 32, 32)
    assert tensor.dtype == torch.float32
    assert float(tensor.max()) <= 3 and float(tensor.min()) >= -3
