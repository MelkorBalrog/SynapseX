import os
import sys

import numpy as np
from PIL import Image

sys.path.append(os.getcwd())
from synapsex.image_processing import load_process_shape_image

def test_load_process_shape_image_angle_control(tmp_path):
    img = Image.fromarray(np.zeros((10, 10), dtype=np.uint8))
    path = tmp_path / "test.png"
    img.save(path)
    multi = load_process_shape_image(str(path), save=False, angles=[0, 90])
    single = load_process_shape_image(str(path), save=False, angles=[0])
    assert len(multi) == 2
    assert len(single) == 1
