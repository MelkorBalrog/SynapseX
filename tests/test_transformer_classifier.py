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
import subprocess
import sys
import shutil

import pytest
import torch

sys.path.append(os.getcwd())
from synapsex.models import TransformerClassifier


@pytest.mark.skipif(shutil.which("iverilog") is None, reason="iverilog not installed")
def test_transformer_classifier_hw_match():
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog not installed")
    model = TransformerClassifier(image_size=8, num_classes=3, dropout=0.0)
    for p in model.parameters():
        torch.nn.init.constant_(p, 0.0)
    x = torch.zeros(1, 1, 8, 8)
    with torch.no_grad():
        out = model(x)
    expected = int(out.argmax(dim=1).item())
    try:
        subprocess.run(
            ["iverilog", "-g2012", "-o", "sim.vvp", "hdl/transformer_classifier.v", "hdl/transformer_classifier_tb.v"],
            check=True,
        )
        sim = subprocess.run(["vvp", "sim.vvp"], check=True, capture_output=True, text=True)
        hw_class = None
        for line in sim.stdout.splitlines():
            if line.startswith("class_out="):
                hw_class = int(line.split("=")[1])
        assert hw_class is not None, "simulation did not produce class_out"
    except FileNotFoundError:
        raise RuntimeError("iverilog not installed")
    assert hw_class == expected
