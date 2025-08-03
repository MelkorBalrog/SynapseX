import os
import subprocess
import sys

import torch

sys.path.append(os.getcwd())
from synapsex.models import TransformerClassifier

def test_transformer_classifier_hw_match():
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
