import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.append(os.getcwd())

from synapse.constants import IMAGE_BUFFER_BASE_ADDR_BYTES
from synapse.models.redundant_ip import RedundantNeuralIP
from synapse.hardware.memory import WishboneMemory


def test_ann_receives_written_image_data():
    ip = RedundantNeuralIP()
    img_size = 2
    img_channels = 3
    pattern = np.arange(img_channels * img_size * img_size, dtype=np.float32)

    class MockANN:
        def __init__(self, size, channels):
            self.hp = SimpleNamespace(image_size=size, image_channels=channels)
            self.last_tensor = None

        def predict(self, tensor, mc_dropout=False):
            self.last_tensor = tensor
            return torch.zeros((1, 1))

        def predict_class(self, tensor, mc_dropout=False):
            self.last_tensor = tensor
            return torch.tensor([0])

    ip.ann_map[0] = MockANN(img_size, img_channels)
    memory = WishboneMemory()
    base_words = IMAGE_BUFFER_BASE_ADDR_BYTES // 4
    for i, val in enumerate(pattern):
        word = np.frombuffer(np.float32(val).tobytes(), dtype=np.uint32)[0]
        memory.write(base_words + i, int(word))

    ip.run_instruction("INFER_ANN 0", memory=memory)
    assert np.allclose(ip.ann_map[0].last_tensor.numpy().flatten(), pattern)
