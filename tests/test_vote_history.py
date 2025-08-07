import numpy as np
from synapse.models.redundant_ip import RedundantNeuralIP


class DummyMemory:
    def __init__(self, data):
        self.data = data

    def read(self, addr: int) -> int:
        idx = addr - 0x5000
        val = self.data[idx]
        return np.frombuffer(np.float32(val).tobytes(), dtype=np.uint32)[0]


def test_vote_history_records_predictions():
    ip = RedundantNeuralIP()
    ip.run_instruction("CONFIG_ANN 0 FINALIZE")
    ann = ip.ann_map[0]
    # Provide a zeroed input image
    data = np.zeros(ann.hp.image_size * ann.hp.image_size, dtype=np.float32)
    memory = DummyMemory(data)
    ip.run_instruction("INFER_ANN 0", memory=memory)
    assert ip.vote_history == [ip.last_result]
