import pytest

from synapse.models.redundant_ip import RedundantNeuralIP
from synapsex.neural import PyTorchANN


def test_load_all_raises_when_weights_missing(tmp_path):
    ip = RedundantNeuralIP()
    ip.ann_map[0] = PyTorchANN()
    with pytest.raises(FileNotFoundError) as excinfo:
        ip.run_instruction(f"LOAD_ALL {tmp_path / 'weights'}")
    assert "ANN 0" in str(excinfo.value)
