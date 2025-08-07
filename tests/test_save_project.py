import json

import matplotlib.pyplot as plt

from synapse.models.redundant_ip import RedundantNeuralIP
from synapsex.neural import PyTorchANN


def test_save_project(tmp_path):
    ip = RedundantNeuralIP()
    ip.ann_map[0] = PyTorchANN()
    ip.metrics_by_ann[0] = {"accuracy": 0.5}
    fig = plt.figure()
    ip.figures_by_ann[0] = [fig]
    json_path = tmp_path / "project.json"
    ip.save_project(str(json_path), "test_weights")
    data = json.loads(json_path.read_text())
    assert "0" in data["anns"]
    ann_data = data["anns"]["0"]
    weight_file = tmp_path / ann_data["weights"]
    assert weight_file.exists()
    fig_file = tmp_path / ann_data["figures"][0]
    assert fig_file.exists()
    assert ann_data["metrics"]["accuracy"] == 0.5
