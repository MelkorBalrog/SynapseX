import torch
from synapse.models.virtual_ann import VirtualANN


def test_virtual_ann_save_load_preserves_config(tmp_path):
    ann = VirtualANN([4, 3, 2], dropout_rate=0.25)
    # Ensure deterministic weights for comparison
    for param in ann.parameters():
        torch.nn.init.constant_(param, 0.5)
    path = tmp_path / "weights.pt"
    ann.save(str(path))

    ann2 = VirtualANN([2, 2], dropout_rate=0.1)
    ann2.load(str(path))

    assert ann2.layer_sizes == [4, 3, 2]
    assert ann2.dropout_rate == 0.25
    orig_state = ann.state_dict()
    new_state = ann2.state_dict()
    for k in orig_state:
        assert torch.allclose(orig_state[k], new_state[k])
