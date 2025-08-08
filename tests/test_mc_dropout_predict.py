import torch
from torch import nn

from synapsex.neural import PyTorchANN
from synapsex.config import HyperParameters


def test_consecutive_mc_dropout_calls_use_new_inputs(monkeypatch):
    class DummyNet(nn.Module):
        def __init__(self, image_size, num_classes, dropout, num_layers, nhead, in_channels):
            super().__init__()
            self.fc = nn.Linear(image_size * image_size * in_channels, num_classes)
            self.dropout = nn.Dropout(dropout)
            with torch.no_grad():
                self.fc.weight[:] = torch.tensor([[1.0], [-1.0]])
                self.fc.bias.zero_()

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            x = self.dropout(x)
            return x

    monkeypatch.setattr("synapsex.neural.TransformerClassifier", DummyNet)

    hp = HyperParameters(
        image_size=1,
        image_channels=1,
        num_classes=2,
        dropout=0.5,
        mc_dropout_passes=3,
        num_layers=1,
        nhead=1,
    )

    ann = PyTorchANN(hp_override=hp)
    x1 = torch.tensor([0.0])
    x2 = torch.tensor([1.0])

    torch.manual_seed(0)
    out1 = ann.predict(x1, mc_dropout=True)
    assert ann.model.training is False

    torch.manual_seed(0)
    out2 = ann.predict(x2, mc_dropout=True)
    assert ann.model.training is False

    assert not torch.allclose(out1, out2)
