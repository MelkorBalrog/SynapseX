"""PyTorch implementation of the virtual ANN used by SynapseX."""
from typing import List
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class VirtualANN(nn.Module):
    def __init__(self, layer_sizes: List[int], dropout_rate: float = 0.2):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i + 1]
            layers.append(nn.Linear(in_dim, out_dim))
            if i < len(layer_sizes) - 2:
                layers.append(nn.LeakyReLU())
                layers.append(nn.Dropout(dropout_rate))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def train_model(self, X: np.ndarray, y: np.ndarray, epochs: int, lr: float, batch_size: int):
        dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()
        for _ in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                preds = self(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

    def predict(self, X: np.ndarray):
        self.eval()
        with torch.no_grad():
            logits = self(torch.from_numpy(X).float())
            return torch.argmax(logits, dim=1).cpu().numpy()

    def predict_with_uncertainty(self, X: np.ndarray, mc_passes: int = 10):
        self.train()  # enable dropout at inference
        outputs = []
        inp = torch.from_numpy(X).float()
        for _ in range(mc_passes):
            outputs.append(self(inp).detach().cpu().numpy())
        mean = np.mean(outputs, axis=0)
        var = np.var(outputs, axis=0)
        return mean, var


class TransformerClassifier(nn.Module):
    """Simple transformer-based classifier with adjustable embedding dimension."""

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.1, nhead: int = 4):
        super().__init__()
        embed_dim = ((input_dim + nhead - 1) // nhead) * nhead
        self.proj = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.unsqueeze(0)  # (1, batch, embed_dim)
        x = self.encoder(x)
        x = x.squeeze(0)
        return self.classifier(x)


class PyTorchANN:
    """Wrapper around ``TransformerClassifier`` providing training and inference helpers."""

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.1, nhead: int = 4):
        self.model = TransformerClassifier(input_dim, num_classes, dropout, nhead)

    def train_model(self, X: np.ndarray, y: np.ndarray, epochs: int, lr: float, batch_size: int):
        dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                preds = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

    def predict(self, X: np.ndarray):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(X).float())
            return torch.argmax(logits, dim=1).cpu().numpy()

    def predict_with_uncertainty(self, X: np.ndarray, mc_passes: int = 10):
        self.model.train()  # enable dropout at inference
        outputs = []
        inp = torch.from_numpy(X).float()
        for _ in range(mc_passes):
            outputs.append(self.model(inp).detach().cpu().numpy())
        mean = np.mean(outputs, axis=0)
        var = np.var(outputs, axis=0)
        return mean, var
