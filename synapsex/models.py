from typing import Tuple
import torch
from torch import nn


class TransformerClassifier(nn.Module):
    """Simple transformer-based classifier used by SynapseX."""

    def __init__(self, image_size: int, num_classes: int = 3, dropout: float = 0.2):
        super().__init__()
        patch_size = max(image_size // 4, 1)
        self.patch_size = patch_size
        embed_dim = patch_size * patch_size
        self.patch_embed = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        n_patches = (image_size // patch_size) ** 2
        # Positional embeddings for each patch
        self.pos_embed = nn.Parameter(torch.zeros(n_patches, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(n_patches * embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, H, W)
        x = self.patch_embed(x)  # (batch, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (batch, n_patches, embed_dim)
        x = x + self.pos_embed  # add positional information
        x = self.transformer(x)  # (batch, n_patches, embed_dim)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.head(x)
