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

from typing import Tuple
import torch
from torch import nn


class TransformerClassifier(nn.Module):
    """Simple transformer-based classifier used by SynapseX."""

    def __init__(
        self,
        image_size: int,
        num_classes: int = 3,
        dropout: float = 0.2,
        *,
        num_layers: int = 2,
        nhead: int = 4,
    ):
        super().__init__()
        patch_size = max(image_size // 4, 1)
        self.patch_size = patch_size
        embed_dim = patch_size * patch_size
        if embed_dim % nhead != 0:
            for candidate in range(nhead, 0, -1):
                if embed_dim % candidate == 0:
                    nhead = candidate
                    break
        self.nhead = nhead
        self.patch_embed = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        n_patches = (image_size // patch_size) ** 2
        # Positional embeddings for each patch
        self.pos_embed = nn.Parameter(torch.zeros(n_patches, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        # Use batch_first=True so inputs are (batch, sequence, feature).
        # This avoids nested tensor warnings and enables better performance.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=self.nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
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
