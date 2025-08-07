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

"""Simple multi-object detection utilities for SynapseX.

The implementation below purposely keeps the network architecture compact so
that it can be used in unit tests without incurring a large computational
cost.  The detector follows a YOLO-style approach where a single forward pass
predicts bounding boxes and class scores for a fixed number of candidate
objects.
"""

from dataclasses import dataclass
from typing import List

import torch
from torch import nn


@dataclass
class Detection:
    """Represents one detected object."""

    bbox: torch.Tensor  # (4,) tensor with [x1, y1, x2, y2]
    scores: torch.Tensor  # (num_classes,) class scores


class MultiObjectDetector(nn.Module):
    """Very small convolutional object detector.

    The network accepts an RGB image of shape ``(N, 3, H, W)`` and outputs a
    tensor of shape ``(N, num_boxes, 4 + num_classes)`` containing bounding box
    coordinates and raw class scores for each candidate box.
    """

    def __init__(self, num_classes: int, num_boxes: int = 10, image_size: int = 64):
        super().__init__()
        self.num_classes = num_classes
        self.num_boxes = num_boxes

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.head = nn.Linear(64, num_boxes * (4 + num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass of the detector.

        Parameters
        ----------
        x: torch.Tensor
            Batch of RGB images of shape ``(N, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(N, num_boxes, 4 + num_classes)`` containing
            bounding box coordinates followed by class scores.
        """

        feats = self.features(x)
        feats = feats.flatten(1)
        out = self.head(feats)
        return out.view(x.shape[0], self.num_boxes, 4 + self.num_classes)


def detect_objects(model: MultiObjectDetector, image: torch.Tensor) -> List[Detection]:
    """Detect objects in an image using ``model``.

    Parameters
    ----------
    model: MultiObjectDetector
        The detector to use for generating predictions.
    image: torch.Tensor
        Either a single image of shape ``(3, H, W)`` or a batch of images of
        shape ``(N, 3, H, W)``.

    Returns
    -------
    List[Detection]
        Structured detections for the first image in the batch.  For testing
        purposes we only process a single image, but the model itself supports
        batches.
    """

    model.eval()
    with torch.no_grad():
        if image.dim() == 3:
            image = image.unsqueeze(0)
        preds = model(image)[0]  # consider first image only

    detections: List[Detection] = []
    for pred in preds:
        bbox = pred[:4]
        scores = pred[4:]
        detections.append(Detection(bbox=bbox, scores=scores))
    return detections
