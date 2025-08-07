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
"""Train a tiny multi-object detector on annotated datasets."""

from __future__ import annotations

import argparse
from typing import List

import torch

from synapsex.config import HyperParameters
from synapsex.neural import PyTorchANN
from synapsex.object_detection import MultiObjectDetector
from synapsex.image_processing import load_annotated_dataset


def collate(samples: List):
    images, boxes, labels = zip(*samples)
    return torch.stack(images), torch.cat(boxes), torch.cat(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a simple detector")
    parser.add_argument("data", help="Path to YOLO/COCO dataset root")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    samples = load_annotated_dataset(args.data)
    images, boxes, labels = collate(samples)

    num_classes = int(labels.max().item()) + 1
    hp = HyperParameters(num_classes=num_classes, image_channels=3)
    ann = PyTorchANN(hp)
    ann.model = MultiObjectDetector(num_classes=num_classes)

    optim = torch.optim.Adam(ann.model.parameters(), lr=hp.learning_rate)
    for epoch in range(args.epochs):
        losses = ann.train_detector(images, {"boxes": boxes, "labels": labels}, optimizer=optim)
        print(f"Epoch {epoch+1}: cls={losses['classification_loss']:.4f} reg={losses['regression_loss']:.4f}")

    ann.save("detector.pt")
    print("Saved detector.pt")


if __name__ == "__main__":
    main()
