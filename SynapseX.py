"""Main entry point for SynapseX refactored with PyTorch."""

import argparse
import os
from typing import List

import numpy as np
import torch

from synapsex.config import hp
from synapsex.image_processing import load_process_shape_image
from synapsex.neural import RedundantNeuralIP


LETTER_MAP = {"A": 0, "B": 1, "C": 2}
INV_LETTER_MAP = {v: k for k, v in LETTER_MAP.items()}


def build_dataset(folder: str) -> tuple:
    X: List[np.ndarray] = []
    y: List[int] = []
    for fn in os.listdir(folder):
        if not fn.lower().endswith((".png", ".jpg")):
            continue
        letter = fn.split("_")[0].upper()
        if letter not in LETTER_MAP:
            continue
        path = os.path.join(folder, fn)
        processed = load_process_shape_image(path, target_size=hp.image_size, save=False)
        X.append(processed[0])  # only keep first rotation
        y.append(LETTER_MAP[letter])
    X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return X_tensor, y_tensor


def train() -> None:
    dataset_path = "train_data"
    X, y = build_dataset(dataset_path)
    ip = RedundantNeuralIP()
    for ann_id in range(3):
        ip.create_ann(ann_id)
        ip.train_ann(ann_id, X, y)
    ip.save_all("weights")
    print("Training complete.")


def classify(image_path: str) -> None:
    processed = load_process_shape_image(image_path, target_size=hp.image_size, save=False)[0]
    X = torch.tensor(processed, dtype=torch.float32).unsqueeze(0)
    ip = RedundantNeuralIP()
    for ann_id in range(3):
        ip.create_ann(ann_id)
    ip.load_all("weights")
    pred, prob = ip.majority_vote(X)
    letter = INV_LETTER_MAP.get(pred, "Unknown")
    conf = float(prob.max())
    print(f"Prediction: {letter} (confidence {conf:.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="SynapseX refactored")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("train", help="train the models")
    classify_p = sub.add_parser("classify", help="classify an image")
    classify_p.add_argument("image", help="path to image")
    args = parser.parse_args()

    if args.cmd == "train":
        train()
    elif args.cmd == "classify":
        classify(args.image)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
