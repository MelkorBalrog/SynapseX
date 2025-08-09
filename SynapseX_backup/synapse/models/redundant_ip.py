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

"""Management of multiple ANNs with majority voting.

The instruction processor mirrors the behaviour expected by the assembly
programs.  It now delegates work to :class:`synapsex.neural.PyTorchANN`
instances so that optimisation features such as the genetic algorithm can be
invoked directly from assembly via new ``TUNE_GA`` commands.
"""

from __future__ import annotations

import logging
import sys

from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch

from synapsex.config import HyperParameters, hp
from synapsex.genetic import genetic_search
from synapsex.neural import PyTorchANN
from synapsex.image_processing import load_process_shape_image


logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class RedundantNeuralIP:
    """Container for multiple ANNs addressable by an ID."""

    def __init__(self, train_data_dir: str | None = None) -> None:
        self.ann_map: Dict[int, PyTorchANN] = {}
        self.last_result: int | None = None
        self.train_data_dir = train_data_dir
        self._cached_dataset: tuple[np.ndarray, np.ndarray, List[str]] | None = None
        self.class_names: List[str] | None = None
        # Metrics and figures generated during training keyed by ANN ID
        self.metrics_by_ann: Dict[int, Dict[str, float]] = {}
        self.figures_by_ann: Dict[int, List] = {}

    # ------------------------------------------------------------------
    # Assembly interface
    # ------------------------------------------------------------------
    def run_instruction(self, subcmd: str, memory=None) -> None:
        """Parse and execute an ``OP_NEUR`` instruction."""
        tokens = subcmd.strip().split()
        if not tokens:
            return
        op = tokens[0].upper()
        if op == "CONFIG_ANN":
            self._config_ann(tokens[1:])
        elif op == "TUNE_GA":
            self._tune_ga(tokens[1:])
        elif op == "TRAIN_ANN":
            self._train_ann(tokens[1:])
        elif op == "INFER_ANN":
            self._infer_ann(tokens[1:], memory)
        elif op == "SAVE_ALL":
            prefix = tokens[1] if len(tokens) > 1 else "weights"
            for ann_id, ann in self.ann_map.items():
                ann.save(f"{prefix}_{ann_id}.pt")
        elif op == "LOAD_ALL":
            prefix = tokens[1] if len(tokens) > 1 else "weights"
            for ann_id, ann in self.ann_map.items():
                try:
                    ann.load(f"{prefix}_{ann_id}.pt")
                except FileNotFoundError:
                    pass

    # ------------------------------------------------------------------
    # CONFIG_ANN helpers
    # ------------------------------------------------------------------
    def _config_ann(self, tokens: List[str]) -> None:
        if len(tokens) < 2:
            return
        ann_id = int(tokens[0])
        cmd = tokens[1]
        # Legacy layer instructions are ignored; only FINALIZE is required to create the ANN
        if cmd == "FINALIZE":
            dropout = float(tokens[2]) if len(tokens) >= 3 else hp.dropout
            hparams = HyperParameters(**{**hp.__dict__, "dropout": dropout})
            self.ann_map[ann_id] = PyTorchANN(hparams)

    # ------------------------------------------------------------------
    # TRAIN_ANN helpers
    # ------------------------------------------------------------------
    def _train_ann(self, tokens: List[str]) -> None:
        if not tokens:
            return
        ann_id = int(tokens[0])
        epochs = int(tokens[1]) if len(tokens) > 1 else 5
        lr = float(tokens[2]) if len(tokens) > 2 else 0.005
        batch_size = int(tokens[3]) if len(tokens) > 3 else 16
        ann = self.ann_map.get(ann_id)
        if ann is None:
            return

        dataset = self._load_dataset()
        if dataset is None:
            return
        X, y, _ = dataset

        # Ensure ANN reflects current dataset class count
        ann.hp = replace(
            ann.hp,
            epochs=epochs,
            num_classes=hp.num_classes,
            image_channels=hp.image_channels,
        )
        ann = PyTorchANN(ann.hp, device=ann.device)
        self.ann_map[ann_id] = ann
        metrics, figs = ann.train(torch.from_numpy(X), torch.from_numpy(y))
        for old in self.figures_by_ann.get(ann_id, []):
            plt.close(old)
        self.figures_by_ann[ann_id] = figs
        self.metrics_by_ann[ann_id] = metrics
        print(f"ANN {ann_id} metrics: {metrics}")

    # ------------------------------------------------------------------
    # INFER_ANN helpers
    # ------------------------------------------------------------------
    def _infer_ann(self, tokens: List[str], memory) -> None:
        if not tokens:
            return
        ann_id = int(tokens[0])
        ann = self.ann_map.get(ann_id)
        if ann is None:
            return
        addr = 0x5000
        in_dim = ann.hp.image_size * ann.hp.image_size
        data: List[float] = []
        for i in range(in_dim):
            word = memory.read(addr + i)
            data.append(np.frombuffer(np.uint32(word).tobytes(), dtype=np.float32)[0])
        X = np.array(data, dtype=np.float32).reshape(1, -1)
        probs = ann.predict(
            torch.from_numpy(X),
            mc_dropout=len(tokens) > 1 and tokens[1].lower() == "true",
        )
        self.last_result = int(probs.argmax(dim=1)[0])
        print(f"ANN {ann_id} prediction: {self.last_result}")

    # ------------------------------------------------------------------
    # TUNE_GA helpers
    # ------------------------------------------------------------------
    def _tune_ga(self, tokens: List[str]) -> None:
        if not tokens:
            return
        ann_id = int(tokens[0])
        generations = int(tokens[1]) if len(tokens) > 1 else 5
        population = int(tokens[2]) if len(tokens) > 2 else 8

        dataset = self._load_dataset()
        if dataset is None:
            return
        X, y, _ = dataset

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        best_hp, best_ann = genetic_search(
            torch.from_numpy(X),
            torch.from_numpy(y),
            generations=generations,
            population_size=population,
            device=device,
        )
        self.ann_map[ann_id] = best_ann

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def predict_majority(self, X: np.ndarray):
        preds = {}
        for ann_id, ann in self.ann_map.items():
            probs = ann.predict(torch.from_numpy(X))
            preds[ann_id] = probs.argmax(dim=1).numpy()
        votes = [preds[ann_id][0] for ann_id in self.ann_map]
        majority = Counter(votes).most_common(1)[0][0]
        return majority, preds

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_dataset(self):
        if self._cached_dataset is None:
            if not self.train_data_dir:
                print("No training data directory specified; aborting training.")
                return None
            logger.info("Building dataset from %s", self.train_data_dir)

            data_path = Path(self.train_data_dir) / "data.npy"
            labels_path = Path(self.train_data_dir) / "labels.npy"
            classes_path = Path(self.train_data_dir) / "classes.npy"
            if not data_path.exists() or not labels_path.exists() or not classes_path.exists():
                X_list: List[np.ndarray] = []
                y_list: List[int] = []
                class_names: List[str] = []
                label_to_idx: Dict[str, int] = {}
                # Support both subdirectory-per-class and flat filename_prefix schemes
                image_paths = []
                base_dir = Path(self.train_data_dir)
                subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
                if subdirs:
                    for idx, d in enumerate(sorted(subdirs)):
                        label_to_idx[d.name] = idx
                        class_names.append(d.name)
                        image_paths.extend(
                            sorted(d.glob("*.png")) + sorted(d.glob("*.jpg"))
                        )
                else:
                    image_paths = (
                        sorted(base_dir.glob("*.png")) + sorted(base_dir.glob("*.jpg"))
                    )
                    for img in image_paths:
                        label = img.stem.split("_")[0]
                        if label not in label_to_idx:
                            label_to_idx[label] = len(label_to_idx)
                            class_names.append(label)
                for img_path in image_paths:
                    if img_path.parent == base_dir:
                        label = img_path.stem.split("_")[0]
                    else:
                        label = img_path.parent.name
                    idx = label_to_idx[label]
                    logger.info("Processing %s", img_path)
                    processed = load_process_shape_image(str(img_path))
                    X_list.extend(processed)
                    y_list.extend([idx] * len(processed))
                if not X_list:
                    print("No training images found; aborting training.")
                    return None
                X = np.stack(X_list).astype(np.float32)
                y = np.array(y_list, dtype=np.int64)
                np.save(data_path, X)
                np.save(labels_path, y)
                np.save(classes_path, np.array(class_names, dtype=object))
                logger.info(
                    "Processed %d images across %d classes",
                    len(X_list),
                    len(class_names),
                )
            else:
                logger.info("Loading cached dataset from %s", self.train_data_dir)
                X = np.load(data_path).astype(np.float32)
                y = np.load(labels_path).astype(np.int64)
                class_names = np.load(classes_path, allow_pickle=True).tolist()
            hp.num_classes = len(class_names)
            self.class_names = class_names
            self._cached_dataset = (X, y, class_names)
        return self._cached_dataset

