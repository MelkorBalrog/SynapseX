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

from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Dict, List
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch

from synapsex.config import HyperParameters, hp
from synapsex.genetic import genetic_search
from synapsex.neural import PyTorchANN
from synapsex.image_processing import load_vehicle_dataset


class RedundantNeuralIP:
    """Container for multiple ANNs addressable by an ID."""

    def __init__(self, train_data_dir: str | None = None) -> None:
        self.ann_map: Dict[int, PyTorchANN] = {}
        self.last_result: int | None = None
        self.train_data_dir = train_data_dir
        self._cached_dataset: tuple[torch.Tensor, torch.Tensor] | None = None
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
        elif op == "SAVE_PROJECT":
            json_path = tokens[1] if len(tokens) > 1 else "project.json"
            prefix = tokens[2] if len(tokens) > 2 else "weights"
            self.save_project(json_path, prefix)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save_project(self, json_path: str, weight_prefix: str = "weights") -> None:
        """Serialise all ANNs, metrics and figures to ``json_path``.

        The network weights remain in individual ``.pt`` files referenced by the
        generated JSON so that large binary blobs do not bloat the manifest. Any
        training figures are saved as PNG images and likewise referenced.
        """

        base = Path(json_path).resolve().parent
        project: Dict[str, Dict[str, object]] = {}

        for ann_id, ann in self.ann_map.items():
            weight_file = f"{weight_prefix}_{ann_id}.pt"
            ann.save(str(base / weight_file))

            fig_paths: List[str] = []
            for idx, fig in enumerate(self.figures_by_ann.get(ann_id, [])):
                fig_name = f"{weight_prefix}_{ann_id}_fig{idx}.png"
                fig.savefig(base / fig_name)
                plt.close(fig)
                fig_paths.append(fig_name)

            project[str(ann_id)] = {
                "weights": weight_file,
                "metrics": self.metrics_by_ann.get(ann_id, {}),
                "figures": fig_paths,
            }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"anns": project}, f, indent=2)

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
        X, y = dataset

        # Update only the epoch count; GA-tuned learning rate and batch size are preserved
        ann.hp = replace(ann.hp, epochs=epochs)
        metrics, figs = ann.train(X, y)
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
        X, y = dataset

        best_hp, best_ann = genetic_search(
            X,
            y,
            generations=generations,
            population_size=population,
        )
        self.ann_map[ann_id] = best_ann

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def predict_majority(self, X: np.ndarray):
        preds = {}
        for ann_id, ann in self.ann_map.items():
            probs = ann.predict(torch.from_numpy(X))
            preds[ann_id] = probs.argmax(dim=1).cpu().numpy()

        # ``Counter`` struggles with NumPy scalar types, so convert each vote to
        # a native ``int`` before tallying the results.
        votes = [int(preds[ann_id][0]) for ann_id in self.ann_map]
        majority = int(Counter(votes).most_common(1)[0][0]) if votes else None
        return majority, preds

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_dataset(self):
        if self._cached_dataset is None:
            if not self.train_data_dir:
                print("No training data directory specified; aborting training.")
                return None
            data_path = Path(self.train_data_dir) / "data.npy"
            labels_path = Path(self.train_data_dir) / "labels.npy"

            X = y = None
            if data_path.exists() and labels_path.exists():
                X = torch.from_numpy(np.load(data_path).astype(np.float32))
                y = torch.from_numpy(np.load(labels_path).astype(np.int64))
                # Rotate augmentation produces 72 images per original sample.
                # If the cached dataset size is not a multiple of 72 it was
                # likely generated without rotation, so regenerate to ensure
                # the confusion matrix accounts for all augmented images.
                if X.shape[0] % 72 != 0:
                    X = y = None

            if X is None or y is None:
                X, y = load_vehicle_dataset(self.train_data_dir, hp.image_size)
                np.save(data_path, X.numpy())
                np.save(labels_path, y.numpy())

            self._cached_dataset = (X, y)
        return self._cached_dataset

