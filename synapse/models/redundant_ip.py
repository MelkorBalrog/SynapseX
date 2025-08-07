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

"""Management of multiple ANNs for assembly-driven majority voting.

The instruction processor mirrors the behaviour expected by the assembly
programs.  It now delegates work to :class:`synapsex.neural.PyTorchANN`
instances so that optimisation features such as the genetic algorithm can be
invoked directly from assembly via new ``TUNE_GA`` commands.
"""

from __future__ import annotations

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
from synapse.constants import IMAGE_BUFFER_BASE_ADDR_BYTES


class RedundantNeuralIP:
    """Container for multiple ANNs addressable by an ID."""

    def __init__(self, train_data_dir: str | None = None) -> None:
        self.ann_map: Dict[int, PyTorchANN] = {}
        self.last_result: int | None = None
        self._argmax: Dict[int, int] = {}
        self.vote_history: List[int] = []
        self.train_data_dir = train_data_dir
        self._cached_dataset: tuple[torch.Tensor, torch.Tensor, List[str]] | None = None
        self.class_names: List[str] = []
        # Metrics and figures generated during training keyed by ANN ID
        self.metrics_by_ann: Dict[int, Dict[str, float]] = {}
        self.figures_by_ann: Dict[int, List] = {}

    # ------------------------------------------------------------------
    # Assembly interface
    # ------------------------------------------------------------------
    def run_instruction(self, subcmd: str, memory=None) -> int | None:
        """Parse and execute an ``OP_NEUR`` instruction."""
        tokens = subcmd.strip().split()
        if not tokens:
            return None
        result: int | None = None
        op = tokens[0].upper()
        if op == "CONFIG_ANN":
            self._config_ann(tokens[1:])
        elif op == "TUNE_GA":
            self._tune_ga(tokens[1:])
        elif op == "TRAIN_ANN":
            self._train_ann(tokens[1:])
        elif op == "INFER_ANN":
            result = self._infer_ann(tokens[1:], memory)
        elif op == "GET_NUM_CLASSES":
            if hp.num_classes == 0:
                raise ValueError(
                    "hp.num_classes is 0; load training metadata or configure an ANN first"
                )
            result = hp.num_classes
        elif op == "GET_ARGMAX":
            if len(tokens) > 1:
                ann_id = int(tokens[1])
                result = self._argmax.get(ann_id, 0)
        elif op == "SAVE_ALL":
            prefix = tokens[1] if len(tokens) > 1 else "weights"
            for ann_id, ann in self.ann_map.items():
                ann.save(f"{prefix}_{ann_id}.pt")
            meta_path = Path(f"{prefix}_meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"num_classes": hp.num_classes}, f)
        elif op == "LOAD_ALL":
            prefix = tokens[1] if len(tokens) > 1 else "weights"
            for ann_id, ann in self.ann_map.items():
                try:
                    ann.load(f"{prefix}_{ann_id}.pt")
                except FileNotFoundError:
                    pass
            meta_path = Path(f"{prefix}_meta.json")
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    hp.num_classes = int(data.get("num_classes", hp.num_classes))
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
        elif op == "SAVE_PROJECT":
            json_path = tokens[1] if len(tokens) > 1 else "project.json"
            prefix = tokens[2] if len(tokens) > 2 else "weights"
            self.save_project(json_path, prefix)
        if result is not None:
            self.last_result = result
        else:
            self.last_result = None
        return result

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
            json.dump({"anns": project, "num_classes": hp.num_classes}, f, indent=2)

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
            meta_prefix = tokens[3] if len(tokens) >= 4 else None
            if meta_prefix:
                meta_path = Path(f"{meta_prefix}_meta.json")
                if meta_path.exists():
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        hp.num_classes = int(data.get("num_classes", hp.num_classes))
                    except (OSError, ValueError, json.JSONDecodeError):
                        pass
            if self.train_data_dir:
                self._load_dataset()
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
        addr_bytes = IMAGE_BUFFER_BASE_ADDR_BYTES
        addr = addr_bytes // 4
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
        result = int(probs.argmax(dim=1)[0])
        self._argmax[ann_id] = result
        self.vote_history.append(result)
        print(f"ANN {ann_id} prediction: {result}")
        return result

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

        best_hp, best_ann = genetic_search(
            X,
            y,
            generations=generations,
            population_size=population,
        )
        self.ann_map[ann_id] = best_ann

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
            if not data_path.exists() or not labels_path.exists():
                X, y, class_names = load_vehicle_dataset(self.train_data_dir, hp.image_size)
                np.save(data_path, X.numpy())
                np.save(labels_path, y.numpy())
            else:
                X = torch.from_numpy(np.load(data_path).astype(np.float32))
                y = torch.from_numpy(np.load(labels_path).astype(np.int64))
                class_names = self.class_names or []
            hp.num_classes = int(torch.unique(y).numel())
            self.class_names = class_names
            self._cached_dataset = (X, y, class_names)
        return self._cached_dataset

