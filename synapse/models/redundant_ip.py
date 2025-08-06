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

import numpy as np
import torch

from synapsex.config import HyperParameters, hp
from synapsex.genetic import genetic_search
from synapsex.neural import PyTorchANN
from synapsex.image_processing import load_process_shape_image


class RedundantNeuralIP:
    """Container for multiple ANNs addressable by an ID."""

    def __init__(self, train_data_dir: str | None = None, show_plots: bool = True) -> None:
        self.ann_map: Dict[int, PyTorchANN] = {}
        self.last_result: int | None = None
        self.train_data_dir = train_data_dir
        self._cached_dataset: tuple[np.ndarray, np.ndarray] | None = None
        self.show_plots = show_plots
        # figures generated during the last training run (unused for PyTorchANN)
        self.last_figures: List = []

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
        X, y = dataset

        # Update only the epoch count; GA-tuned learning rate and batch size are preserved
        ann.hp = replace(ann.hp, epochs=epochs)
        ann.train(torch.from_numpy(X), torch.from_numpy(y))

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
        in_dim = ann.hp.image_size
        data = []
        for i in range(in_dim):
            word = memory.read(addr + i)
            data.append(np.frombuffer(np.uint32(word).tobytes(), dtype=np.float32)[0])
        X = np.array(data, dtype=np.float32).reshape(1, -1)
        probs = ann.predict(torch.from_numpy(X), mc_dropout=len(tokens) > 1 and tokens[1].lower() == "true")
        self.last_result = int(probs.argmax(dim=1)[0])

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
            torch.from_numpy(X), torch.from_numpy(y),
            generations=generations, population_size=population,
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
            data_path = Path(self.train_data_dir) / "data.npy"
            labels_path = Path(self.train_data_dir) / "labels.npy"
            if not data_path.exists() or not labels_path.exists():
                X_list: List[np.ndarray] = []
                y_list: List[int] = []
                letter2label = {"A": 0, "B": 1, "C": 2}
                image_files = (
                    sorted(Path(self.train_data_dir).glob("*.png"))
                    + sorted(Path(self.train_data_dir).glob("*.jpg"))
                )
                for img_path in image_files:
                    letter = img_path.stem.split("_")[0].upper()
                    if letter not in letter2label:
                        continue
                    processed = load_process_shape_image(
                        str(img_path), out_dir=Path(self.train_data_dir) / "processed"
                    )
                    X_list.extend(processed)
                    y_list.extend([letter2label[letter]] * len(processed))
                if not X_list:
                    print("No training images found; aborting training.")
                    return None
                X = np.stack(X_list).astype(np.float32)
                y = np.array(y_list, dtype=np.int64)
                np.save(data_path, X)
                np.save(labels_path, y)
            else:
                X = np.load(data_path).astype(np.float32)
                y = np.load(labels_path).astype(np.int64)
            self._cached_dataset = (X, y)
        return self._cached_dataset

