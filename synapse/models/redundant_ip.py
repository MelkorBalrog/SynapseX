"""Management of multiple ANNs with majority voting.

This module now exposes a small instruction processor which mirrors the
behaviour expected by the assembly programs.  The processor understands a
subset of the commands from the original project and delegates the actual
neural network work to :class:`VirtualANN` instances.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np

from .virtual_ann import VirtualANN
from synapsex.image_processing import load_process_shape_image


class RedundantNeuralIP:
    """Container for multiple ANNs addressable by an ID."""

    def __init__(self, train_data_dir: str | None = None, collect_figures: bool = False) -> None:
        self.ann_map: Dict[int, VirtualANN] = {}
        self.layer_defs: Dict[int, List[int]] = {}
        self.last_result: int | None = None
        self.train_data_dir = train_data_dir
        self._cached_dataset: tuple[np.ndarray, np.ndarray] | None = None
        self.collect_figures = collect_figures
        self.figures: List[object] = []

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
        if cmd == "CREATE_LAYER" and len(tokens) >= 5:
            in_dim = int(tokens[2])
            out_dim = int(tokens[3])
            # activation token ignored in this simplified implementation
            self.layer_defs[ann_id] = [in_dim, out_dim]
        elif cmd == "ADD_LAYER" and len(tokens) >= 4:
            out_dim = int(tokens[2])
            self.layer_defs.setdefault(ann_id, []).append(out_dim)
        elif cmd == "FINALIZE":
            layers = self.layer_defs.get(ann_id)
            if layers and len(layers) >= 2:
                self.ann_map[ann_id] = VirtualANN(layers)

    # ------------------------------------------------------------------
    # TRAIN_ANN helpers
    # ------------------------------------------------------------------
    def _train_ann(self, tokens: List[str]) -> None:
        if not tokens:
            return
        ann_id = int(tokens[0])
        epochs = int(tokens[1]) if len(tokens) > 1 else 5
        ann = self.ann_map.get(ann_id)
        if ann is None:
            return
        in_dim = ann.layer_sizes[0]
        out_dim = ann.layer_sizes[-1]

        if self._cached_dataset is None:
            if not self.train_data_dir:
                print("No training data directory specified; aborting training.")
                return
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
                    X_list.append(processed[0])
                    y_list.append(letter2label[letter])
                if not X_list:
                    print("No training images found; aborting training.")
                    return
                X = np.stack(X_list).astype(np.float32)
                y = np.array(y_list, dtype=np.int64)
                np.save(data_path, X)
                np.save(labels_path, y)
            else:
                X = np.load(data_path).astype(np.float32)
                y = np.load(labels_path).astype(np.int64)
            self._cached_dataset = (X, y)
        else:
            X, y = self._cached_dataset

        if X.shape[1] != in_dim or y.max() >= out_dim:
            print("Training data dimensions do not match ANN configuration.")
            return

        figs = ann.train_model(
            X, y, epochs=epochs, lr=0.005, batch_size=16, return_figures=self.collect_figures
        )
        if self.collect_figures and figs:
            self.figures.extend(figs)

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
        in_dim = ann.layer_sizes[0]
        data = []
        for i in range(in_dim):
            word = memory.read(addr + i)
            data.append(np.frombuffer(np.uint32(word).tobytes(), dtype=np.float32)[0])
        X = np.array(data, dtype=np.float32).reshape(1, -1)
        pred = ann.predict(X)
        self.last_result = int(pred[0])

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def predict_majority(self, X: np.ndarray):
        preds = {}
        for ann_id, ann in self.ann_map.items():
            preds[ann_id] = ann.predict(X)
        votes = [preds[ann_id][0] for ann_id in self.ann_map]
        majority = Counter(votes).most_common(1)[0][0]
        return majority, preds

