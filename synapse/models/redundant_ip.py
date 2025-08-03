"""Management of multiple ANNs with majority voting.

This module now exposes a small instruction processor which mirrors the
behaviour expected by the assembly programs.  The processor understands a
subset of the commands from the original project and delegates the actual
neural network work to :class:`VirtualANN` instances.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List

import numpy as np

from .virtual_ann import VirtualANN


class RedundantNeuralIP:
    """Container for multiple ANNs addressable by an ID."""

    def __init__(self) -> None:
        self.ann_map: Dict[int, VirtualANN] = {}
        self.layer_defs: Dict[int, List[int]] = {}
        self.last_result: int | None = None

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
        # create a small random dataset purely for demonstration
        X = np.random.rand(20, in_dim).astype(np.float32)
        y = np.random.randint(0, out_dim, size=20).astype(np.int64)
        ann.train_model(X, y, epochs=epochs, lr=0.005, batch_size=16)

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

