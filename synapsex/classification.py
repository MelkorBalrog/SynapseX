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

"""Classification utilities driven by the assembly program.

These helpers mirror the exact preprocessing pipeline used during
training so that inference results remain consistent.  Images are
converted to edge maps via :func:`synapsex.image_processing.process_shape_image`
and fed into the `classification.asm` program to obtain predictions from
the configured neural networks.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

from synapse.soc import SoC
from synapse.constants import IMAGE_BUFFER_BASE_ADDR_BYTES
from synapsex.config import hp
from synapsex.image_processing import (
    load_process_shape_image,
    load_vehicle_dataset,
)


def _load_asm_file(path: str | Path) -> list[str]:
    """Read an assembly file and return a list of lines."""

    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def classify_with_assembly(
    image_path: str,
    *,
    angles: Iterable[int] = range(0, 360, 5),
    soc: SoC | None = None,
):
    """Classify ``image_path`` running the assembly program on all rotations.

    The preprocessing exactly matches the training pipeline to ensure
    consistent results.
    """

    soc = soc or SoC()
    asm_lines = _load_asm_file(Path("asm") / "classification.asm")
    soc.load_assembly(asm_lines)
    processed_list = load_process_shape_image(
        str(image_path), target_size=hp.image_size, angles=angles
    )
    base_addr = IMAGE_BUFFER_BASE_ADDR_BYTES // 4
    preds: list[int] = []
    for processed in processed_list:
        for i, val in enumerate(processed):
            word = np.frombuffer(np.float32(val).tobytes(), dtype=np.uint32)[0]
            soc.memory.write(base_addr + i, int(word))
        soc.cpu.pc = 0
        soc.cpu.running = True
        for reg in list(soc.cpu.regs):
            if reg != "$zero":
                soc.cpu.regs[reg] = 0
        soc.run(max_steps=3000)
        preds.append(soc.cpu.get_reg("$t9"))

    counts = Counter(preds)
    result = max(counts.items(), key=lambda kv: kv[1])[0]
    names = soc.neural_ip.class_names
    label = names[result] if names and 0 <= result < len(names) else result
    return result, label


def evaluate_with_assembly(
    train_dir: str,
    *,
    rotate: bool = True,
    soc: SoC | None = None,
):
    """Classify all images in ``train_dir`` using the assembly pipeline."""

    soc = soc or SoC(train_data_dir=train_dir)
    asm_lines = _load_asm_file(Path("asm") / "classification.asm")
    X, y, class_names = load_vehicle_dataset(
        train_dir, target_size=hp.image_size, rotate=rotate
    )
    soc.load_assembly(asm_lines)
    preds: list[int] = []
    base_addr = IMAGE_BUFFER_BASE_ADDR_BYTES // 4
    for img in X:
        flat = img.flatten().numpy()
        for i, val in enumerate(flat):
            word = np.frombuffer(np.float32(val).tobytes(), dtype=np.uint32)[0]
            soc.memory.write(base_addr + i, int(word))
        soc.cpu.pc = 0
        soc.cpu.running = True
        for reg in list(soc.cpu.regs):
            if reg != "$zero":
                soc.cpu.regs[reg] = 0
        soc.run(max_steps=3000)
        preds.append(soc.cpu.get_reg("$t9"))

    y_np = y.numpy()
    preds_np = np.array(preds)
    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_np, preds_np):
        cm[t, p] += 1
    precision_list = []
    recall_list = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision_list.append(tp / (tp + fp + 1e-8))
        recall_list.append(tp / (tp + fn + 1e-8))
    precision = float(sum(precision_list) / num_classes)
    recall = float(sum(recall_list) / num_classes)
    f1 = float(2 * precision * recall / (precision + recall + 1e-8))
    accuracy = float((preds_np == y_np).mean())
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    return metrics, cm


__all__ = ["classify_with_assembly", "evaluate_with_assembly"]

