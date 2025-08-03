#!/usr/bin/env python3
"""Utility to run SynapseX assembly programs from external ``.asm`` files.

This script supersedes the legacy ``chip.py`` entry point. It loads
assembly instructions from text files to program the System-on-Chip (SoC)
model. Two programs are provided:
  * ``asm/training.asm`` for training the on-chip ANNs.
  * ``asm/classification.asm`` for running inference and majority voting.

Usage:
    python SynapseX.py train
    python SynapseX.py classify path/to/image.png

The SoC and CPU models are simplified and primarily execute the assembly
instructions for configuration. Image handling for classification is left
as an exercise; this refactor focuses on moving the assembly programs out
of the Python source and into standalone ``.asm`` files.
"""

from __future__ import annotations

import sys
from pathlib import Path

from synapse.soc import SoC


def load_asm_file(path: str | Path) -> list[str]:
    """Read an assembly file and return a list of lines."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage:\n  python SynapseX.py train\n  python SynapseX.py classify path/to/image.png")
        return

    mode = sys.argv[1].lower()
    soc = SoC()

    if mode == "train":
        asm_lines = load_asm_file(Path("asm") / "training.asm")
        soc.load_assembly(asm_lines)
        soc.run(max_steps=3000)
        print("\nTraining Phase Completed!")
    elif mode == "classify":
        if len(sys.argv) < 3:
            print("Usage: python SynapseX.py classify path/to/image.png")
            return
        # Placeholder for image handling; assembly handles ANN operations.
        asm_lines = load_asm_file(Path("asm") / "classification.asm")
        soc.load_assembly(asm_lines)
        soc.run(max_steps=3000)
        print("\nClassification Phase Completed!")
    else:
        print(f"Unknown mode: {mode}. Use 'train' or 'classify'.")


if __name__ == "__main__":
    main()
