#!/usr/bin/env python3
"""Utility to run SynapseX assembly programs or launch a small GUI.

This script supersedes the legacy ``chip.py`` entry point. Assembly programs
are stored in the ``asm/`` directory and can be executed either via the
command line or through a graphical interface.

Usage::

    python SynapseX.py gui
    python SynapseX.py train /path/to/train_data
    python SynapseX.py classify path/to/image.png
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk

from synapse.soc import SoC


def load_asm_file(path: str | Path) -> list[str]:
    """Read an assembly file and return a list of lines."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


class SynapseXGUI(tk.Tk):
    """Simple GUI to run assembly programs on the SoC."""

    def __init__(self) -> None:
        super().__init__()
        self.title("SynapseX")
        self.geometry("1000x600")
        self._build_ui()

    def _build_ui(self) -> None:
        paned = tk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=1)

        left = tk.Frame(paned)
        right = tk.Frame(paned, width=200)
        paned.add(left, stretch="always")
        paned.add(right)

        left_paned = tk.PanedWindow(left, orient=tk.VERTICAL)
        left_paned.pack(fill=tk.BOTH, expand=1)

        self.asm_text = tk.Text(left_paned, wrap="none")
        self.asm_text.tag_configure("instr", foreground="blue")
        left_paned.add(self.asm_text, stretch="always")

        self.results_nb = ttk.Notebook(left_paned)
        left_paned.add(self.results_nb)

        self.asm_list = tk.Listbox(right)
        self.asm_list.pack(fill=tk.BOTH, expand=1)
        self.asm_list.bind("<<ListboxSelect>>", self.on_select_asm)

        run_btn = tk.Button(right, text="Run", command=self.run_selected)
        run_btn.pack(fill=tk.X)

        self.data_entry = tk.Entry(right)
        self.data_entry.pack(fill=tk.X)
        browse = tk.Button(right, text="Train Data…", command=self.choose_data_dir)
        browse.pack(fill=tk.X)

        for asm_path in sorted(Path("asm").glob("*.asm")):
            self.asm_list.insert(tk.END, str(asm_path))

    def choose_data_dir(self) -> None:
        path = filedialog.askdirectory(title="Select Training Data Directory")
        if path:
            self.data_entry.delete(0, tk.END)
            self.data_entry.insert(0, path)

    def on_select_asm(self, _event) -> None:
        sel = self.asm_list.curselection()
        if not sel:
            return
        path = self.asm_list.get(sel[0])
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.asm_text.delete("1.0", tk.END)
        for line in lines:
            start = self.asm_text.index(tk.END)
            self.asm_text.insert(tk.END, line)
            tokens = line.strip().split()
            if tokens:
                end = f"{start}+{len(tokens[0])}c"
                self.asm_text.tag_add("instr", start, end)

    def run_selected(self) -> None:
        sel = self.asm_list.curselection()
        if not sel:
            return
        asm_path = Path(self.asm_list.get(sel[0]))
        train_dir = self.data_entry.get() or None
        soc = SoC(train_data_dir=train_dir)
        asm_lines = load_asm_file(asm_path)
        soc.load_assembly(asm_lines)
        buf = io.StringIO()
        with redirect_stdout(buf):
            soc.run(max_steps=3000)
        out = buf.getvalue()
        text = tk.Text(self.results_nb, wrap="word")
        text.insert(tk.END, out)
        self.results_nb.add(text, text=f"Run {len(self.results_nb.tabs())+1}")
        self.results_nb.select(text)


def main() -> None:
    if len(sys.argv) == 1 or sys.argv[1].lower() == "gui":
        gui = SynapseXGUI()
        gui.mainloop()
        return

    mode = sys.argv[1].lower()

    if mode == "train":
        if len(sys.argv) < 3:
            print("Usage: python SynapseX.py train /path/to/train_data")
            return
        train_dir = Path(sys.argv[2])
        if not train_dir.is_dir():
            print(f"Training data directory '{train_dir}' not found.")
            return
        soc = SoC(train_data_dir=str(train_dir))
        asm_lines = load_asm_file(Path("asm") / "training.asm")
        soc.load_assembly(asm_lines)
        soc.run(max_steps=3000)
        print("\nTraining Phase Completed!")
    elif mode == "classify":
        if len(sys.argv) < 3:
            print("Usage: python SynapseX.py classify path/to/image.png")
            return
        soc = SoC()
        asm_lines = load_asm_file(Path("asm") / "classification.asm")
        soc.load_assembly(asm_lines)
        soc.run(max_steps=3000)
        print("\nClassification Phase Completed!")
    else:
        print("Unknown mode. Use 'train', 'classify' or 'gui'.")


if __name__ == "__main__":
    main()

