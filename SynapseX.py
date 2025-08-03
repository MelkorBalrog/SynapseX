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
import re
from contextlib import redirect_stdout
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

try:  # Python <=3.10 ships scrolledtext as a submodule
    from tkinter.scrolledtext import ScrolledText
except Exception:  # pragma: no cover - fallback for some platforms
    import tkinter.scrolledtext as _scrolledtext
    ScrolledText = _scrolledtext.ScrolledText

from synapse.soc import SoC


def load_asm_file(path: str | Path) -> list[str]:
    """Read an assembly file and return a list of lines."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


class SynapseXGUI(tk.Tk):
    """GUI to run assembly programs on the SoC."""

    def __init__(self) -> None:
        super().__init__()
        self.title("SynapseX")
        self.geometry("1100x650")
        style = ttk.Style(self)
        style.theme_use("clam")
        self._build_ui()

    def _build_ui(self) -> None:
        paned = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=1)

        left = ttk.Frame(paned)
        right = ttk.Frame(paned, width=240)
        paned.add(left, weight=3)
        paned.add(right, weight=1)

        left_paned = ttk.Panedwindow(left, orient=tk.VERTICAL)
        left_paned.pack(fill=tk.BOTH, expand=1)

        self.asm_text = ScrolledText(left_paned, wrap="none", font=("Consolas", 11))
        self.asm_text.tag_configure("instr", foreground="#0066CC")
        self.asm_text.tag_configure("number", foreground="#CC0000")
        left_paned.add(self.asm_text, weight=3)

        self.results_nb = ttk.Notebook(left_paned)
        left_paned.add(self.results_nb, weight=2)

        ttk.Label(right, text="Assembly Programs").pack(anchor="w", padx=5, pady=(5, 0))
        self.asm_tree = ttk.Treeview(right, show="tree")
        self.asm_tree.pack(fill=tk.BOTH, expand=1, padx=5)
        self.asm_tree.bind("<<TreeviewSelect>>", self.on_select_asm)

        ttk.Separator(right, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        run_btn = ttk.Button(right, text="Run Program", command=self.run_selected)
        run_btn.pack(fill=tk.X, padx=5)

        ttk.Label(right, text="Training Data").pack(anchor="w", padx=5, pady=(10, 0))
        self.data_entry = ttk.Entry(right)
        self.data_entry.pack(fill=tk.X, padx=5)
        browse = ttk.Button(right, text="Browse…", command=self.choose_data_dir)
        browse.pack(fill=tk.X, padx=5, pady=5)

        for asm_path in sorted(Path("asm").glob("*.asm")):
            self.asm_tree.insert("", tk.END, iid=str(asm_path), text=asm_path.name)

    def choose_data_dir(self) -> None:
        path = filedialog.askdirectory(title="Select Training Data Directory")
        if path:
            self.data_entry.delete(0, tk.END)
            self.data_entry.insert(0, path)

    def on_select_asm(self, _event) -> None:
        sel = self.asm_tree.selection()
        if not sel:
            return
        path = sel[0]
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.asm_text.delete("1.0", tk.END)
        for line in lines:
            line_start = self.asm_text.index(tk.END)
            self.asm_text.insert(tk.END, line)
            for i, match in enumerate(re.finditer(r"\S+", line)):
                token = match.group(0).strip(",")
                token_start = f"{line_start}+{match.start()}c"
                token_end = f"{line_start}+{match.end()}c"
                if i == 0:
                    self.asm_text.tag_add("instr", token_start, token_end)
                elif re.fullmatch(r"-?(0x[0-9a-fA-F]+|\d+)", token):
                    self.asm_text.tag_add("number", token_start, token_end)

    def run_selected(self) -> None:
        sel = self.asm_tree.selection()
        if not sel:
            return
        asm_path = Path(sel[0])
        train_dir = self.data_entry.get() or None
        soc = SoC(train_data_dir=train_dir, collect_figures=True)
        asm_lines = load_asm_file(asm_path)
        soc.load_assembly(asm_lines)
        buf = io.StringIO()
        with redirect_stdout(buf):
            soc.run(max_steps=3000)
        out = buf.getvalue()
        run_idx = len(self.results_nb.tabs()) + 1
        text = ScrolledText(self.results_nb, wrap="word", font=("Segoe UI", 10))
        text.insert(tk.END, out)
        text.config(state="disabled")
        self.results_nb.add(text, text=f"Run {run_idx}")
        self.results_nb.select(text)
        for i, fig in enumerate(soc.neural_ip.figures, start=1):
            frame = ttk.Frame(self.results_nb)
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
            self.results_nb.add(frame, text=f"Run {run_idx} Fig {i}")
            self.results_nb.select(frame)
            plt.close(fig)
        soc.neural_ip.figures.clear()


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

