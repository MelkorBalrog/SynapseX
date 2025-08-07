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
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import numpy as np

from synapsex.image_processing import load_process_shape_image

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
        self.current_asm_path: Path | None = None
        self._build_menu()
        self._build_ui()
        # keep references to PhotoImage objects to avoid garbage collection
        self._figure_images: list[ImageTk.PhotoImage] = []

    def _build_menu(self) -> None:
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open ASM...", command=self.menu_open_asm)
        file_menu.add_command(label="Save ASM...", command=self.menu_save_asm)
        file_menu.add_separator()
        file_menu.add_command(label="Load Image...", command=self.menu_load_image)
        file_menu.add_command(label="Save Report...", command=self.menu_save_report)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)
        self.config(menu=menubar)

    def _build_ui(self) -> None:
        paned = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=1)

        left = ttk.Frame(paned)
        right = ttk.Frame(paned, width=240)
        paned.add(left, weight=3)
        paned.add(right, weight=1)

        left_paned = ttk.Panedwindow(left, orient=tk.VERTICAL)
        left_paned.pack(fill=tk.BOTH, expand=1)

        self.asm_frame = ttk.Frame(left_paned)
        self.asm_text = tk.Text(self.asm_frame, wrap="none", font=("Consolas", 11))
        self.asm_text.tag_configure("instr", foreground="#0066CC")
        self.asm_text.tag_configure("number", foreground="#CC0000")
        self.asm_text.tag_configure("comment", foreground="#008000")
        self.asm_text.bind("<<Modified>>", self._on_asm_modified)
        x_scroll = ttk.Scrollbar(self.asm_frame, orient="horizontal", command=self.asm_text.xview)
        y_scroll = ttk.Scrollbar(self.asm_frame, orient="vertical", command=self.asm_text.yview)
        self.asm_text.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)
        self.asm_text.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        self.asm_frame.rowconfigure(0, weight=1)
        self.asm_frame.columnconfigure(0, weight=1)
        left_paned.add(self.asm_frame, weight=3)

        self.results_nb = ttk.Notebook(left_paned)
        left_paned.add(self.results_nb, weight=2)
        self.network_tabs: dict[str, ttk.Notebook] = {}

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
        self.current_asm_path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()
        self.asm_text.delete("1.0", tk.END)
        self.asm_text.insert(tk.END, data)
        self._highlight_asm()

    def menu_open_asm(self) -> None:
        path = filedialog.askopenfilename(
            title="Open ASM Program",
            filetypes=[("ASM Files", "*.asm"), ("All Files", "*.*")],
        )
        if not path:
            return
        self.current_asm_path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()
        self.asm_text.delete("1.0", tk.END)
        self.asm_text.insert(tk.END, data)
        self._highlight_asm()

    def menu_save_asm(self) -> None:
        if self.current_asm_path is None:
            path = filedialog.asksaveasfilename(
                title="Save ASM Program",
                defaultextension=".asm",
                filetypes=[("ASM Files", "*.asm"), ("All Files", "*.*")],
            )
            if not path:
                return
            self.current_asm_path = Path(path)
        with open(self.current_asm_path, "w", encoding="utf-8") as f:
            f.write(self.asm_text.get("1.0", tk.END))

    def menu_load_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Image to Classify",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.bmp"),
                ("All Files", "*.*"),
            ],
        )
        if not path:
            return
        processed_dir = Path.cwd() / "processed"
        processed = load_process_shape_image(path, out_dir=processed_dir, angles=[0])[0]
        soc = SoC()
        base_addr = 0x5000
        for i, val in enumerate(processed):
            word = np.frombuffer(np.float32(val).tobytes(), dtype=np.uint32)[0]
            soc.memory.write(base_addr + i, int(word))
        asm_lines = load_asm_file(Path("asm") / "classification.asm")
        soc.load_assembly(asm_lines)
        buf = io.StringIO()
        with redirect_stdout(buf):
            soc.run(max_steps=3000)
        out = buf.getvalue()
        result = soc.cpu.get_reg("$t9")
        if "Classification" not in self.network_tabs:
            sub_nb = ttk.Notebook(self.results_nb)
            self.results_nb.add(sub_nb, text="Classification")
            self.network_tabs["Classification"] = sub_nb
        sub_nb = self.network_tabs["Classification"]
        text = ScrolledText(sub_nb, wrap="word", font=("Segoe UI", 10))
        text.insert(tk.END, out + f"\nPredicted class: {result}\n")
        text.config(state="disabled")
        sub_nb.add(text, text=f"Run {len(sub_nb.tabs())+1}")
        sub_nb.select(text)
        self.results_nb.select(sub_nb)

    def menu_save_report(self) -> None:
        current = self.results_nb.select()
        if not current:
            return
        widget = self.nametowidget(current)
        if isinstance(widget, ttk.Notebook):
            sub_widget = widget.nametowidget(widget.select())
        else:
            sub_widget = widget
        if not isinstance(sub_widget, (tk.Text, ScrolledText)):
            return
        content = sub_widget.get("1.0", tk.END)
        path = filedialog.asksaveasfilename(
            title="Save Report",
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

    def _on_asm_modified(self, _event) -> None:
        if self.asm_text.edit_modified():
            self.asm_text.edit_modified(False)
            self._highlight_asm()

    def _highlight_asm(self) -> None:
        text = self.asm_text.get("1.0", tk.END)
        self.asm_text.tag_remove("instr", "1.0", tk.END)
        self.asm_text.tag_remove("number", "1.0", tk.END)
        self.asm_text.tag_remove("comment", "1.0", tk.END)
        for line_no, line in enumerate(text.splitlines(), start=1):
            code, sep, _comment = line.partition(";")
            tokens = code.split()
            if tokens:
                token = tokens[0]
                col = code.find(token)
                if token.endswith(":") and len(tokens) > 1:
                    token = tokens[1]
                    col = code.find(token)
                start = f"{line_no}.{col}"
                end = f"{start}+{len(token)}c"
                self.asm_text.tag_add("instr", start, end)
            for match in re.finditer(r"\b-?(0x[0-9a-fA-F]+|\d+)\b", code):
                num_start = f"{line_no}.{match.start()}"
                num_end = f"{line_no}.{match.end()}"
                self.asm_text.tag_add("number", num_start, num_end)
            if sep:
                col = len(code)
                start = f"{line_no}.{col}"
                end = f"{line_no}.{len(line)}"
                self.asm_text.tag_add("comment", start, end)

    def run_selected(self) -> None:
        sel = self.asm_tree.selection()
        if not sel:
            return
        asm_path = Path(sel[0])
        train_dir = self.data_entry.get() or None
        soc = SoC(train_data_dir=train_dir, show_plots=False)
        asm_lines = load_asm_file(asm_path)
        soc.load_assembly(asm_lines)
        buf = io.StringIO()
        with redirect_stdout(buf):
            soc.run(max_steps=3000)
        out = buf.getvalue()
        net_name = asm_path.stem
        if net_name not in self.network_tabs:
            sub_nb = ttk.Notebook(self.results_nb)
            self.results_nb.add(sub_nb, text=net_name)
            self.network_tabs[net_name] = sub_nb
        sub_nb = self.network_tabs[net_name]
        text = ScrolledText(sub_nb, wrap="word", font=("Segoe UI", 10))
        text.insert(tk.END, out)
        text.config(state="disabled")
        sub_nb.add(text, text=f"Run {len(sub_nb.tabs())+1}")
        sub_nb.select(text)
        self.results_nb.select(sub_nb)

        # add generated figures as notebook tabs with scrollbars
        for fig in soc.neural_ip.last_figures:
            buf_img = io.BytesIO()
            fig.savefig(buf_img, format="png")
            buf_img.seek(0)
            image = Image.open(buf_img)
            photo = ImageTk.PhotoImage(image)

            frame = ttk.Frame(self.results_nb)
            canvas = tk.Canvas(frame, width=min(800, image.width), height=min(600, image.height))
            hbar = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=canvas.xview)
            vbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
            canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
            canvas.create_image(0, 0, image=photo, anchor="nw")
            canvas.configure(scrollregion=(0, 0, image.width, image.height))

            canvas.grid(row=0, column=0, sticky="nsew")
            vbar.grid(row=0, column=1, sticky="ns")
            hbar.grid(row=1, column=0, sticky="ew")
            frame.rowconfigure(0, weight=1)
            frame.columnconfigure(0, weight=1)

            self._figure_images.append(photo)
            self.results_nb.add(frame, text=f"Fig {len(self.results_nb.tabs()) + 1}")
            plt.close(fig)


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

