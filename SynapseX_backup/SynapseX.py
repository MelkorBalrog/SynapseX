#!/usr/bin/env python3
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
from tkinter.scrolledtext import ScrolledText
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

from synapsex.image_processing import load_process_shape_image

from synapse.soc import SoC


class ScrollableNotebook(ttk.Frame):
    """A ``ttk.Notebook`` with a horizontal scrollbar for overflowing tabs."""

    def __init__(self, master, **kwargs):
        super().__init__(master)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.h_scroll = ttk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(xscrollcommand=self.h_scroll.set)
        self.notebook = ttk.Notebook(self.canvas, **kwargs)
        self._window = self.canvas.create_window((0, 0), window=self.notebook, anchor="nw")
        self.canvas.pack(fill=tk.BOTH, expand=1)
        self.h_scroll.pack(fill=tk.X)
        self.notebook.bind("<Configure>", self._on_configure)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

    def _on_configure(self, _event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_resize(self, _event) -> None:
        self.canvas.itemconfigure(
            self._window,
            width=self.canvas.winfo_width(),
            height=self.canvas.winfo_height(),
        )

    # proxy common notebook methods
    def add(self, child, **kw):
        kw.setdefault("sticky", "nsew")
        return self.notebook.add(child, **kw)

    def tabs(self):
        return self.notebook.tabs()

    def select(self, tab=None):
        return self.notebook.select(tab)

    def nametowidget(self, name):
        return self.notebook.nametowidget(name)

    def bind(self, sequence=None, func=None, add=None):
        return self.notebook.bind(sequence, func, add)


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
        self.dark_mode = True
        self._build_menu()
        self._build_ui()
        self._set_dark_mode(self.dark_mode)
        self.dark_mode_btn.config(text="Light Mode")

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
        self.asm_line_numbers = tk.Text(
            self.asm_frame,
            width=4,
            padx=3,
            takefocus=0,
            borderwidth=0,
            highlightthickness=0,
            state="disabled",
            font=("Consolas", 11),
        )
        self.asm_text = tk.Text(self.asm_frame, wrap="none", font=("Consolas", 11))
        self.asm_text.tag_configure("instr", foreground="#0066CC")
        self.asm_text.tag_configure("number", foreground="#CC0000")
        self.asm_text.tag_configure("comment", foreground="#008000")
        self.asm_text.tag_configure("register", foreground="#FFA500")
        self.asm_text.bind("<<Modified>>", self._on_asm_modified)
        x_scroll = ttk.Scrollbar(self.asm_frame, orient="horizontal", command=self.asm_text.xview)
        self.asm_vscroll = ttk.Scrollbar(
            self.asm_frame, orient="vertical", command=self._on_asm_scroll
        )
        self.asm_text.configure(
            xscrollcommand=x_scroll.set, yscrollcommand=self._on_asm_yview
        )
        self.asm_line_numbers.configure(yscrollcommand=self._on_asm_yview)
        self.asm_line_numbers.grid(row=0, column=0, sticky="ns")
        self.asm_text.grid(row=0, column=1, sticky="nsew")
        self.asm_vscroll.grid(row=0, column=2, sticky="ns")
        x_scroll.grid(row=1, column=1, sticky="ew")
        self.asm_frame.rowconfigure(0, weight=1)
        self.asm_frame.columnconfigure(1, weight=1)
        left_paned.add(self.asm_frame, weight=3)

        self.results_nb = ScrollableNotebook(left_paned)
        left_paned.add(self.results_nb, weight=2)
        self.network_tabs: dict[str, ScrollableNotebook] = {}

        ttk.Label(right, text="Assembly Programs").pack(anchor="w", padx=5, pady=(5, 0))
        self.asm_tree = ttk.Treeview(right, show="tree")
        self.asm_tree.pack(fill=tk.BOTH, expand=1, padx=5)
        self.asm_tree.bind("<<TreeviewSelect>>", self.on_select_asm)

        ttk.Separator(right, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        run_btn = ttk.Button(right, text="Run Program", command=self.run_selected)
        run_btn.pack(fill=tk.X, padx=5)

        self.dark_mode_btn = ttk.Button(
            right, text="Dark Mode", command=self.toggle_dark_mode
        )
        self.dark_mode_btn.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(right, text="Training Data").pack(anchor="w", padx=5, pady=(10, 0))
        self.data_entry = ttk.Entry(right)
        self.data_entry.pack(fill=tk.X, padx=5)
        browse = ttk.Button(right, text="Browse…", command=self.choose_data_dir)
        browse.pack(fill=tk.X, padx=5, pady=5)

        for asm_path in sorted(Path("asm").glob("*.asm")):
            self.asm_tree.insert("", tk.END, iid=str(asm_path), text=asm_path.name)
        self._update_line_numbers()

    def _create_scrolled_text(self, parent: tk.Widget) -> tuple[ttk.Frame, tk.Text]:
        """Return a text widget with horizontal and vertical scrollbars."""
        frame = ttk.Frame(parent)
        text = tk.Text(frame, wrap="none", font=("Segoe UI", 10))
        x_scroll = ttk.Scrollbar(frame, orient="horizontal", command=text.xview)
        y_scroll = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        text.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)
        text.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        return frame, text

    def _on_asm_scroll(self, *args) -> None:
        """Scroll assembly text and line numbers together."""
        self.asm_text.yview(*args)
        self.asm_line_numbers.yview(*args)

    def _on_asm_yview(self, *args) -> None:
        """Update scrollbar and line numbers when text widget scrolls."""
        self.asm_vscroll.set(*args)
        self.asm_line_numbers.yview_moveto(args[0])

    def _update_line_numbers(self) -> None:
        """Refresh line numbers for the assembly text widget."""
        line_count = int(self.asm_text.index("end-1c").split(".")[0])
        numbers = "\n".join(str(i) for i in range(1, line_count + 1))
        self.asm_line_numbers.configure(state="normal")
        self.asm_line_numbers.delete("1.0", tk.END)
        if numbers:
            self.asm_line_numbers.insert("1.0", numbers)
        self.asm_line_numbers.configure(state="disabled")

    def _create_scrolled_figure(self, parent: tk.Widget, fig: Figure) -> ttk.Frame:
        """Return a frame that displays ``fig`` with horizontal and vertical scrollbars."""
        frame = ttk.Frame(parent)
        canvas = tk.Canvas(frame, highlightthickness=0)
        x_scroll = ttk.Scrollbar(frame, orient="horizontal", command=canvas.xview)
        y_scroll = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        canvas.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        fig_canvas = FigureCanvasTkAgg(fig, canvas)
        fig_canvas.draw()
        widget = fig_canvas.get_tk_widget()
        canvas.create_window((0, 0), window=widget, anchor="nw")

        def _on_configure(_event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        widget.bind("<Configure>", _on_configure)
        return frame

    def _set_dark_mode(self, enable: bool) -> None:
        if enable:
            bg = "#1e1e1e"
            fg = "#d4d4d4"
            instr = "#569CD6"
            number = "#FF00FF"
            comment = "#6A9955"
        else:
            bg = "white"
            fg = "black"
            instr = "#0066CC"
            number = "#CC0000"
            comment = "#008000"
        self.asm_text.configure(background=bg, foreground=fg, insertbackground=fg)
        self.asm_text.tag_configure("instr", foreground=instr)
        self.asm_text.tag_configure("number", foreground=number)
        self.asm_text.tag_configure("comment", foreground=comment)

    def toggle_dark_mode(self) -> None:
        self.dark_mode = not self.dark_mode
        self._set_dark_mode(self.dark_mode)
        self.dark_mode_btn.config(
            text="Light Mode" if self.dark_mode else "Dark Mode"
        )
        self._highlight_asm()

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
        processed = load_process_shape_image(path, angles=[0])[0]
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
            sub_nb = ScrollableNotebook(self.results_nb)
            self.results_nb.add(sub_nb, text="Classification")
            self.network_tabs["Classification"] = sub_nb
        sub_nb = self.network_tabs["Classification"]
        frame, text = self._create_scrolled_text(sub_nb)
        text.insert(tk.END, out + f"\nPredicted class: {result}\n")
        text.config(state="disabled")
        sub_nb.add(frame, text=f"Run {len(sub_nb.tabs())+1}")

        img_arr = processed.reshape(28, 28)
        fig = Figure(figsize=(2, 2))
        ax = fig.add_subplot(111)
        ax.imshow(img_arr, cmap="gray")
        ax.axis("off")
        try:
            fig_frame = self._create_scrolled_figure(sub_nb, fig)
            sub_nb.add(fig_frame, text="Processed")
        except tk.TclError:
            pass

        sub_nb.select(frame)
        self.results_nb.select(sub_nb)

    def menu_save_report(self) -> None:
        current = self.results_nb.select()
        if not current:
            return
        widget = self.nametowidget(current)
        if isinstance(widget, ScrollableNotebook):
            sub_widget = widget.nametowidget(widget.select())
        elif isinstance(widget, ttk.Notebook):
            sub_widget = widget.nametowidget(widget.select())
        else:
            sub_widget = widget
        if isinstance(sub_widget, ttk.Frame):
            for child in sub_widget.winfo_children():
                if isinstance(child, tk.Text):
                    sub_widget = child
                    break
            else:
                sub_widget = None
        if not isinstance(sub_widget, tk.Text):
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
        self._update_line_numbers()
        text = self.asm_text.get("1.0", tk.END)
        self.asm_text.tag_remove("instr", "1.0", tk.END)
        self.asm_text.tag_remove("number", "1.0", tk.END)
        self.asm_text.tag_remove("comment", "1.0", tk.END)
        self.asm_text.tag_remove("register", "1.0", tk.END)
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
            for match in re.finditer(r"\$[A-Za-z0-9]+", code):
                reg_start = f"{line_no}.{match.start()}"
                reg_end = f"{line_no}.{match.end()}"
                self.asm_text.tag_add("register", reg_start, reg_end)
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
        soc = SoC(train_data_dir=train_dir)
        asm_lines = load_asm_file(asm_path)
        soc.load_assembly(asm_lines)
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                soc.run(max_steps=3000)
        except tk.TclError as exc:
            buf.write(f"\nTk error: {exc}\n")
        out = buf.getvalue()
        if soc.neural_ip.last_result is not None:
            out += f"\nPredicted class: {soc.neural_ip.last_result}\n"
        net_name = asm_path.stem
        if net_name not in self.network_tabs:
            sub_nb = ScrollableNotebook(self.results_nb)
            self.results_nb.add(sub_nb, text=net_name)
            self.network_tabs[net_name] = sub_nb
        sub_nb = self.network_tabs[net_name]
        frame, text = self._create_scrolled_text(sub_nb)
        text.insert(tk.END, out)
        text.config(state="disabled")
        sub_nb.add(frame, text=f"Run {len(sub_nb.tabs())+1}")
        sub_nb.select(frame)
        self.results_nb.select(sub_nb)

        # Add generated figures for each ANN as tabs within its own notebook
        tab_titles = ["Metrics", "Weights", "Confusion"]
        for ann_id, figs in soc.neural_ip.figures_by_ann.items():
            key = f"ANN {ann_id}"
            if key not in self.network_tabs:
                ann_nb = ttk.Notebook(self.results_nb)
                self.results_nb.add(ann_nb, text=key)
                self.network_tabs[key] = ann_nb
            ann_nb = self.network_tabs[key]

            # Remove old tabs for this ANN to free memory
            for tab in ann_nb.tabs():
                widget = ann_nb.nametowidget(tab)
                ann_nb.forget(tab)
                widget.destroy()

            metrics = soc.neural_ip.metrics_by_ann.get(ann_id)
            if metrics:
                metric_txt = ScrolledText(ann_nb, wrap="word", font=("Segoe UI", 10))
                for name, val in metrics.items():
                    metric_txt.insert(tk.END, f"{name}: {val:.4f}\n")
                metric_txt.config(state="disabled")
                ann_nb.add(metric_txt, text="Summary", sticky="nsew")
            for fig, title in zip(figs, tab_titles):
                try:
                    frame = self._create_scrolled_figure(ann_nb, fig)
                    ann_nb.add(frame, text=title, sticky="nsew")
                except tk.TclError:
                    pass

        soc.neural_ip.figures_by_ann.clear()
        soc.neural_ip.metrics_by_ann.clear()


def main() -> None:
    if len(sys.argv) == 1 or sys.argv[1].lower() == "gui":
        try:
            gui = SynapseXGUI()
        except tk.TclError as exc:
            print(f"GUI unavailable: {exc}")
            return
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
        image_path = Path(sys.argv[2])
        if not image_path.is_file():
            print(f"Image '{image_path}' not found.")
            return
        soc = SoC()
        processed = load_process_shape_image(str(image_path), angles=[0])[0]
        base_addr = 0x5000
        for i, val in enumerate(processed):
            word = np.frombuffer(np.float32(val).tobytes(), dtype=np.uint32)[0]
            soc.memory.write(base_addr + i, int(word))
        asm_lines = load_asm_file(Path("asm") / "classification.asm")
        soc.load_assembly(asm_lines)
        soc.run(max_steps=3000)
        result = soc.cpu.get_reg("$t9")
        print(f"\nClassification Phase Completed!\nPredicted class: {result}")
    else:
        print("Unknown mode. Use 'train', 'classify' or 'gui'.")


if __name__ == "__main__":
    main()

