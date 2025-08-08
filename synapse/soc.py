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

"""System-on-chip model for SynapseX."""
from .hardware.memory import WishboneMemory
from .hardware.pcie import PCIeBridge
from .hardware.videoproc import VideoProcIP
from .hardware.mmu import MMU
from .hardware.cpu import CPU
from .models.redundant_ip import RedundantNeuralIP


class SoC:
    def __init__(self, train_data_dir: str | None = None):
        self.memory = WishboneMemory()
        self.pcie_bridge = PCIeBridge()
        self.mmu = MMU()
        self.video_ip = VideoProcIP()
        self.neural_ip = RedundantNeuralIP(train_data_dir=train_data_dir)
        self.cpu = CPU("CPU1", self.video_ip, self.neural_ip, self.memory, self.mmu)
        # Ensure result register starts clean to avoid stale predictions
        self.cpu.set_reg("$t9", 0)
        self.asm_program = []
        self.label_map = {}
        self.data_map = {}

    # ------------------------------------------------------------------
    def load_assembly(self, lines):
        """Load assembly program and build label map.

        Lines are stored with 1-based line numbers to aid debugging."""
        self.asm_program = list(enumerate(lines, start=1))
        self._preprocess_labels()
        self._preprocess_data()
        self.cpu.set_label_map(self.label_map)
        self.cpu.set_data_map(self.data_map)

    def _preprocess_labels(self):
        self.label_map = {}
        for idx, (_, line) in enumerate(self.asm_program):
            stripped = line.strip()
            if stripped.endswith(":"):
                self.label_map[stripped[:-1]] = idx

    def _preprocess_data(self):
        self.data_map = {}
        in_data = False
        addr = 0x4000
        for _, line in self.asm_program:
            stripped = line.strip()
            if stripped == ".data":
                in_data = True
                continue
            if not in_data:
                continue
            if stripped.endswith(":"):
                label = stripped[:-1]
                self.data_map[label] = addr
            elif stripped.startswith(".space"):
                size = int(stripped.split()[1])
                addr += size

    # ------------------------------------------------------------------
    def run(self, max_steps=100, debug: bool = False):
        for _ in range(max_steps):
            if self.cpu.running:
                self.cpu.step(self.asm_program, debug=debug)
            else:
                break

    def debug_run(self, max_steps=100):
        """Interactively step through the assembly program."""
        for _ in range(max_steps):
            if not self.cpu.running:
                break
            self.cpu.step(self.asm_program, debug=True)
            input("Press Enter to continue...")
