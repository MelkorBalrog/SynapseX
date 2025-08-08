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
import torch

from .hardware.memory import WishboneMemory
from .hardware.pcie import PCIeBridge
from .hardware.videoproc import VideoProcIP
from .hardware.mmu import MMU
from .hardware.cpu import CPU
from .models.redundant_ip import RedundantNeuralIP


class SoC:
    def __init__(
        self,
        train_data_dir: str | None = None,
        device: torch.device | str | None = None,
    ):
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.memory = WishboneMemory()
        self.pcie_bridge = PCIeBridge()
        self.mmu = MMU()
        self.video_ip = VideoProcIP()
        self.neural_ip = RedundantNeuralIP(train_data_dir=train_data_dir, device=self.device)
        self.cpu = CPU("CPU1", self.video_ip, self.neural_ip, self.memory, self.mmu)
        # Ensure CPU starts in a clean state
        self.reset_cpu()
        self.asm_program = []
        self.label_map = {}
        self.data_map = {}

    # ------------------------------------------------------------------
    def reset_cpu(self):
        """Reset CPU state so each run starts clean."""
        self.cpu.pc = 0
        self.cpu.running = True
        for reg in list(self.cpu.regs.keys()):
            if reg != "$zero":
                self.cpu.set_reg(reg, 0)

    # ------------------------------------------------------------------
    def load_assembly(self, lines):
        """Load assembly program and build label map."""
        self.asm_program = lines[:]
        self._preprocess_labels()
        self._preprocess_data()
        self.cpu.set_label_map(self.label_map)
        self.cpu.set_data_map(self.data_map)
        self.reset_cpu()

    def _preprocess_labels(self):
        self.label_map = {}
        for idx, line in enumerate(self.asm_program):
            stripped = line.strip()
            if stripped.endswith(":"):
                self.label_map[stripped[:-1]] = idx

    def _preprocess_data(self):
        self.data_map = {}
        in_data = False
        addr = 0x4000
        for line in self.asm_program:
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
    def run(self, max_steps=100):
        self.reset_cpu()
        for _ in range(max_steps):
            if self.cpu.running:
                self.cpu.step(self.asm_program)
            else:
                break
