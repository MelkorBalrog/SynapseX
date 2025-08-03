"""System-on-chip model for SynapseX."""
from .hardware.memory import WishboneMemory
from .hardware.pcie import PCIeBridge
from .hardware.videoproc import VideoProcIP
from .hardware.mmu import MMU
from .hardware.cpu import CPU
from .models.redundant_ip import RedundantNeuralIP


class SoC:
    def __init__(self, train_data_dir: str | None = None, show_plots: bool = True):
        self.memory = WishboneMemory()
        self.pcie_bridge = PCIeBridge()
        self.mmu = MMU()
        self.video_ip = VideoProcIP()
        self.neural_ip = RedundantNeuralIP(train_data_dir=train_data_dir, show_plots=show_plots)
        self.cpu = CPU("CPU1", self.video_ip, self.neural_ip, self.memory, self.mmu)
        self.asm_program = []
        self.label_map = {}

    # ------------------------------------------------------------------
    def load_assembly(self, lines):
        """Load assembly program and build label map."""
        self.asm_program = lines[:]
        self._preprocess_labels()

    def _preprocess_labels(self):
        self.label_map = {}
        for idx, line in enumerate(self.asm_program):
            stripped = line.strip()
            if stripped.endswith(":"):
                self.label_map[stripped[:-1]] = idx
        self.cpu.set_label_map(self.label_map)

    # ------------------------------------------------------------------
    def run(self, max_steps=100):
        for _ in range(max_steps):
            if self.cpu.running:
                self.cpu.step(self.asm_program)
            else:
                break
