"""System-on-chip model for SynapseX."""
from .hardware.memory import WishboneMemory
from .hardware.pcie import PCIeBridge
from .hardware.videoproc import VideoProcIP
from .hardware.mmu import MMU
from .hardware.cpu import CPU
from .models.redundant_ip import RedundantNeuralIP


class SoC:
    def __init__(self):
        self.memory = WishboneMemory()
        self.pcie_bridge = PCIeBridge()
        self.mmu = MMU()
        self.video_ip = VideoProcIP()
        self.neural_ip = RedundantNeuralIP()
        self.cpu = CPU("CPU1", self.video_ip, self.neural_ip, self.memory, self.mmu)
        self.asm_program = []
        self.label_map = {}

    def load_assembly(self, lines):
        self.asm_program = lines[:]

    def run(self, max_steps=100):
        for _ in range(max_steps):
            if self.cpu.running:
                self.cpu.step(self.asm_program)
            else:
                break
