"""Simplified CPU model used by SynapseX."""
from .memory import WishboneMemory
from .videoproc import VideoProcIP
from .pcie import PCIeBridge
from .mmu import MMU


class CPU:
    def __init__(self, name, video_ip, neural_ip, memory, mmu):
        self.name = name
        self.video_ip = video_ip
        self.neural_ip = neural_ip
        self.memory = memory
        self.mmu = mmu
        self.pc = 0
        self.running = True
        self.regs = {"$zero": 0, "$t0": 0, "$t1": 0, "$t2": 0, "$t9": 0}
        self.label_map = {}

    def set_label_map(self, label_map):
        self.label_map = label_map

    def get_reg(self, r):
        return self.regs.get(r, 0)

    def set_reg(self, r, val):
        if r != "$zero":
            self.regs[r] = val & 0xFFFFFFFF

    def step(self, program):
        if not self.running:
            return
        if self.pc >= len(program):
            self.running = False
            return
        line = program[self.pc].strip()
        self.pc += 1
        if line.startswith(";") or line == "" or line.endswith(":"):
            return
        parts = line.split()
        instr = parts[0].upper()
        if instr == "HALT":
            self.running = False
        # Additional instructions could be implemented here as needed for simulation
