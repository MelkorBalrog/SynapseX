"""Wishbone-like memory module."""
import numpy as np


class WishboneMemory:
    def __init__(self, size_words=0x100000):
        self.size_words = size_words
        self.mem = np.zeros(size_words, dtype=np.uint32)

    def read(self, addr):
        if 0 <= addr < self.size_words:
            return int(self.mem[addr])
        raise ValueError(f"Address 0x{addr:X} out of range")

    def write(self, addr, data):
        if 0 <= addr < self.size_words:
            self.mem[addr] = data & 0xFFFFFFFF
        else:
            raise ValueError(f"Address 0x{addr:X} out of range")
