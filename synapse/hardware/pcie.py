"""Minimal PCIe bridge placeholder."""


class PCIeBridge:
    def __init__(self):
        self.reg = 0

    def read(self):
        return self.reg

    def write(self, data):
        self.reg = data & 0xFFFFFFFF
