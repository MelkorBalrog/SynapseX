"""Minimal pass-through MMU."""


class MMU:
    def __init__(self):
        self.tlb = {}

    def translate(self, va, mem_write=False):
        return (va, False)
