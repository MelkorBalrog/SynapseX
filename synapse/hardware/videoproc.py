"""Simple video processing IP block."""


class VideoProcIP:
    def __init__(self):
        self.start = False
        self.in_addr = 0
        self.out_addr = 0x1200
        self.done = False

    def do_cycle(self, memory):
        if self.start and not self.done:
            raw = memory.read(self.in_addr)
            processed = (raw + 0x100) & 0xFFFFFFFF
            memory.write(self.out_addr, processed)
            self.done = True

    def read_status(self):
        return 1 if self.done else 0

    def reset(self):
        self.start = False
        self.done = False
