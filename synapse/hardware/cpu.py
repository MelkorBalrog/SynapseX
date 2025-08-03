"""Simplified CPU model used by SynapseX.

Only a subset of instructions required by the example assembly programs is
implemented.  The model is not intended to be a realistic CPU but merely a
driver for the neural IP and control flow instructions used in the demo.
"""


class CPU:
    def __init__(self, name, video_ip, neural_ip, memory, mmu):
        self.name = name
        self.video_ip = video_ip
        self.neural_ip = neural_ip
        self.memory = memory
        self.mmu = mmu
        self.pc = 0
        self.running = True
        # register file: $zero, $t0-$t12, $t9 (result), $at temporary
        self.regs = {"$zero": 0}
        for i in range(13):
            self.regs[f"$t{i}"] = 0
        self.regs["$t9"] = 0
        self.regs["$at"] = 0
        self.label_map = {}

    # ------------------------------------------------------------------
    def set_label_map(self, label_map):
        self.label_map = label_map

    def get_reg(self, r):
        return self.regs.get(r, 0)

    def set_reg(self, r, val):
        if r != "$zero":
            self.regs[r] = val & 0xFFFFFFFF

    # ------------------------------------------------------------------
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
        elif instr == "ADDI":
            rd = parts[1].rstrip(",")
            rs = parts[2].rstrip(",")
            imm = int(parts[3], 0)
            self.set_reg(rd, self.get_reg(rs) + imm)
        elif instr == "ADD":
            rd = parts[1].rstrip(",")
            rs = parts[2].rstrip(",")
            rt = parts[3]
            self.set_reg(rd, self.get_reg(rs) + self.get_reg(rt))
        elif instr == "BEQ":
            rs = parts[1].rstrip(",")
            rt = parts[2].rstrip(",")
            label = parts[3]
            if self.get_reg(rs) == self.get_reg(rt):
                self.pc = self.label_map.get(label, self.pc)
        elif instr == "BGT":
            rs = parts[1].rstrip(",")
            rt = parts[2].rstrip(",")
            label = parts[3]
            if self.get_reg(rs) > self.get_reg(rt):
                self.pc = self.label_map.get(label, self.pc)
        elif instr == "J":
            label = parts[1]
            self.pc = self.label_map.get(label, self.pc)
        elif instr == "OP_NEUR":
            subcmd = " ".join(parts[1:])
            self.neural_ip.run_instruction(subcmd, memory=self.memory)
            if subcmd.upper().startswith("INFER_ANN") and self.neural_ip.last_result is not None:
                self.set_reg("$t9", int(self.neural_ip.last_result))

