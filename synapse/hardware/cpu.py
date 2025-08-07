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
        self.data_map = {}

    # ------------------------------------------------------------------
    def set_label_map(self, label_map):
        self.label_map = label_map

    def set_data_map(self, data_map):
        self.data_map = data_map

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
            if self.neural_ip.last_result is not None:
                result = self.get_reg("$t9")
                label_map = {0: "A", 1: "B", 2: "Unknown"}
                print(f"Final classification: {label_map.get(result, result)}")
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
            if self._to_signed(self.get_reg(rs)) > self._to_signed(self.get_reg(rt)):
                self.pc = self.label_map.get(label, self.pc)
        elif instr == "J":
            label = parts[1]
            self.pc = self.label_map.get(label, self.pc)
        elif instr == "SLL":
            rd = parts[1].rstrip(",")
            rt = parts[2].rstrip(",")
            sa = int(parts[3], 0)
            self.set_reg(rd, (self.get_reg(rt) << sa) & 0xFFFFFFFF)
        elif instr == "LW":
            rt = parts[1].rstrip(",")
            addr_token = parts[2]
            addr = self._resolve_addr(addr_token) // 4
            self.set_reg(rt, self.memory.read(addr))
        elif instr == "SW":
            rt = parts[1].rstrip(",")
            addr_token = parts[2]
            addr = self._resolve_addr(addr_token) // 4
            self.memory.write(addr, self.get_reg(rt))
        elif instr == "OP_NEUR":
            subcmd = " ".join(parts[1:])
            res = self.neural_ip.run_instruction(subcmd, memory=self.memory)
            if res is not None:
                self.set_reg("$t9", int(res))

    def _resolve_addr(self, token: str) -> int:
        token = token.strip()
        if token.endswith(")") and "(" in token:
            base, reg = token[:-1].split("(")
            addr = self.data_map.get(base, 0)
            addr += self.get_reg(reg)
            return addr
        if "+" in token:
            label, imm = token.split("+")
            addr = self.data_map.get(label, 0) + int(imm, 0)
            return addr
        if token in self.data_map:
            return self.data_map[token]
        return int(token, 0)

    def _to_signed(self, val: int) -> int:
        return val if val < 0x80000000 else val - 0x100000000

