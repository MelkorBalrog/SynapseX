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
