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
