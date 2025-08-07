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

from dataclasses import dataclass

@dataclass
class HyperParameters:
    image_size: int = 28
    image_channels: int = 1
    num_classes: int = 3
    dropout: float = 0.2
    learning_rate: float = 1e-3
    epochs: int = 10
    batch_size: int = 32
    mc_dropout_passes: int = 10
    num_layers: int = 2
    nhead: int = 4

hp = HyperParameters()
