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

"""Hyperparameter configuration for SynapseX."""

# Training hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10
DROPOUT_RATE = 0.2
MC_PASSES = 10

# Model architecture
LAYER_SIZES = [784, 256, 256, 3]  # Example architecture for 28x28 input and 3 classes

# Transformer-specific settings
IMAGE_SIDE = 28
IMAGE_SIZE = IMAGE_SIDE * IMAGE_SIDE  # flattened input dimension
NUM_CLASSES = 3
DROPOUT = 0.2

# Data/weights locations
TRAIN_DATA_DIR = "train_data"
WEIGHTS_DIR = "weights"

# Lower-case aliases for compatibility with older code
image_size = IMAGE_SIZE
num_classes = NUM_CLASSES
dropout = DROPOUT
