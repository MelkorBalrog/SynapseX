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
