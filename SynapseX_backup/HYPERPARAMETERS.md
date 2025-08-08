<!--
Copyright (C) 2025 Miguel Marina
Author: Miguel Marina <karel.capek.robotics@gmail.com>
LinkedIn: https://www.linkedin.com/in/progman32/

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
-->

# Hyperparameter Reference

SynapseX's machine-learning components are configurable through a set of hyperparameters. These parameters are defined in `config/hyperparameters.py` and the `HyperParameters` dataclass in `synapsex/config.py`.

## config/hyperparameters.py

| Name | Default | Description |
|------|---------|-------------|
| `BATCH_SIZE` | `32` | Number of samples processed in each optimisation step. |
| `LEARNING_RATE` | `0.001` | Step size used by the Adam optimiser during training. |
| `EPOCHS` | `10` | Number of passes through the training dataset. |
| `DROPOUT_RATE` | `0.2` | Dropout probability applied to fully connected layers during training. |
| `MC_PASSES` | `10` | Stochastic forward passes run during Monte Carlo dropout inference. |
| `LAYER_SIZES` | `[784, 256, 256, 3]` | Example dense network architecture listing neurons per layer. |
| `IMAGE_SIDE` | `28` | Height and width (in pixels) of input images. |
| `IMAGE_SIZE` | `784` | Flattened input dimension derived from `IMAGE_SIDE`. |
| `NUM_CLASSES` | `3` | Number of target classes. |
| `DROPOUT` | `0.2` | Dropout rate for the transformer classifier. |
| `TRAIN_DATA_DIR` | `"train_data"` | Default directory containing training images. |
| `WEIGHTS_DIR` | `"weights"` | Location where trained model checkpoints are saved. |

Lower-case aliases (`image_size`, `num_classes`, `dropout`) mirror their upper-case counterparts for backwards compatibility.

## synapsex/config.py

The `HyperParameters` dataclass exposes similar knobs for runtime configuration.

| Field | Default | Description |
|-------|---------|-------------|
| `image_size` | `28` | Side length of square input images provided to the Transformer classifier. |
| `dropout` | `0.2` | Dropout probability applied to model layers. |
| `learning_rate` | `1e-3` | Optimiser step size during training. |
| `epochs` | `10` | Number of training epochs. |
| `batch_size` | `32` | Mini-batch size for the `DataLoader`. |
| `mc_dropout_passes` | `10` | Number of forward passes to average when using Monte Carlo dropout. |
| `num_layers` | `2` | Number of transformer encoder layers used by the classifier. |
| `nhead` | `4` | Attention heads per transformer layer. Adjusted to divide the embedding dimension. |

## Genetic Algorithm Search

The hyper-parameters above form the search space for the built-in genetic
algorithm implemented in `synapsex/genetic.py`. The GA mutates dropout, learning
rate, transformer depth and attention heads, evaluating each candidate on a
validation split and keeping the configuration with the strongest F1 score.

## Target Metrics

Training tracks accuracy, precision, recall and F1 each epoch. Early stopping
uses validation F1 as the target metric and restores the weights from the best
scoring epoch.
