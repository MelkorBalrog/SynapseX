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
