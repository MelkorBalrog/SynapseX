from dataclasses import dataclass

@dataclass
class HyperParameters:
    image_size: int = 28
    dropout: float = 0.2
    learning_rate: float = 1e-3
    # Minimum epochs before early stopping is considered
    epochs: int = 10
    max_epochs: int = 1000
    batch_size: int = 32
    mc_dropout_passes: int = 10
    # Target metrics required to stop training
    target_accuracy: float = 0.95
    target_precision: float = 0.95
    target_recall: float = 0.95
    target_f1: float = 0.95
    mutate_patience: int = 5
    mutation_std: float = 0.1
    lr_decay_factor: float = 0.5
    lr_decay_patience: int = 3

hp = HyperParameters()
