from dataclasses import dataclass

@dataclass
class HyperParameters:
    image_size: int = 28
    dropout: float = 0.2
    learning_rate: float = 1e-3
    epochs: int = 10
    batch_size: int = 32
    mc_dropout_passes: int = 10

hp = HyperParameters()
