"""Main entry point for the refactored SynapseX project."""
import sys
import numpy as np
from config import hyperparameters as hp
from synapse.soc import SoC
from synapse.models.virtual_ann import PyTorchANN


def build_soc_with_anns():
    soc = SoC()
    # Create a few transformer-based ANNs using the hyperparameters
    for ann_id in range(3):
        ann = PyTorchANN(hp.IMAGE_SIZE, num_classes=hp.NUM_CLASSES, dropout=hp.DROPOUT)
        soc.neural_ip.add_ann(ann_id, ann)
    return soc


def train_mode():
    soc = build_soc_with_anns()
    # Dummy dataset for demonstration (random data)
    X = np.random.rand(100, hp.IMAGE_SIZE).astype(np.float32)
    y = np.random.randint(0, hp.NUM_CLASSES, size=100)
    for ann_id in soc.neural_ip.ann_map:
        soc.neural_ip.train_ann(ann_id, X, y, epochs=hp.EPOCHS, lr=hp.LEARNING_RATE, batch_size=hp.BATCH_SIZE)
    print("Training complete")


def classify_mode():
    soc = build_soc_with_anns()
    # Normally we would load trained weights; for demo we use untrained models
    sample = np.random.rand(1, hp.IMAGE_SIZE).astype(np.float32)
    majority, preds = soc.neural_ip.predict_majority(sample, mc_passes=hp.MC_PASSES)
    print(f"Predicted class {majority} from votes {preds}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python SynapseX.py [train|classify]")
        return
    mode = sys.argv[1].lower()
    if mode == "train":
        train_mode()
    elif mode == "classify":
        classify_mode()
    else:
        print(f"Unknown mode {mode}")


if __name__ == "__main__":
    main()
