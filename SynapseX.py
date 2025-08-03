"""Main entry point for the refactored SynapseX project."""
import sys
import os
import numpy as np
from config import hyperparameters as hp
from synapse.soc import SoC
from synapse.models.virtual_ann import PyTorchANN
from synapse.utils.image_processing import load_and_preprocess


def build_soc_with_anns():
    soc = SoC()
    # Create a few transformer-based ANNs using the hyperparameters
    for ann_id in range(3):
        ann = PyTorchANN(hp.IMAGE_SIZE, num_classes=hp.NUM_CLASSES, dropout=hp.DROPOUT)
        soc.neural_ip.add_ann(ann_id, ann)
    return soc


def _load_training_data():
    class_map = {"A": 0, "B": 1, "C": 2}
    X_list, y_list = [], []
    if not os.path.isdir(hp.TRAIN_DATA_DIR):
        raise FileNotFoundError(f"Training data folder '{hp.TRAIN_DATA_DIR}' not found")
    for fn in os.listdir(hp.TRAIN_DATA_DIR):
        if fn.lower().endswith((".png", ".jpg", ".jpeg")):
            label = fn.split("_")[0].upper()
            if label in class_map:
                path = os.path.join(hp.TRAIN_DATA_DIR, fn)
                X_list.append(load_and_preprocess(path, hp.IMAGE_SIDE))
                y_list.append(class_map[label])
    if not X_list:
        raise RuntimeError("No training images found in train_data")
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=int)


def train_mode():
    soc = build_soc_with_anns()
    X, y = _load_training_data()
    os.makedirs(hp.WEIGHTS_DIR, exist_ok=True)
    for ann_id, ann in soc.neural_ip.ann_map.items():
        soc.neural_ip.train_ann(ann_id, X, y, epochs=hp.EPOCHS, lr=hp.LEARNING_RATE, batch_size=hp.BATCH_SIZE)
        weight_path = os.path.join(hp.WEIGHTS_DIR, f"ann{ann_id}.pt")
        ann.save(weight_path)
    print("Training complete")


def classify_mode(img_path: str):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image '{img_path}' not found")
    soc = build_soc_with_anns()
    for ann_id, ann in soc.neural_ip.ann_map.items():
        weight_path = os.path.join(hp.WEIGHTS_DIR, f"ann{ann_id}.pt")
        if os.path.exists(weight_path):
            ann.load(weight_path)
        else:
            print(f"Warning: weights for ANN{ann_id} not found; using untrained model")
    sample = load_and_preprocess(img_path, hp.IMAGE_SIDE).reshape(1, -1)
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
        if len(sys.argv) < 3:
            print("Usage: python SynapseX.py classify path/to/image.png")
            return
        classify_mode(sys.argv[2])
    else:
        print(f"Unknown mode {mode}")


if __name__ == "__main__":
    main()
