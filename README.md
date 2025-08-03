# SynapseX

SynapseX is a small System-on-Chip (SoC) simulator paired with image-processing
and neural-network utilities.  It runs simple assembly programs to exercise a
virtual CPU and a neural accelerator used for training and classifying
hand-written characters.

## Features

- **Assembly-driven SoC.** Programs stored in the `asm/` directory control the
  emulated hardware or launch neural-network operations.
- **Graphical interface.** `python SynapseX.py gui` opens a Tk GUI to edit and
  execute assembly files, load images and inspect results.
- **Training and inference.** Neural networks can be trained on a directory of
  labelled images or used to classify a single image from the command line.

## Getting Started

```bash
pip install -r requirements.txt
python SynapseX.py gui                # launch the GUI
python SynapseX.py train data/        # train on images in data/
python SynapseX.py classify img.png   # classify an image
```

## Architecture

The project models a simple SoC composed of a CPU, memory and several hardware
blocks.  The CPU executes a small subset of MIPS‑like instructions and forwards
neural‑network commands to the accelerator via the `OP_NEUR` instruction.

```mermaid
graph TD
    CPU((CPU)) -->|load/store| Memory[Wishbone\nMemory]
    CPU --> MMU
    CPU --> PCIe[PCIe Bridge]
    CPU --> Video[VideoProc IP]
    CPU --> Neural[Redundant\nNeural IP]
    Neural -->|model weights| Memory
    Video -->|processed frames| Memory
```

## Execution Flow

During classification the script performs the following high level steps:

```mermaid
sequenceDiagram
    participant U as User
    participant S as SynapseX.py
    participant C as SoC
    participant N as Neural IP
    U->>S: classify image
    S->>C: load data & assembly
    C->>N: OP_NEUR INFER_ANN
    N-->>C: prediction in memory
    C-->>S: result in register $t9
    S-->>U: show predicted class
```

## Repository Layout

- `SynapseX.py` – entry point script and GUI
- `asm/` – example assembly programs
- `synapse/` – SoC and hardware models
- `synapsex/` – image processing and neural-network helpers

## License

This project is released under the terms of the [MIT License](LICENSE).

