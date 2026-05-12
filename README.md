# MultiCue-DF — Multi-Cue Deepfake Detection

**Course:** CS-419 Deep Learning — SEECS, NUST  
**Dataset:** FaceForensics++ (C23 compression)

## Overview

MultiCue-DF is a deepfake video detection system that fuses multiple visual cues (spatial, frequency, and temporal) to classify videos as real or fake. It is trained on the FaceForensics++ benchmark covering six manipulation methods.

## Manipulation Methods

| Folder | Type |
|---|---|
| `original` | Real / pristine videos |
| `Deepfakes` | Identity swap via autoencoder |
| `Face2Face` | Facial reenactment |
| `FaceShifter` | High-fidelity face swap |
| `FaceSwap` | Face replacement |
| `NeuralTextures` | Texture-based reenactment |
| `DeepFakeDetection` | Google DFD dataset |

## Folder Structure

```
multicue-df/
├── FaceForensics++_C23/     # Raw video dataset (read-only)
│   ├── csv/                 # Metadata CSVs
│   ├── original/            # 1000 real videos
│   ├── Deepfakes/           # 1000 fake videos
│   ├── Face2Face/
│   ├── FaceShifter/
│   ├── FaceSwap/
│   ├── NeuralTextures/
│   └── DeepFakeDetection/
├── frames/                  # Extracted JPEG frames
│   ├── real/
│   └── fake/
│       ├── Deepfakes/
│       ├── Face2Face/
│       ├── FaceShifter/
│       ├── FaceSwap/
│       ├── NeuralTextures/
│       └── DeepFakeDetection/
├── src/                     # All Python source code
│   ├── config.py            # Central hyperparameter config
│   └── extract_frames.py    # Frame sampling & extraction
├── notebooks/               # Jupyter exploration notebooks
├── checkpoints/             # Saved model weights
├── logs/                    # Training logs & skipped video lists
├── results/                 # Evaluation outputs
├── plots/                   # Generated figures
└── requirements.txt
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Extract frames from videos (run from multicue-df/ root)
python src/extract_frames.py
```

## Configuration

All hyperparameters live in `src/config.py` — edit there to change sampling counts, image size, batch size, learning rate, train/val/test splits, and model dimensions.
