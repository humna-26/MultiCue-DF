# MultiCue-DF: Multi-Cue Deepfake Detection

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)
![Institution](https://img.shields.io/badge/SEECS-NUST-darkblue)

MultiCue-DF is a three-stream deep learning system for deepfake video detection, trained on the FaceForensics++ C23 benchmark. It fuses complementary visual evidence from three parallel CNN streams — full-face spatial features (ResNet-18), eye/mouth landmark regions (custom CNN), and FFT frequency-domain maps (shallow CNN) — through a Squeeze-and-Excitation attention module before a final classifier. A 12-experiment ablation study identifies the contribution of each component. The best configuration achieves **96.6% test accuracy, AUC 0.990, and F1 0.980** on the held-out test set.

---

## Results

**Table 1 — Ablation Study (12 Experiments)**

| # | Experiment | Test Acc | AUC | F1 |
|---|---|---|---|---|
| E01 | MLP Baseline | 66.81% | 0.7184 | 0.7760 |
| E02 | Single CNN | 83.24% | 0.6579 | 0.9061 |
| E03 | Stream 1 Only (ResNet-18) | 95.10% | 0.9863 | 0.9710 |
| E04 | Stream 2 Only (Eye/Mouth) | 85.71% | 0.6501 | 0.9231 |
| E05 | Stream 3 Only (FFT) | 71.86% | 0.5525 | 0.8301 |
| E06 | Stream 1 + 2 | 96.14% | 0.9897 | 0.9776 |
| E07 | Stream 1 + 3 | 95.90% | 0.9846 | 0.9760 |
| E08 | Stream 2 + 3 | 78.57% | 0.6232 | 0.8760 |
| E09 | Full Model — Adam, dropout=0.5 | 95.48% | 0.9838 | 0.9736 |
| E10 | Full Model — SGD, dropout=0.5 | 92.76% | 0.9568 | 0.9579 |
| **E11** | **Full Model — Adam, dropout=0.3** ★ | **96.57%** | **0.9904** | **0.9799** |
| E12 | Full Model — No Attention | 96.43% | 0.9921 | 0.9790 |

---

## Architecture

```
Input: [224x224 Face]    [224x224 Eye/Mouth]    [224x224 FFT Map]
            |                     |                      |
            v                     v                      v
    +---------------+    +--------------+    +--------------+
    |   STREAM 1    |    |   STREAM 2   |    |   STREAM 3   |
    |  ResNet-18    |    | 4-block CNN  |    | 3-block CNN  |
    | (pretrained,  |    | BN + ReLU    |    | BN + LReLU   |
    |  partial      |    | MaxPool x3   |    | MaxPool x2   |
    |  fine-tune)   |    | FC 256→256   |    | FC 128→128   |
    | FC 512→512    |    +--------------+    +--------------+
    +---------------+          |                    |
          |                 [256-d]              [128-d]
       [512-d]                 |                    |
          +-------------------+--------------------+
                              |
                      cat([512, 256, 128])
                              |
                          [896-d]
                              |
               +----------------------------+
               |  SE Channel Attention      |
               |  896 → 56 → 896 (Sigmoid) |
               |  element-wise multiply     |
               +----------------------------+
                              |
               +----------------------------+
               |  Classifier Head           |
               |  Linear(896 → 256)         |
               |  BatchNorm1d, ReLU         |
               |  Dropout(0.3)              |
               |  Linear(256 → 2)           |
               +----------------------------+
                              |
                   [Real / Fake logits]

Total params: ~12.4M  |  Optimizer: Adam (lr=1e-4)  |  Scheduler: CosineAnnealingLR
```

---

## Key Findings

1. **Multi-stream fusion outperforms all single-stream baselines.** Stream 1 alone reaches 95.1%; the full 3-stream fusion (E11) reaches 96.6%, confirming the three streams are genuinely complementary.

2. **Stream 1 (ResNet-18) is the dominant contributor.** Pretrained spatial features carry the most signal (95.1% solo), while Stream 2 (85.7%) and Stream 3 (71.9%) are progressively weaker stand-alone — but each adds measurable gain when fused.

3. **Channel attention is modestly beneficial.** Full model with attention (E09, 95.5%) vs. without (E12, 96.4%) shows a mixed picture, but the best overall model (E11, 96.6%) uses attention with a tuned dropout, suggesting its benefit is regularisation-dependent.

4. **Dropout 0.3 beats 0.5 for this model.** E11 (dropout=0.3) achieves the highest test accuracy (96.6%) vs. E09 (dropout=0.5, 95.5%), indicating the model is not underfitting and benefits from lighter regularisation.

5. **Adam significantly outperforms SGD.** Adam (E09, 95.5%) vs. SGD (E10, 92.8%) — a 2.7-point gap consistent with fine-tuning pretrained networks where adaptive learning rates are critical.

6. **ELU outperforms ReLU and LeakyReLU in Stream 2.** On the full model with activation varied only in Stream 2: ELU 95.7% (AUC 0.9888) > ReLU 94.6% (AUC 0.9773) > LeakyReLU 94.3% (AUC 0.9790). ELU's smooth negative half-plane aids fine-grained feature learning in the eye/mouth region.

7. **BatchNorm in Stream 2 has negligible impact at the full model level.** With BN: 94.6% vs. without BN: 95.0% — a <0.5% difference, suggesting Stream 1's extensive BatchNorm layers provide sufficient normalisation for the fused representation.

---

## Repository Structure

```
multicue-df/
├── src/
│   ├── models.py                          # All model architectures (7 models)
│   ├── trainer.py                         # Training loop with early stopping
│   ├── dataset.py                         # MultiCueDataset + DataLoader factory
│   ├── config.py                          # Central hyperparameter config
│   ├── experiment_runner.py               # Runs all 12 ablation experiments
│   ├── preprocess.py                      # 3-stream preprocessing pipeline
│   ├── extract_frames.py                  # Frame extraction from videos
│   ├── activation_experiment_fullmodel.py # Activation function comparison
│   ├── batchnorm_experiment_fullmodel.py  # BatchNorm ablation
│   └── utils.py                           # Shared utilities
├── notebooks/
│   ├── MultiCue_DF.ipynb                  # Main analysis notebook (clean)
│   └── MultiCue_DF_executed.ipynb         # Executed with all outputs
├── results/
│   ├── ablation_summary.csv               # All 12 experiments (metrics)
│   └── *.json                             # Per-experiment result files
├── plots/                                 # All generated PNG figures
├── logs/                                  # Training history JSON files
├── reports/
│   └── MultiCue_DF_Report.pdf
├── presentations/
│   └── MultiCue_DF_Slides.pptx
├── requirements.txt
│
│   — NOT included (too large) —
├── FaceForensics++_C23/   # not included — see Data section below
├── frames/                # not included — see Data section below
├── processed/             # not included — see Data section below
└── checkpoints/           # not included — download from links below
```

---

## Data & Checkpoints

### FaceForensics++ Dataset

The raw video dataset is available from the official source:
**https://www.kaggle.com/datasets/xdxd003/ff-c23**

We use the **C23 (light compression)** variant. The dataset contains 7,000 videos across 6 manipulation methods (Deepfakes, Face2Face, FaceShifter, FaceSwap, NeuralTextures, DeepFakeDetection) plus 1,000 real originals. This project samples 200 videos per class and extracts 10 frames per video (~14,000 frames total).

### Processed Dataset (42,000 images — stream1/stream2/stream3 crops)

Upload coming soon to Google Drive. Once available, a direct download link will be added here.

### Model Checkpoints (.pth files)

Upload coming soon to Google Drive. Once available, direct download links will be added here.

---

## Reproducing Results

```bash
# 1. Clone the repository
git clone <repo-url>
cd multicue-df

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download FaceForensics++ C23
#    Place the dataset at: multicue-df/FaceForensics++_C23/
#    See: https://www.kaggle.com/datasets/xdxd003/ff-c23

# 4. Extract frames from videos
python src/extract_frames.py

# 5. Preprocess frames into 3 streams
python src/preprocess.py

# 6. Run all 12 ablation experiments
python src/experiment_runner.py

# 7. Open the analysis notebook
jupyter notebook notebooks/MultiCue_DF.ipynb
```

All hyperparameters (batch size, learning rate, split ratios, image size, stream dimensions) are centralised in `src/config.py`.

---

## Course

**CS-419 Deep Learning** — SEECS, NUST — May 2026

**Author:** Humna Tariq

---

## References

1. Rössler, A., Cozzolino, D., Verdoliva, L., Riess, C., Thies, J., & Nießner, M. (2019). **FaceForensics++: Learning to Detect Manipulated Facial Images.** *ICCV 2019.* https://github.com/ondyari/FaceForensics

2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). **Deep Residual Learning for Image Recognition.** *CVPR 2016.*

3. Hu, J., Shen, L., & Sun, G. (2018). **Squeeze-and-Excitation Networks.** *CVPR 2018.*

4. Zhang, K., Zhang, Z., Li, Z., & Qiao, Y. (2016). **Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks.** *IEEE Signal Processing Letters, 23*(10), 1499–1503.

5. Qian, Y., Yin, G., Sheng, L., Chen, Z., & Shao, J. (2020). **Thinking in Frequency: Face Forgery Detection by Mining Frequency-aware Clues.** *ECCV 2020.*
