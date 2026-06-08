# Autoencoder-Enhanced Balanced GAN Framework for Imbalanced Sonar Image Recognition

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

[![Stars](https://img.shields.io/github/stars/cwcaiken/imbalanced-sonar-recognition?style=social)](https://github.com/cwcaiken/imbalanced-sonar-recognition/stargazers)
[![Forks](https://img.shields.io/github/forks/cwcaiken/imbalanced-sonar-recognition?style=social)](https://github.com/cwcaiken/imbalanced-sonar-recognition/network/members)
[![Issues](https://img.shields.io/github/issues/cwcaiken/imbalanced-sonar-recognition.svg)](https://github.com/cwcaiken/imbalanced-sonar-recognition/issues)
[![Last commit](https://img.shields.io/github/last-commit/cwcaiken/imbalanced-sonar-recognition.svg)](https://github.com/cwcaiken/imbalanced-sonar-recognition/commits/main)
![Visitors](https://visitor-badge.laobi.icu/badge?page_id=cwcaiken.imbalanced-sonar-recognition)

An autoencoder-enhanced **Balanced GAN with Gradient Penalty (BAGAN-GP)** framework for
recognising sonar images under severe class imbalance. A pre-trained autoencoder initialises
the GAN generator/discriminator, WGAN-GP stabilises adversarial training, and CBAM attention
improves both generation and classification. The synthetic samples produced for
under-represented classes are then used to train a balanced attention-based classifier.

## Method overview

The pipeline has three stages:

1. **Autoencoder pre-training** — an attention-based encoder/decoder (ChannelAttention,
   SpatialAttention, CBAM) compresses 64×64 sonar images into a latent space and reconstructs
   them, providing strong initial weights for the GAN.
2. **BAGAN-GP training** — a conditional GAN built on the pre-trained decoder. The generator
   synthesises class-conditional images from noise + label embeddings; the discriminator
   enforces real/fake and class conditioning under a WGAN gradient penalty. This rebalances
   minority classes by generating high-quality samples.
3. **Balanced classification** — an attention-based residual classifier is trained on the
   real data augmented with GAN-generated minority-class samples, then evaluated with standard
   metrics and FID.

## Repository structure

```text
imbalanced-sonar-recognition/
├── sonar/
│   ├── fetch_data.py     # Load raw images, letterbox-resize to 64×64×3, save train/val .npy
│   ├── main2.py          # Autoencoder pre-training + BAGAN-GP training loop
│   ├── tran_cls1.py      # Train the balanced attention classifier (real + synthetic samples)
│   ├── pred.py           # Build/evaluate a model; report accuracy/precision/recall/F1/AUC
│   ├── predictor.py      # Model build + evaluation on the NKSID test set
│   ├── fid_score.py      # Per-class Fréchet Inception Distance (generation quality)
│   └── tsne_plot.py      # t-SNE visualisation of learned feature embeddings
├── requirements.txt
├── LICENSE
└── README.md
```

## Datasets

The framework is evaluated on two public imbalanced forward-looking sonar datasets:

- **NKSID** — 8 object categories.
- **FLSMDD** — forward-looking sonar marine debris dataset.

Place each dataset under a folder at the repository root and point the scripts at it via the
`path` variable (relative paths `./NKSID/` and `./FLSMDD/` are used by default). The expected
layout after running `fetch_data.py` is:

```text
NKSID/                  # or FLSMDD/
├── x_train.npy         # training images, 64×64×3, normalised to [-1, 1]
├── y_train.npy         # integer class labels
├── x_val.npy
└── y_val.npy
```

> **Note:** Hard-coded Windows paths have been replaced with relative `./NKSID/` and
> `./FLSMDD/` paths. Adjust the `path` variable at the top of each script if your data lives
> elsewhere. Switch datasets by toggling the commented `path` lines.

## Installation

```bash
git clone https://github.com/cwcaiken/imbalanced-sonar-recognition.git
cd imbalanced-sonar-recognition

python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

GPU-enabled TensorFlow is recommended for training.

## Usage

Run the scripts from the repository root (so the relative data paths resolve):

```bash
# 1. Prepare the dataset (.npy files)
python sonar/fetch_data.py

# 2. Pre-train the autoencoder and train BAGAN-GP
python sonar/main2.py

# 3. Train the balanced attention classifier with synthetic augmentation
python sonar/tran_cls1.py

# 4. Evaluate the classifier (accuracy / precision / recall / F1 / AUC)
python sonar/pred.py        # or: python sonar/predictor.py

# 5. Assess generation quality and feature separability
python sonar/fid_score.py   # per-class FID
python sonar/tsne_plot.py   # t-SNE embedding plot
```

Key hyperparameters (set in `main2.py`): latent dim 128, batch size 128, discriminator/generator
update ratio 3:1, generator LR 2e-4, discriminator LR 1e-4.

## Requirements

See [`requirements.txt`](requirements.txt). Core dependencies: TensorFlow 2.x / Keras, NumPy,
OpenCV, scikit-learn, matplotlib, scipy.

## License

This project is released under the [MIT License](LICENSE).

## Contact

For questions about the code, contact **cwcaiken@163.com**.
