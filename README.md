# DDPM from Scratch

[中文文档](README_zh.md)

A minimal, single-file reimplementation of **Denoising Diffusion Probabilistic Models** (Ho et al., NeurIPS 2020) on CIFAR-10. Inspired by the [nanoGPT](https://github.com/karpathy/nanoGPT) style: one script, readable code, easy to learn and extend.

**Paper:** [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (arXiv:2006.11239)

## Features

- Forward diffusion, L_simple training, and full 1000-step DDPM sampling
- U-Net noise predictor with sinusoidal timestep embedding and spatial self-attention
- EMA weights for sampling
- TensorBoard logging (loss, speed, sample grids, FID)
- FID evaluation against the CIFAR-10 **training set** (same protocol as the paper)

## Project Layout

```
my_DDPM/
├── train.py              # training, sampling, FID eval (all-in-one)
├── requirements.txt
├── docs/
│   ├── DDPM-note.html    # paper reading notes (Chinese)
│   └── 2006.11239v2.pdf  # original paper
├── data/                 # CIFAR-10 (auto-downloaded)
└── runs/                 # checkpoints, samples, tensorboard logs
```

## Setup

Requires Python 3.11+, NVIDIA GPU recommended (CUDA 12.0+ driver).

```bash
pip install -r requirements.txt
```

## Quick Start

### Train

```bash
python train.py --out runs/ddpm_cifar
```

CIFAR-10 is downloaded to `data/` automatically. Checkpoints and sample grids are saved under `runs/ddpm_cifar/`.

### Resume

```bash
python train.py --out runs/ddpm_cifar --resume runs/ddpm_cifar/ckpt.pt
```

### Sample

```bash
python train.py --sample --ckpt runs/ddpm_cifar/ckpt.pt --out runs/ddpm_cifar
```

Outputs `samples_final.png` and `progression.png` (coarse-to-fine denoising).

### TensorBoard

```bash
tensorboard --logdir runs/ddpm_cifar/tb
```

Logs: `train/loss`, `train/ms_per_step`, `samples/grid`, `eval/fid`.

### FID Evaluation

Evaluate an existing checkpoint:

```bash
python train.py --eval_fid --ckpt runs/ddpm_cifar/ckpt.pt --n_fid 10000
```

Run FID periodically during training (every 20k steps by default):

```bash
python train.py --out runs/ddpm_cifar --fid_every 20000
```

For paper-comparable numbers, use 50k samples (slower):

```bash
python train.py --eval_fid --ckpt runs/ddpm_cifar/ckpt.pt --n_fid 50000
```

> FID is expensive: each image requires a full 1000-step sampling chain. Use `--n_fid 5000` for quick checks during development.

## Common Options

| Flag | Default | Description |
|------|---------|-------------|
| `--max_steps` | 200000 | Training steps |
| `--batch_size` | 128 | Batch size |
| `--dim` | 64 | U-Net base channels (~3.6M params) |
| `--T` | 1000 | Diffusion timesteps |
| `--sample_every` | 5000 | Save sample grid every N steps |
| `--fid_every` | 20000 | FID every N steps (`0` = disable) |
| `--n_fid` | 10000 | Images for FID (paper: 50000) |

## Dataset

| Item | This repo | Paper |
|------|-----------|-------|
| Dataset | [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) | CIFAR-10 |
| Task | Unconditional generation | Unconditional generation |
| Training data | 50,000 train images | 50,000 train images |
| Resolution | 32×32 RGB | 32×32 RGB |
| Preprocessing | Scale to **[-1, 1]** (`Normalize(0.5)`) | Scale to [-1, 1] |
| Augmentation | Random horizontal flip | Random horizontal flip |
| FID reference set | CIFAR-10 **train** split (50k) | CIFAR-10 **train** split (50k) |

## Results

FID evaluated with **50,000 generated samples** vs the CIFAR-10 training set (Inception-v3, 2048-dim; same protocol as the paper). EMA weights used for sampling.

| Metric | This repo | Paper (L_simple) |
|--------|-----------|------------------|
| **FID ↓** | **17.54** | **3.17** |
| Inception Score | not evaluated | 9.46 |
| NLL (bits/dim) | not evaluated | ≤ 3.75 |

The gap is expected: this run uses a smaller U-Net and fewer training steps (see below). The pipeline (train → sample → FID) is verified end-to-end.

## vs. Original Paper

Core algorithm matches the paper (ε-prediction, L_simple, linear β schedule, EMA sampling). **Intentionally simplified** for learning and limited compute:

| Item | This repo | Paper (CIFAR-10) |
|------|-----------|------------------|
| U-Net params | ~3.6M (`dim=64`) | ~35.7M (`dim=128`) |
| ResBlocks per level | 1 | 2 |
| Self-attention | 16×16 (+ 4×4 bottleneck) | 16×16 |
| Training steps | 200k (default) | 800k |
| Batch size | 128 | 128 |
| Optimizer / LR | Adam, 2×10⁻⁴ | Adam, 2×10⁻⁴ |
| EMA decay | 0.9999 | 0.9999 |
| Dropout | 0.1 | 0.1 |
| Diffusion steps T | 1000 | 1000 |
| Sampling σ² | β_t (default) | β_t or β̃_t (similar) |
| L₀ discrete decoder | not implemented | yes |
| Weighted variational bound L | not implemented (L_simple only) | ablated in paper |

See `docs/DDPM-note.html` for a detailed paper walkthrough.

## Reference

```bibtex
@inproceedings{ho2020ddpm,
  title={Denoising Diffusion Probabilistic Models},
  author={Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  booktitle={NeurIPS},
  year={2020}
}
```
