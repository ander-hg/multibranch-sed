# Multi-Branch Feature Fusion for Weakly Labeled Audio Classification

Code for the experiments reported in:

> Anderson H. Giacomini, Elaine Parros Machado de Sousa.
> **Multi-Branch Feature Fusion for Weakly Labeled Audio Classification:
> An Empirical Comparison of CNN, Transformer, and Mamba Architectures.**
> SBBD 2026.

## Overview

We compare four neural architectures — CNN, Inception-style CNN, Transformer,
and Mamba — for multi-label audio classification on AudioSet balanced
(16,306 clips, 527 classes), training entirely from scratch.
A multi-branch design (one encoder per acoustic feature, merged before
the classification head) consistently improves all architectures by 17–28% mAP.

| Architecture | Single-feat. mAP | Multi-feat. mAP       |
|---|---|---|
| CNN          | 0.097†           | 0.123 ± 0.006          |
| Inception    | 0.115†           | **0.135 ± 0.004**      |
| Transformer  | 0.085†           | 0.109 ± 0.005          |
| Mamba        | 0.086†           | 0.110†                 |

†single-run; others are 5-fold CV mean ± std.

## Repository layout

```
config.py              # global constants and seeds
features/
  extractors.py        # per-feature extraction functions (log-mel, MFCC, chroma, …)
  dataset.py           # dataset preparation and label encoding
models/
  cnn.py               # CNN and multi-branch CNN
  inception.py         # Inception-style CNN and multi-branch variant
  transformer.py       # Transformer encoder + LinearWarmup schedule
  mamba.py             # MambaBlock and multi-branch Mamba
experiments/
  utils.py             # metrics, data prep, single-run and multi-branch experiment runners
  cv.py                # 5-fold cross-validation runner
figures/
  plot.py              # reproduces all paper figures (SBC format)
  fig*.pdf / fig*.png  # pre-generated paper figures
results/
  *.json               # cached results (one file per architecture)
```

## Setup

```bash
pip install -r requirements.txt
```

## Data

Download AudioSet balanced from the official source. Set `audio_base_path` and
`balanced_path` in your loading script to point to the audio files and the
`balanced_train_segments.csv` annotation file. Pre-extracted features (`.pkl`)
are not versioned due to size; run `features/dataset.py` to regenerate them.

## Reproducibility

All randomness is controlled via `RANDOM_SEED = 42` in `config.py`.
Per-run seeds are derived from the result key via `hashlib.md5`, ensuring
full reproducibility regardless of which experiments are cached and skipped.
CV fold seeds use `RANDOM_SEED + fold_idx`.

## Reproducing an experiment

```python
import config                              # sets TF env vars before TF import
import tensorflow as tf
from config import set_seeds, WINDOW_SIZES, FEATURE_NAMES
set_seeds()

from features.dataset import load_all_data
from models.inception import build_inception_multibranch
from experiments.utils import run_multibranch_experiment

all_data, y_fixed = load_all_data(
    features_dir="features/",
    segments_df=balanced_segments,         # pandas DataFrame from balanced_train_segments.csv
    audio_base_path="/path/to/audioset/",
    window_sizes=WINDOW_SIZES,
    feature_names=FEATURE_NAMES,
)

run_multibranch_experiment(
    combos=[("all_w128_norm_es", ["log_mel", "mfcc", "chroma",
                                   "zcr", "rms", "statistical", "spectral_centroid"])],
    all_data=all_data, win_hop=128, y_fixed=y_fixed,
    build_fn=build_inception_multibranch,
    results_path="results/inception_multibranch_es_results.json",
    normalize=True,
)
```

## Regenerating figures

```bash
python figures/plot.py
```
