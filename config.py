import os
import random

os.environ['PYTHONHASHSEED']        = '0'
os.environ['TF_DETERMINISTIC_OPS']  = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np

RANDOM_SEED  = 42
MAX_EPOCHS   = 200
WINDOW_SIZES = [128, 256, 512, 1024]  # ms per frame; at 44 100 Hz: 5644/11289/22579/45158 samples/frame → ~78/39/19/9 frames per 10s clip (no overlap)
BATCH_SIZE   = 32
ES_PATIENCE  = 5
VAL_SPLIT    = 0.1

N_FFT  = 2048
N_MELS = 40
N_MFCC = 13
F_MAX  = 8000

FEATURE_NAMES = ['log_mel', 'mfcc', 'chroma', 'zcr', 'rms', 'statistical', 'spectral_centroid']


def set_seeds():
    import tensorflow as tf
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
