import os
import random

os.environ['PYTHONHASHSEED']        = '0'
os.environ['TF_DETERMINISTIC_OPS']  = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np

RANDOM_SEED  = 42
MAX_EPOCHS   = 200
WINDOW_SIZES = [128, 256, 512, 1024]  # samples (≈5.8, 11.6, 23.2, 46.4 ms at 22 050 Hz)
BATCH_SIZE   = 32
ES_PATIENCE  = 5
VAL_SPLIT    = 0.1

N_FFT  = 2048
N_MELS = 40
N_MFCC = 13
F_MAX  = 8000

FEATURE_NAMES = ['log_mel', 'mfcc', 'chroma', 'zcr', 'rms', 'statistical', 'spectral_centroid']

# log_mel excluded: already db-normalized per clip
SELNORM_FEATURES = {'mfcc', 'chroma', 'zcr', 'rms', 'statistical', 'spectral_centroid'}


def set_seeds():
    import tensorflow as tf
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
