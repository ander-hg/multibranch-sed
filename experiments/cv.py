import json
import os
import time
import random

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from config import RANDOM_SEED, MAX_EPOCHS, BATCH_SIZE, ES_PATIENCE, VAL_SPLIT
from experiments.utils import prepare_for_experiment, normalize_split, evaluate_model

ALL_FEATURES = ['log_mel', 'mfcc', 'chroma', 'zcr', 'rms', 'statistical', 'spectral_centroid']

CV_CONFIGS = {
    'cnn': {
        'results_path': 'results/cnn_multibranch_cv5_results.json',
        'configs': [
            ('log_mel+mfcc+chroma_w128_norm_es', ['log_mel', 'mfcc', 'chroma'], 128, True),
            ('all_w128_norm_es', ALL_FEATURES, 128, True),
        ],
    },
    'inception': {
        'results_path': 'results/inception_multibranch_cv5_results.json',
        'configs': [
            ('all_w256_norm_es', ALL_FEATURES, 256, True),
            ('all_w128_norm_es', ALL_FEATURES, 128, True),
            ('log_mel+mfcc+chroma_w128_norm_es', ['log_mel', 'mfcc', 'chroma'], 128, True),
            ('log_mel+mfcc+chroma+statistical_w128_norm_es',
             ['log_mel', 'mfcc', 'chroma', 'statistical'], 128, True),
        ],
    },
    'transformer': {
        'results_path': 'results/transformer_multibranch_cv5_results.json',
        'configs': [
            ('all_w256_norm_es', ALL_FEATURES, 256, True),
            ('all_w128_norm_es', ALL_FEATURES, 128, True),
        ],
    },
    'mamba': {
        'results_path': 'results/mamba_multibranch_cv5_results.json',
        'configs': [
            ('all_w256_norm_es', ALL_FEATURES, 256, True),
            ('all_w128_norm_es', ALL_FEATURES, 128, True),
        ],
    },
}


def run_cv(arch, all_data, y_fixed, n_folds=5):
    from models.cnn         import build_cnn_multibranch
    from models.inception   import build_inception_multibranch
    from models.transformer import build_transformer_multibranch
    from models.mamba       import build_mamba_multibranch

    build_fns = {
        'cnn':         build_cnn_multibranch,
        'inception':   build_inception_multibranch,
        'transformer': build_transformer_multibranch,
        'mamba':       build_mamba_multibranch,
    }
    build_fn     = build_fns[arch]
    spec         = CV_CONFIGS[arch]
    results_path = spec['results_path']

    cv_results = json.load(open(results_path)) if os.path.exists(results_path) else {}
    mskf       = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    for key_prefix, combo, win, do_norm in spec['configs']:
        print(f'\n{"="*60}')
        print(f'Config: {key_prefix}  (win={win}, norm={do_norm})')
        print(f'{"="*60}')

        X_list = [prepare_for_experiment(np.array(all_data[win][f])) for f in combo]

        for fold_idx, (train_idx, test_idx) in enumerate(mskf.split(X_list[0], y_fixed)):
            key = f'{key_prefix}_fold{fold_idx}'
            if key in cv_results:
                print(f'  fold {fold_idx} (cached)')
                continue

            print(f'  fold {fold_idx}/{n_folds - 1}')
            t0 = time.time()

            if do_norm:
                X_train_list, X_test_list = [], []
                for X in X_list:
                    Xtr, Xte = normalize_split(X[train_idx], X[test_idx])
                    X_train_list.append(Xtr)
                    X_test_list.append(Xte)
            else:
                X_train_list = [X[train_idx] for X in X_list]
                X_test_list  = [X[test_idx]  for X in X_list]

            y_train, y_test = y_fixed[train_idx], y_fixed[test_idx]

            _fold_seed = RANDOM_SEED + fold_idx
            random.seed(_fold_seed); np.random.seed(_fold_seed); tf.random.set_seed(_fold_seed)

            model = build_fn([X.shape[1:] for X in X_train_list], y_fixed.shape[1])
            cb    = EarlyStopping(monitor='val_loss', patience=ES_PATIENCE, restore_best_weights=True)
            history = model.fit(X_train_list, y_train, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
                                validation_split=VAL_SPLIT, callbacks=[cb], verbose=0)
            actual_epochs = len(history.history['loss'])

            metrics           = evaluate_model(model, X_test_list, y_test)
            metrics['epochs'] = actual_epochs
            metrics['time']   = time.time() - t0
            cv_results[key]   = metrics

            with open(results_path, 'w') as f:
                json.dump(cv_results, f, indent=2)
            print(f'    mAP={metrics["mAP"]:.4f}  mAUC={metrics["mAUC"]:.4f}  '
                  f'epochs={actual_epochs}  {metrics["time"]:.0f}s')


if __name__ == '__main__':
    import argparse
    from features.dataset import load_all_data
    from config import WINDOW_SIZES, FEATURE_NAMES

    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', choices=list(CV_CONFIGS.keys()), required=True)
    parser.add_argument('--features_dir', default='features/data')
    parser.add_argument('--audio_dir',    default='audio')
    args = parser.parse_args()

    all_data, y_fixed = load_all_data(
        args.features_dir, None, args.audio_dir, WINDOW_SIZES, FEATURE_NAMES
    )
    run_cv(args.arch, all_data, y_fixed)
