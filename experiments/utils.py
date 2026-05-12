import os
import json
import time
import random
import hashlib

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from config import RANDOM_SEED, MAX_EPOCHS, BATCH_SIZE, ES_PATIENCE, VAL_SPLIT



def compute_mauc(y_true, y_pred):
    aucs = [roc_auc_score(y_true[:, c], y_pred[:, c])
            for c in range(y_true.shape[1])
            if len(np.unique(y_true[:, c])) == 2]
    return np.mean(aucs) if aucs else None


def compute_map(y_true, y_pred):
    aps = [average_precision_score(y_true[:, c], y_pred[:, c])
           for c in range(y_true.shape[1])
           if len(np.unique(y_true[:, c])) == 2]
    return np.mean(aps) if aps else None


def prepare_for_experiment(data):
    data = np.array(data)
    data = np.nan_to_num(data)
    if data.ndim == 3:
        data = data.transpose(0, 2, 1)  # (N, F, T) → (N, T, F)
    return data


def normalize_split(X_train, X_test):
    _, T, F = X_train.shape
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, F)).reshape(X_train.shape)
    X_test  = scaler.transform(X_test.reshape(-1, F)).reshape(X_test.shape)
    return X_train, X_test


def evaluate_model(model, X_test, y_test):
    y_pred     = model.predict(X_test, verbose=0)
    mauc       = compute_mauc(y_test, y_pred)
    map_score  = compute_map(y_test, y_pred)
    print(f"   mAUC: {mauc:.4f}" if mauc else "   mAUC failed")
    print(f"   mAP : {map_score:.4f}" if map_score else "   mAP failed")
    return {"mAUC": mauc, "mAP": map_score}


def run_multibranch_experiment(combos, all_data, win_hop, y_fixed,
                               build_fn=None, epochs=MAX_EPOCHS,
                               results_path=None, normalize=False,
                               norm_features=None, key_suffix=""):
    if results_path and os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results from {results_path}.")
    else:
        results = {}
    total = len(combos)

    for i, (combo_name, feature_list) in enumerate(combos, 1):
        norm_suffix = "_norm" if normalize else ("_selnorm" if norm_features else "")
        result_key  = f"{combo_name}{norm_suffix}{key_suffix}"
        if result_key in results:
            print(f"[{i}/{total}] Multi-branch: {result_key} (cached)")
            continue

        print(f"\n[{i}/{total}] Multi-branch: {result_key}")
        start_time = time.time()

        X_list = [prepare_for_experiment(np.array(all_data[win_hop][f])) for f in feature_list]

        mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        for train_idx, test_idx in mskf.split(X_list[0], y_fixed):
            break

        if normalize:
            pairs = [normalize_split(X[train_idx], X[test_idx]) for X in X_list]
            X_train_list = [p[0] for p in pairs]
            X_test_list  = [p[1] for p in pairs]
        elif norm_features:
            X_train_list, X_test_list = [], []
            for feat, X in zip(feature_list, X_list):
                if feat in norm_features:
                    Xtr, Xte = normalize_split(X[train_idx], X[test_idx])
                else:
                    Xtr, Xte = X[train_idx], X[test_idx]
                X_train_list.append(Xtr)
                X_test_list.append(Xte)
        else:
            X_train_list = [X[train_idx] for X in X_list]
            X_test_list  = [X[test_idx]  for X in X_list]

        y_train, y_test = y_fixed[train_idx], y_fixed[test_idx]
        input_shapes    = [X.shape[1:] for X in X_train_list]

        _run_seed = int(hashlib.md5(result_key.encode()).hexdigest(), 16) % (2**31)
        random.seed(_run_seed); np.random.seed(_run_seed); tf.random.set_seed(_run_seed)

        from models.inception import build_inception_multibranch
        _build = build_fn if build_fn is not None else build_inception_multibranch
        model  = _build(input_shapes, y_fixed.shape[1])

        callbacks  = [EarlyStopping(monitor="val_loss", patience=ES_PATIENCE, restore_best_weights=True)]
        fit_kwargs = dict(epochs=epochs, batch_size=BATCH_SIZE, verbose=0,
                         validation_split=VAL_SPLIT, callbacks=callbacks)
        history      = model.fit(X_train_list, y_train, **fit_kwargs)
        actual_epochs = len(history.history["loss"])
        print(f"  Epochs: {actual_epochs}")

        metrics = evaluate_model(model, X_test_list, y_test)
        metrics["time"]   = time.time() - start_time
        metrics["epochs"] = actual_epochs
        results[result_key] = metrics

        if results_path:
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)

    return results
