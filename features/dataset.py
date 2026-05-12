import os
import pickle

import numpy as np
from tqdm import tqdm

from features.extractors import extract_features


def prepare_dataset(segments_df, audio_base_path, save_path,
                    feature_name, win_length_ms=40, hop_length_ms=40,
                    load_from_files=True, y_len_accepted=None):
    if os.path.exists(save_path) and load_from_files:
        print(f"Loading existing features from {save_path}...")
        with open(save_path, "rb") as f:
            return pickle.load(f)

    features, labels = [], []

    for _, row in tqdm(segments_df.iterrows(), total=len(segments_df), desc="Processing segments"):
        audio_file = os.path.join(audio_base_path, f"{row['YTID']}.wav")
        if not os.path.exists(audio_file):
            continue

        result = extract_features(audio_file, feature_name,
                                  win_length_ms=win_length_ms, hop_length_ms=hop_length_ms)
        if result is None:
            continue

        audio_features, y, sr = result

        if y_len_accepted is not None and len(y) != y_len_accepted:
            continue

        features.append(audio_features)
        labels.append(row["mapped_labels"])

    print(f"Saving extracted features to {save_path}...")
    with open(save_path, "wb") as f:
        pickle.dump((features, labels), f)

    return features, labels


def labels_to_one_hot(label_list, label_to_index):
    label_vector = np.zeros(len(label_to_index), dtype=np.float32)
    for label in label_list:
        label_vector[label_to_index[label]] = 1
    return label_vector


def load_all_data(features_dir, segments_df, audio_base_path,
                  window_sizes, feature_names, y_len_accepted=441000):
    all_data   = {w: {} for w in window_sizes}
    all_labels = None

    for win_hop in window_sizes:
        for feature_name in feature_names:
            file_name = os.path.join(
                features_dir,
                f"balanced_features_{feature_name}_wl{win_hop}_hl{win_hop}.pkl"
            )
            if os.path.exists(file_name):
                try:
                    with open(file_name, 'rb') as f:
                        features, labels = pickle.load(f)
                except (pickle.UnpicklingError, EOFError):
                    print(f"Corrupted file: {file_name}. Re-extracting...")
                    os.remove(file_name)
                    features, labels = prepare_dataset(
                        segments_df, audio_base_path, file_name,
                        feature_name=feature_name,
                        win_length_ms=win_hop, hop_length_ms=win_hop,
                        load_from_files=False, y_len_accepted=y_len_accepted,
                    )
            else:
                print(f"Extracting {feature_name} | window={win_hop}")
                features, labels = prepare_dataset(
                    segments_df, audio_base_path, file_name,
                    feature_name=feature_name,
                    win_length_ms=win_hop, hop_length_ms=win_hop,
                    load_from_files=False, y_len_accepted=y_len_accepted,
                )

            all_data[win_hop][feature_name] = features
            if all_labels is None:
                all_labels = labels

    all_unique_labels = {label for label_set in all_labels for label in label_set}
    label_to_index    = {label: i for i, label in enumerate(sorted(all_unique_labels))}
    y_fixed = np.array([labels_to_one_hot(ls, label_to_index) for ls in all_labels])
    print(f"Loaded: {y_fixed.shape[0]} samples, {y_fixed.shape[1]} classes")
    return all_data, y_fixed


def get_feature_blocks(all_data):
    win_hops = sorted(all_data.keys())
    feat_names = sorted(next(iter(all_data.values())).keys())
    blocks = []
    for win_hop in win_hops:
        for feat in feat_names:
            data = np.array(all_data[win_hop][feat])
            blocks.append((f"{feat}_w{win_hop}", data))
    return blocks
