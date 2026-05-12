import numpy as np
import librosa

from config import N_FFT, N_MELS, N_MFCC, F_MAX


def extract_log_mel_spectrogram(y, sr, win_length, hop_length, n_mels=N_MELS):
    n_fft = max(N_FFT, win_length)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, fmax=F_MAX,
        win_length=win_length, hop_length=hop_length, n_fft=n_fft
    )
    return librosa.power_to_db(mel, ref=np.max)


def extract_mfcc(y, sr, win_length, hop_length):
    n_fft = max(N_FFT, win_length)
    return librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, hop_length=hop_length, n_fft=n_fft, win_length=win_length
    )


def extract_chroma(y, sr, win_length, hop_length):
    n_fft = max(N_FFT, win_length)
    return librosa.feature.chroma_stft(
        y=y, sr=sr, hop_length=hop_length, n_fft=n_fft, win_length=win_length
    )


def extract_zcr(y, win_length, hop_length):
    return librosa.feature.zero_crossing_rate(y, hop_length=hop_length, frame_length=win_length)


def extract_rms(y, win_length, hop_length):
    return librosa.feature.rms(y=y, hop_length=hop_length, frame_length=win_length)


def extract_spectral_centroid(y, sr, hop_length, win_length):
    n_fft = max(N_FFT, win_length)
    return librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=hop_length, n_fft=n_fft, win_length=win_length
    )


def extract_statistical_windowed(y, win_length, hop_length):
    pad = win_length // 2
    y_padded = np.pad(y, pad_width=pad, mode="reflect")
    frames = librosa.util.frame(y_padded, frame_length=win_length, hop_length=hop_length)
    stats = np.stack([
        np.mean(frames, axis=0), np.std(frames, axis=0),
        np.max(frames, axis=0),  np.min(frames, axis=0),
        np.percentile(frames, 25, axis=0), np.percentile(frames, 75, axis=0),
    ], axis=0)
    return stats.astype(np.float32)


def extract_features(file_path, feature_name, win_length_ms=40, hop_length_ms=40):
    try:
        y, sr = librosa.load(file_path, sr=None)
        win_length = int(sr * win_length_ms / 1000)
        hop_length = int(sr * hop_length_ms / 1000)

        if feature_name == "log_mel":
            feature = extract_log_mel_spectrogram(y, sr, win_length, hop_length)
        elif feature_name == "mfcc":
            feature = extract_mfcc(y, sr, win_length, hop_length)
        elif feature_name == "chroma":
            feature = extract_chroma(y, sr, win_length, hop_length)
        elif feature_name == "zcr":
            feature = extract_zcr(y, win_length, hop_length)
        elif feature_name == "rms":
            feature = extract_rms(y, win_length, hop_length)
        elif feature_name == "spectral_centroid":
            feature = extract_spectral_centroid(y, sr, win_length, hop_length)
        elif feature_name == "statistical":
            feature = extract_statistical_windowed(y, win_length, hop_length)
        elif feature_name == "log_mel64":
            feature = extract_log_mel_spectrogram(y, sr, win_length, hop_length, n_mels=64)
        else:
            raise ValueError(f"Feature '{feature_name}' not supported")

        return feature, y, sr

    except Exception as e:
        print(f"Error extracting features for {file_path}: {e}")
        return None
