"""
DSP Feature Extraction Module
Computes Short-Time Fourier Transform (STFT), Mel-Frequency Cepstral Coefficients (MFCC),
Delta & Delta-Delta dynamic features, Spectral descriptors, Temporal Segment Pooling,
and 2D Log-Mel Spectrograms for deep convolutional architectures.
"""

from typing import Tuple, List, Dict, Any
import numpy as np
import librosa


def extract_stft(
    y: np.ndarray,
    n_fft: int = 512,
    hop_length: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Short-Time Fourier Transform (STFT) and magnitude spectrogram in dB."""
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    return D, S_db


def extract_mfcc(
    y: np.ndarray,
    sr: int = 16000,
    n_mfcc: int = 13,
    n_fft: int = 512,
    hop_length: int = 256,
) -> np.ndarray:
    """Compute Mel-Frequency Cepstral Coefficients (MFCCs)."""
    return librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )


def extract_log_mel_spectrogram(
    y: np.ndarray,
    sr: int = 16000,
    n_mels: int = 40,
    n_fft: int = 512,
    hop_length: int = 256,
) -> np.ndarray:
    """Compute Log-Mel Spectrogram as a 2D matrix (frequency_bins x time_frames).

    Standard representation for 2D Audio CNNs.
    """
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel.astype(np.float32)


def segment_statistics(matrix_2d: np.ndarray, n_segments: int = 3) -> np.ndarray:
    """Divide a 2D feature matrix (features x time) into temporal segments

    (e.g. onset, nucleus, coda) and compute mean and std within each segment.
    Preserves phonetic time sequence to eliminate consonant-vowel confusion.
    """
    n_feats, n_time = matrix_2d.shape
    step = max(1, n_time // n_segments)
    stats = []

    for seg in range(n_segments):
        start = seg * step
        end = n_time if seg == n_segments - 1 else (seg + 1) * step
        chunk = matrix_2d[:, start:end]
        if chunk.shape[1] > 0:
            stats.append(np.mean(chunk, axis=1))
            stats.append(np.std(chunk, axis=1))
        else:
            stats.append(np.zeros(n_feats))
            stats.append(np.zeros(n_feats))

    return np.concatenate(stats)


def extract_advanced_dsp_vector(
    y: np.ndarray,
    sr: int = 16000,
    n_mfcc: int = 20,
    n_segments: int = 3,
) -> Tuple[np.ndarray, List[str]]:
    """Extract temporal-segmented MFCCs, Deltas, Delta-Deltas, and Spectral Descriptors.

    Solves the 'time-smearing' issue by tracking how acoustic properties evolve
    from the beginning to the end of the spoken word.

    Returns:
        Tuple of (feature_vector_1d, feature_names_list).
    """
    # 1. MFCC + Delta + Delta2
    mfcc = extract_mfcc(y, sr=sr, n_mfcc=n_mfcc, n_fft=512, hop_length=256)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # 2. Spectral descriptors
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=512, hop_length=256)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=512, hop_length=256)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=256)

    # Stack descriptors
    spectral_matrix = np.vstack([cent, rolloff, zcr])  # shape (3, time)

    # 3. Temporal Segment Pooling (Onset, Nucleus, Coda)
    mfcc_seg_stats = segment_statistics(mfcc, n_segments=n_segments)
    delta_seg_stats = segment_statistics(delta, n_segments=n_segments)
    delta2_seg_stats = segment_statistics(delta2, n_segments=n_segments)
    spectral_seg_stats = segment_statistics(spectral_matrix, n_segments=n_segments)

    # 4. Global statistics
    global_mfcc = np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])
    global_delta = np.concatenate([np.mean(delta, axis=1), np.std(delta, axis=1)])
    global_spectral = np.concatenate([np.mean(spectral_matrix, axis=1), np.std(spectral_matrix, axis=1)])

    feat_vector = np.concatenate([
        mfcc_seg_stats,
        delta_seg_stats,
        delta2_seg_stats,
        spectral_seg_stats,
        global_mfcc,
        global_delta,
        global_spectral,
    ])

    # Construct feature names
    names = []
    for group, count in [("mfcc", n_mfcc), ("delta", n_mfcc), ("delta2", n_mfcc)]:
        for seg in range(n_segments):
            names.extend([f"{group}_seg{seg}_c{i}_mean" for i in range(count)])
            names.extend([f"{group}_seg{seg}_c{i}_std" for i in range(count)])
    for seg in range(n_segments):
        for s_name in ["centroid", "rolloff", "zcr"]:
            names.extend([f"{s_name}_seg{seg}_mean", f"{s_name}_seg{seg}_std"])

    for group, count in [("mfcc_glob", n_mfcc), ("delta_glob", n_mfcc)]:
        names.extend([f"{group}_c{i}_mean" for i in range(count)])
        names.extend([f"{group}_c{i}_std" for i in range(count)])
    for s_name in ["centroid_glob", "rolloff_glob", "zcr_glob"]:
        names.extend([f"{s_name}_mean", f"{s_name}_std"])

    return feat_vector.astype(np.float32), names


# Backward compatibility alias
def extract_full_feature_vector(
    y: np.ndarray,
    sr: int = 16000,
    n_mfcc: int = 13,
) -> Tuple[np.ndarray, List[str]]:
    """Legacy extractor: backward compatible with baseline."""
    return extract_advanced_dsp_vector(y, sr=sr, n_mfcc=20, n_segments=3)
