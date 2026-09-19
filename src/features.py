"""
DSP Feature Extraction Module
Computes Short-Time Fourier Transform (STFT), Mel-Frequency Cepstral Coefficients (MFCC),
and time-aggregated summary statistics for classical ML classification.
"""

from typing import Tuple, Dict, Any
import numpy as np
import librosa


def extract_stft(
    y: np.ndarray,
    n_fft: int = 512,
    hop_length: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Short-Time Fourier Transform (STFT) and magnitude spectrogram in dB.

    Args:
        y: 1D audio time series.
        n_fft: Length of the FFT window.
        hop_length: Number of audio samples between adjacent STFT columns.

    Returns:
        Tuple of:
            - D: Complex STFT matrix (1 + n_fft // 2, time_frames)
            - S_db: Magnitude spectrogram in decibels (dB)
    """
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
    """Compute Mel-Frequency Cepstral Coefficients (MFCCs).

    Args:
        y: 1D audio time series.
        sr: Sample rate in Hz.
        n_mfcc: Number of MFCCs to extract.
        n_fft: FFT window length.
        hop_length: Hop length between frames.

    Returns:
        MFCC matrix of shape (n_mfcc, time_frames).
    """
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    return mfcc


def mfcc_to_feature_vector(mfcc: np.ndarray) -> np.ndarray:
    """Aggregate MFCC matrix across time into a fixed-length 1D feature vector

    using mean and standard deviation per coefficient.

    Args:
        mfcc: MFCC matrix of shape (n_mfcc, time_frames).

    Returns:
        1D numpy array of length 2 * n_mfcc (e.g. 26 for n_mfcc=13).
        [mean_0, mean_1, ..., mean_k, std_0, std_1, ..., std_k]
    """
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    return np.concatenate([mfcc_mean, mfcc_std])


def extract_full_feature_vector(
    y: np.ndarray,
    sr: int = 16000,
    n_mfcc: int = 13,
) -> Tuple[np.ndarray, list]:
    """Extract standard aggregated MFCC feature vector and return corresponding feature names.

    Args:
        y: Preprocessed 1D audio time series.
        sr: Sample rate.
        n_mfcc: Number of MFCC coefficients.

    Returns:
        Tuple of (feature_vector_1d, feature_names_list).
    """
    mfcc = extract_mfcc(y, sr=sr, n_mfcc=n_mfcc)
    feat_vec = mfcc_to_feature_vector(mfcc)

    feature_names = [f"mfcc_{i}_mean" for i in range(n_mfcc)] + [
        f"mfcc_{i}_std" for i in range(n_mfcc)
    ]
    return feat_vec, feature_names
