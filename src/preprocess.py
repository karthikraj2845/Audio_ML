"""
Audio Preprocessing Module
Handles audio loading, resampling, silence trimming, and fixed-length padding/truncation.
"""

from typing import Tuple
import numpy as np
import librosa

DEFAULT_SR = 16000
DEFAULT_DURATION = 1.0  # seconds
DEFAULT_N_SAMPLES = int(DEFAULT_SR * DEFAULT_DURATION)


def load_and_fix_length(
    filepath: str,
    sr: int = DEFAULT_SR,
    duration: float = DEFAULT_DURATION,
    top_db: int = 20,
) -> np.ndarray:
    """Load an audio file, resample to target rate, trim leading/trailing silence,

    and ensure fixed length through truncation or zero-padding.

    Args:
        filepath: Path to the .wav audio file.
        sr: Target sample rate (Hz). Defaults to 16,000 Hz.
        duration: Desired duration in seconds. Defaults to 1.0s.
        top_db: Threshold (in dB) below peak to consider as silence for trimming.

    Returns:
        1D numpy array of audio samples of shape (n_samples,).
    """
    target_samples = int(sr * duration)

    # 1. Load with librosa (ensures mono and fixed sample rate)
    y, _ = librosa.load(filepath, sr=sr, mono=True)

    # 2. Trim silence
    y, _ = librosa.effects.trim(y, top_db=top_db)

    # Handle edge case where signal was entirely silent or empty
    if len(y) == 0:
        return np.zeros(target_samples, dtype=np.float32)

    # 3. Fix length: pad or truncate to target_samples
    if len(y) > target_samples:
        y = y[:target_samples]
    else:
        padding = target_samples - len(y)
        y = np.pad(y, (0, padding), mode="constant")

    return y.astype(np.float32)


def normalize_audio(y: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Peak-normalize an audio signal to range [-1.0, 1.0].

    Args:
        y: 1D audio array.
        eps: Small epsilon to prevent division by zero.

    Returns:
        Normalized audio array.
    """
    max_val = np.max(np.abs(y))
    if max_val > eps:
        return y / max_val
    return y
