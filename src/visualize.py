"""
Visualization Module for Audio ML Pipeline
Generates publication-quality waveforms, STFT spectrograms, MFCC heatmaps,
confusion matrices, and model comparison charts.
"""

from typing import List, Dict, Optional
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display

# Style settings for clean, professional figures
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 14,
    }
)


def plot_waveform(
    y: np.ndarray,
    sr: int = 16000,
    label: str = "",
    output_path: Optional[str] = "outputs/sample_waveform.png",
) -> plt.Figure:
    """Plot the audio waveform in the time domain."""
    fig, ax = plt.subplots(figsize=(9, 3.5), tight_layout=True)
    time_axis = np.linspace(0, len(y) / sr, num=len(y))

    ax.plot(time_axis, y, color="#1f77b4", linewidth=1.0, alpha=0.9)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.set_title(
        f"Audio Waveform{' - Class: ' + label if label else ''} (16 kHz Mono)"
    )
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim([0, len(y) / sr])
    ax.grid(True, linestyle=":", alpha=0.5)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300)
    return fig


def plot_spectrogram(
    S_db: np.ndarray,
    sr: int = 16000,
    hop_length: int = 256,
    label: str = "",
    output_path: Optional[str] = "outputs/sample_spectrogram.png",
) -> plt.Figure:
    """Plot the STFT log-magnitude spectrogram."""
    fig, ax = plt.subplots(figsize=(9, 4), tight_layout=True)
    img = librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="hz",
        ax=ax,
        cmap="magma",
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB", label="Intensity (dB)")
    ax.set_title(
        f"STFT Spectrogram{' - Class: ' + label if label else ''} (N_FFT=512, Hop=256)"
    )
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Frequency (Hz)")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300)
    return fig


def plot_mfcc(
    mfcc: np.ndarray,
    sr: int = 16000,
    hop_length: int = 256,
    label: str = "",
    output_path: Optional[str] = "outputs/sample_mfcc.png",
) -> plt.Figure:
    """Plot Mel-Frequency Cepstral Coefficients (MFCCs) over time."""
    fig, ax = plt.subplots(figsize=(9, 4), tight_layout=True)
    img = librosa.display.specshow(
        mfcc,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        ax=ax,
        cmap="coolwarm",
    )
    fig.colorbar(img, ax=ax, label="MFCC Magnitude")
    ax.set_title(
        f"Mel-Frequency Cepstral Coefficients (MFCC){' - Class: ' + label if label else ''}"
    )
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("MFCC Index (0-12)")
    ax.set_yticks(np.arange(0, mfcc.shape[0], 2))

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300)
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Optional[str] = "outputs/confusion_matrix.png",
    title: str = "Confusion Matrix (Random Forest Classifier)",
) -> plt.Figure:
    """Plot an annotated confusion matrix with count and normalized percentages."""
    fig, ax = plt.subplots(figsize=(7, 6), tight_layout=True)

    # Compute row percentages
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_perc = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_perc = np.nan_to_num(cm_perc)

    # Combined annotation: Count\n(Percent%)
    labels = [
        [f"{val}\n({p * 100:.1f}%)" for val, p in zip(row, row_p)]
        for row, row_p in zip(cm, cm_perc)
    ]

    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(title, pad=15)
    ax.set_xlabel("Predicted Label", labelpad=10)
    ax.set_ylabel("True Label", labelpad=10)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300)
    return fig


def plot_model_comparison(
    metrics_summary: Dict[str, Dict[str, float]],
    output_path: Optional[str] = "outputs/model_comparison.png",
) -> plt.Figure:
    """Plot bar comparison of metrics across models."""
    fig, ax = plt.subplots(figsize=(8, 4.5), tight_layout=True)

    models = list(metrics_summary.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    n_models = len(models)
    n_metrics = len(metrics)

    bar_width = 0.8 / n_models
    x = np.arange(n_metrics)

    colors = ["#2b5c8f", "#d95f02", "#7570b3"]

    for i, model in enumerate(models):
        values = [metrics_summary[model].get(m, 0.0) * 100 for m in metrics]
        offset = (i - n_models / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            values,
            width=bar_width,
            label=model,
            color=colors[i % len(colors)],
            edgecolor="black",
            linewidth=0.5,
        )
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_ylabel("Score (%)")
    ax.set_title("Model Performance Benchmark Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300)
    return fig


def generate_dsp_visualizations(
    sample_wav_path: Optional[str] = None,
    output_dir: str = "outputs",
    sr: int = 16000,
) -> None:
    """Generate and save sample waveform, spectrogram, and MFCC plots for DSP verification."""
    from src.preprocess import load_and_fix_length
    from src.features import extract_stft, extract_mfcc

    os.makedirs(output_dir, exist_ok=True)

    if sample_wav_path is None or not os.path.exists(sample_wav_path):
        # Find first available sample in data/raw
        for root, _, files in os.walk("data/raw"):
            for f in files:
                if f.endswith(".wav"):
                    sample_wav_path = os.path.join(root, f)
                    break
            if sample_wav_path and os.path.exists(sample_wav_path):
                break

    if sample_wav_path is None or not os.path.exists(sample_wav_path):
        raise FileNotFoundError("No sample audio file found to generate DSP visualizations.")

    label = os.path.basename(os.path.dirname(sample_wav_path))
    print(f"Generating DSP visualizations using sample audio: {sample_wav_path} (Class: {label})")

    y = load_and_fix_length(sample_wav_path, sr=sr, duration=1.0)
    _, S_db = extract_stft(y, n_fft=512, hop_length=256)
    mfcc = extract_mfcc(y, sr=sr, n_mfcc=13, hop_length=256)

    wav_path = os.path.join(output_dir, "sample_waveform.png")
    spec_path = os.path.join(output_dir, "sample_spectrogram.png")
    mfcc_path = os.path.join(output_dir, "sample_mfcc.png")

    plot_waveform(y, sr=sr, label=label, output_path=wav_path)
    plot_spectrogram(S_db, sr=sr, hop_length=256, label=label, output_path=spec_path)
    plot_mfcc(mfcc, sr=sr, hop_length=256, label=label, output_path=mfcc_path)

    print(f"Saved: {wav_path}")
    print(f"Saved: {spec_path}")
    print(f"Saved: {mfcc_path}")


if __name__ == "__main__":
    generate_dsp_visualizations()
