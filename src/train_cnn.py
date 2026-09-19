"""
2D Audio Convolutional Neural Network (AudioCNN) Training Module
Directly consumes 2D Log-Mel Spectrograms (40 frequency bins x 63 time frames)
to model local time-frequency acoustic patterns without global time-averaging.
"""

from typing import Tuple, Dict, Any, List
import os
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from src.data_loader import build_file_index, download_speech_commands_subset, DEFAULT_CLASSES
from src.preprocess import load_and_fix_length
from src.features import extract_log_mel_spectrogram


class AudioCNN(nn.Module):
    """Deep 2D Convolutional Neural Network for Audio Spectrogram Classification."""

    def __init__(self, n_classes: int = 5):
        super().__init__()
        # Conv Block 1: (1, 40, 63) -> (32, 20, 31)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.15),
        )

        # Conv Block 2: (32, 20, 31) -> (64, 10, 15)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.20),
        )

        # Conv Block 3: (64, 10, 15) -> (128, 5, 7)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
        )

        # Global Pooling + Dense Classification Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.head(x)


class SpectrogramDataset(Dataset):
    """PyTorch Dataset that loads audio, extracts Log-Mel spectrograms, and applies augmentation."""

    def __init__(
        self,
        filepaths: List[str],
        labels: np.ndarray,
        augment: bool = False,
        sr: int = 16000,
        n_mels: int = 40,
    ):
        self.filepaths = filepaths
        self.labels = labels
        self.augment = augment
        self.sr = sr
        self.n_mels = n_mels

        # Precompute spectrograms in memory for fast epoch throughput
        self.specs = []
        for fp in filepaths:
            y = load_and_fix_length(fp, sr=sr, duration=1.0)
            spec = extract_log_mel_spectrogram(y, sr=sr, n_mels=n_mels)
            # Standardize spectrogram per-sample
            spec = (spec - spec.mean()) / (spec.std() + 1e-6)
            self.specs.append(spec)
        self.specs = np.array(self.specs)  # (N, n_mels, time)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        spec = self.specs[idx].copy()

        if self.augment:
            # Random time shift (horizontal roll)
            shift = np.random.randint(-5, 6)
            spec = np.roll(spec, shift, axis=1)
            # Subtle Gaussian noise injection
            spec = spec + np.random.normal(0, 0.05, spec.shape).astype(np.float32)

        tensor_x = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)  # (1, n_mels, time)
        tensor_y = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor_x, tensor_y


def train_audio_cnn(
    data_dir: str = "data/raw",
    models_dir: str = "models",
    epochs: int = 45,
    batch_size: int = 32,
    lr: float = 0.003,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train 2D AudioCNN on Log-Mel Spectrograms."""
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    os.makedirs(models_dir, exist_ok=True)

    df_index = build_file_index(
        data_dir=data_dir,
        selected_classes=DEFAULT_CLASSES,
        samples_per_class=100,
        random_state=random_state,
    )

    le_path = os.path.join(models_dir, "label_encoder.pkl")
    if os.path.exists(le_path):
        le = joblib.load(le_path)
        y = le.transform(df_index["label"].values)
    else:
        le = LabelEncoder()
        y = le.fit_transform(df_index["label"].values)
        joblib.dump(le, le_path)

    filepaths = df_index["filepath"].values

    # Stratified 80/20 train/test split
    train_fps, test_fps, y_train, y_test = train_test_split(
        filepaths, y, test_size=0.20, stratify=y, random_state=random_state
    )

    print(f"Loading & precomputing Log-Mel Spectrograms for {len(df_index)} clips...")
    train_dataset = SpectrogramDataset(train_fps, y_train, augment=True)
    test_dataset = SpectrogramDataset(test_fps, y_test, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training 2D AudioCNN on device: {device}")

    model = AudioCNN(n_classes=len(le.classes_)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc = 0.0
    model_save_path = os.path.join(models_dir, "cnn_classifier.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            _, pred = outputs.max(1)
            total_train += batch_y.size(0)
            correct_train += pred.eq(batch_y).sum().item()

        scheduler.step()
        train_loss /= total_train
        train_acc = correct_train / total_train

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                _, pred = outputs.max(1)
                total_val += batch_y.size(0)
                correct_val += pred.eq(batch_y).sum().item()

        val_loss /= total_val
        val_acc = correct_val / total_val

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "n_classes": len(le.classes_),
                    "classes": list(le.classes_),
                    "best_val_acc": best_val_acc,
                },
                model_save_path,
            )

        if epoch % 5 == 0 or epoch == epochs:
            print(
                f"Epoch [{epoch:02d}/{epochs}] | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}% (Best: {best_val_acc*100:.1f}%)"
            )

    print(f"\nAudioCNN Training Complete! Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    print(f"Model saved to: {model_save_path}")

    # Save test specs array for evaluate module
    np.savez_compressed(
        "data/processed/test_cnn_split.npz",
        specs=test_dataset.specs,
        labels=y_test,
    )
    return {"best_val_acc": best_val_acc}


if __name__ == "__main__":
    train_audio_cnn()
