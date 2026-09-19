"""
Deep Learning Training Module (PyTorch MLP Stretch Goal)
Trains a multi-layer perceptron on normalized MFCC statistical feature vectors,
evaluates validation performance, and saves model weights.
"""

from typing import Tuple, Dict, Any
import os
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

from src.train import train_classical_models


class SpeechMLP(nn.Module):
    """Multi-Layer Perceptron for Audio Command Classification."""

    def __init__(self, in_dim: int = 26, n_classes: int = 5, dropout_p: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeatureDataset(Dataset):
    """PyTorch Dataset wrapper for numpy feature matrices and integer labels."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def train_pytorch_mlp(
    features_csv_path: str = "data/processed/features.csv",
    models_dir: str = "models",
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 0.003,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train the PyTorch MLP classifier on the extracted feature dataset."""
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    os.makedirs(models_dir, exist_ok=True)

    if not os.path.exists(features_csv_path):
        print(f"Features file {features_csv_path} not found. Running classical pipeline first...")
        train_classical_models()

    features_df = pd.read_csv(features_csv_path)
    feature_cols = [c for c in features_df.columns if c not in ["label", "filepath"]]
    X = features_df[feature_cols].values
    y_raw = features_df["label"].values

    # Load pre-fitted label encoder and scaler if available, or fit
    le_path = os.path.join(models_dir, "label_encoder.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")

    if os.path.exists(le_path) and os.path.exists(scaler_path):
        le = joblib.load(le_path)
        scaler = joblib.load(scaler_path)
        y = le.transform(y_raw)
    else:
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        joblib.dump(le, le_path)
        scaler = StandardScaler()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=random_state
    )

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, scaler_path)

    train_dataset = FeatureDataset(X_train_scaled, y_train)
    test_dataset = FeatureDataset(X_test_scaled, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining PyTorch MLP on device: {device}")

    model = SpeechMLP(in_dim=len(feature_cols), n_classes=len(le.classes_)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_acc = 0.0
    model_save_path = os.path.join(models_dir, "mlp_classifier.pt")

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
            _, predicted = outputs.max(1)
            total_train += batch_y.size(0)
            correct_train += predicted.eq(batch_y).sum().item()

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
                _, predicted = outputs.max(1)
                total_val += batch_y.size(0)
                correct_val += predicted.eq(batch_y).sum().item()

        val_loss /= total_val
        val_acc = correct_val / total_val

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "in_dim": len(feature_cols),
                    "n_classes": len(le.classes_),
                    "classes": list(le.classes_),
                    "best_val_acc": best_val_acc,
                },
                model_save_path,
            )

        if epoch % 10 == 0 or epoch == epochs:
            print(
                f"Epoch [{epoch:02d}/{epochs}] | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}% (Best: {best_val_acc*100:.1f}%)"
            )

    print(f"PyTorch MLP training complete. Best model checkpoint saved to {model_save_path}")
    return {"best_val_acc": best_val_acc}


if __name__ == "__main__":
    train_pytorch_mlp()
