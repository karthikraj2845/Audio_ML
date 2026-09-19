"""
Model Training Module (Classical Machine Learning)
Extracts DSP features, saves features.csv, trains RandomForest and SVM classifiers,
and serializes models and preprocessors.
"""

from typing import Tuple, Dict, Any
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.data_loader import build_file_index, download_speech_commands_subset, DEFAULT_CLASSES
from src.preprocess import load_and_fix_length
from src.features import extract_full_feature_vector


def extract_dataset_features(
    df_index: pd.DataFrame,
    sr: int = 16000,
    duration: float = 1.0,
    n_mfcc: int = 13,
) -> pd.DataFrame:
    """Iterate through audio files, preprocess, extract MFCC summary statistics,

    and construct a tabular feature DataFrame.

    Args:
        df_index: DataFrame containing 'filepath' and 'label' columns.
        sr: Audio sample rate.
        duration: Target clip length in seconds.
        n_mfcc: Number of MFCC coefficients.

    Returns:
        pd.DataFrame containing feature columns, label, and source filepath.
    """
    feature_rows = []

    print(f"Extracting DSP features for {len(df_index)} audio clips...")
    for idx, row in df_index.iterrows():
        filepath = row["filepath"]
        label = row["label"]

        try:
            y = load_and_fix_length(filepath, sr=sr, duration=duration)
            feat_vec, feat_names = extract_full_feature_vector(y, sr=sr, n_mfcc=n_mfcc)

            row_dict = {name: val for name, val in zip(feat_names, feat_vec)}
            row_dict["label"] = label
            row_dict["filepath"] = filepath
            feature_rows.append(row_dict)
        except Exception as e:
            print(f"Warning: Failed to process {filepath}: {e}")

    features_df = pd.DataFrame(feature_rows)
    return features_df


def train_classical_models(
    data_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    models_dir: str = "models",
    samples_per_class: int = 100,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Complete pipeline to build dataset, extract DSP features, train classifiers,

    and persist models.
    """
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 1. Ensure audio data is available
    download_speech_commands_subset(
        data_dir=data_dir,
        selected_classes=DEFAULT_CLASSES,
        samples_per_class=samples_per_class,
    )

    df_index = build_file_index(
        data_dir=data_dir,
        selected_classes=DEFAULT_CLASSES,
        samples_per_class=samples_per_class,
        random_state=random_state,
    )
    if df_index.empty:
        raise RuntimeError(f"No audio files found in {data_dir} for classes {DEFAULT_CLASSES}")

    # 2. Extract DSP Features
    features_csv_path = os.path.join(processed_dir, "features.csv")
    features_df = extract_dataset_features(df_index)
    features_df.to_csv(features_csv_path, index=False)
    print(f"Features table saved to {features_csv_path} (Shape: {features_df.shape})")

    # 3. Prepare Feature Matrix X and Label Vector y
    feature_cols = [c for c in features_df.columns if c not in ["label", "filepath"]]
    X = features_df[feature_cols].values
    y_raw = features_df["label"].values

    # Encode categorical class labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=random_state
    )

    # Standardize features for gradient/margin based models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train RandomForest Classifier
    print("\nTraining Random Forest Classifier (n_estimators=200)...")
    rf_clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=random_state,
        n_jobs=-1,
    )
    rf_clf.fit(X_train, y_train)
    rf_train_acc = accuracy_score(y_train, rf_clf.predict(X_train))
    rf_test_acc = accuracy_score(y_test, rf_clf.predict(X_test))
    print(f"Random Forest -> Train Acc: {rf_train_acc * 100:.2f}%, Test Acc: {rf_test_acc * 100:.2f}%")

    # Train SVM Benchmark
    print("Training Support Vector Machine (RBF Kernel)...")
    svm_clf = SVC(kernel="rbf", C=5.0, gamma="scale", probability=True, random_state=random_state)
    svm_clf.fit(X_train_scaled, y_train)
    svm_train_acc = accuracy_score(y_train, svm_clf.predict(X_train_scaled))
    svm_test_acc = accuracy_score(y_test, svm_clf.predict(X_test_scaled))
    print(f"SVM -> Train Acc: {svm_train_acc * 100:.2f}%, Test Acc: {svm_test_acc * 100:.2f}%")

    # 5. Save Artifacts
    joblib.dump(rf_clf, os.path.join(models_dir, "rf_classifier.pkl"))
    joblib.dump(svm_clf, os.path.join(models_dir, "svm_classifier.pkl"))
    joblib.dump(le, os.path.join(models_dir, "label_encoder.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))

    # Save test dataset split for evaluation module
    np.savez_compressed(
        os.path.join(processed_dir, "test_split.npz"),
        X_test=X_test,
        X_test_scaled=X_test_scaled,
        y_test=y_test,
        feature_cols=feature_cols,
    )
    print("Models and test dataset splits successfully serialized to models/ and data/processed/.")

    return {
        "rf_test_acc": rf_test_acc,
        "svm_test_acc": svm_test_acc,
        "classes": list(le.classes_),
    }


if __name__ == "__main__":
    train_classical_models()
