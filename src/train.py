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
    n_mfcc: int = 20,
) -> pd.DataFrame:
    """Iterate through audio files, preprocess, extract advanced temporal-segmented DSP features,
    and construct a tabular feature DataFrame.
    """
    from src.features import extract_advanced_dsp_vector

    feature_rows = []
    print(f"Extracting temporal-segmented DSP features (n_mfcc={n_mfcc}, 3 segments + deltas) for {len(df_index)} clips...")

    for idx, row in df_index.iterrows():
        filepath = row["filepath"]
        label = row["label"]

        try:
            y = load_and_fix_length(filepath, sr=sr, duration=duration)
            feat_vec, feat_names = extract_advanced_dsp_vector(y, sr=sr, n_mfcc=n_mfcc, n_segments=3)

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
    features_df = extract_dataset_features(df_index, n_mfcc=20)
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

    # 4. Train RandomForest Classifier with tuned parameters
    print("\nTraining Tuned Random Forest Classifier (n_estimators=300, min_samples_split=3)...")
    rf_clf = RandomForestClassifier(
        n_estimators=300,
        min_samples_split=3,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )
    rf_clf.fit(X_train, y_train)
    rf_train_acc = accuracy_score(y_train, rf_clf.predict(X_train))
    rf_test_acc = accuracy_score(y_test, rf_clf.predict(X_test))
    print(f"Random Forest -> Train Acc: {rf_train_acc * 100:.2f}%, Test Acc: {rf_test_acc * 100:.2f}%")

    # Train Tuned SVM Benchmark (RBF Kernel with C=10.0)
    print("Training Tuned Support Vector Machine (RBF Kernel, C=10.0)...")
    svm_clf = SVC(kernel="rbf", C=10.0, gamma="scale", random_state=random_state)
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
