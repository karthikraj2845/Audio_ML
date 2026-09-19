"""
Evaluation Module for Audio ML Pipeline
Evaluates Random Forest, SVM, PyTorch MLP, and 2D AudioCNN classifiers on the test split,
generates classification reports, confusion matrices, and benchmark comparison plots.
"""

from typing import Dict, Any
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import torch

from src.visualize import plot_confusion_matrix, plot_model_comparison
from src.train_dl import SpeechMLP
from src.train_cnn import AudioCNN


def evaluate_all_models(
    processed_dir: str = "data/processed",
    models_dir: str = "models",
    outputs_dir: str = "outputs",
) -> Dict[str, Any]:
    """Load serialized models and held-out test data, evaluate metrics, and produce visualizations."""
    os.makedirs(outputs_dir, exist_ok=True)

    # 1. Load tabular test data and preprocessors
    test_split_path = os.path.join(processed_dir, "test_split.npz")
    if not os.path.exists(test_split_path):
        raise FileNotFoundError(f"Test split {test_split_path} not found. Run training first.")

    data = np.load(test_split_path, allow_pickle=True)
    X_test = data["X_test"]
    X_test_scaled = data["X_test_scaled"]
    y_test = data["y_test"]

    le_path = os.path.join(models_dir, "label_encoder.pkl")
    le = joblib.load(le_path)
    class_names = list(le.classes_)

    results_summary = {}
    best_acc = 0.0
    best_cm = None
    best_model_name = ""

    print("\n" + "=" * 65)
    print("           AUDIO CLASSIFIER EVALUATION REPORT")
    print("=" * 65)

    # 2. Evaluate Random Forest
    rf_path = os.path.join(models_dir, "rf_classifier.pkl")
    if os.path.exists(rf_path):
        rf_clf = joblib.load(rf_path)
        y_pred_rf = rf_clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred_rf)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred_rf, average="weighted")
        results_summary["Random Forest"] = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        }

        print("\n[1] RANDOM FOREST CLASSIFIER (Temporal DSP):")
        print(classification_report(y_test, y_pred_rf, target_names=class_names, digits=4))

        if acc > best_acc:
            best_acc = acc
            best_cm = confusion_matrix(y_test, y_pred_rf)
            best_model_name = "Random Forest"

    # 3. Evaluate SVM
    svm_path = os.path.join(models_dir, "svm_classifier.pkl")
    if os.path.exists(svm_path):
        svm_clf = joblib.load(svm_path)
        y_pred_svm = svm_clf.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred_svm)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred_svm, average="weighted")
        results_summary["SVM (RBF)"] = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        }

        print("\n[2] SUPPORT VECTOR MACHINE (RBF Kernel, Temporal DSP):")
        print(classification_report(y_test, y_pred_svm, target_names=class_names, digits=4))

        if acc > best_acc:
            best_acc = acc
            best_cm = confusion_matrix(y_test, y_pred_svm)
            best_model_name = "Support Vector Machine"

    # 4. Evaluate PyTorch MLP
    mlp_path = os.path.join(models_dir, "mlp_classifier.pt")
    if os.path.exists(mlp_path):
        checkpoint = torch.load(mlp_path, map_location="cpu")
        model = SpeechMLP(in_dim=checkpoint["in_dim"], n_classes=checkpoint["n_classes"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        with torch.no_grad():
            tensor_X = torch.tensor(X_test_scaled, dtype=torch.float32)
            logits = model(tensor_X)
            y_pred_mlp = logits.argmax(dim=1).numpy()

        acc = accuracy_score(y_test, y_pred_mlp)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred_mlp, average="weighted")
        results_summary["PyTorch MLP"] = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        }

        print("\n[3] PYTORCH MULTI-LAYER PERCEPTRON (Temporal DSP):")
        print(classification_report(y_test, y_pred_mlp, target_names=class_names, digits=4))

        if acc > best_acc:
            best_acc = acc
            best_cm = confusion_matrix(y_test, y_pred_mlp)
            best_model_name = "PyTorch MLP"

    # 5. Evaluate 2D AudioCNN
    cnn_path = os.path.join(models_dir, "cnn_classifier.pt")
    cnn_test_path = os.path.join(processed_dir, "test_cnn_split.npz")
    if os.path.exists(cnn_path) and os.path.exists(cnn_test_path):
        cnn_data = np.load(cnn_test_path, allow_pickle=True)
        specs = cnn_data["specs"]
        y_cnn_test = cnn_data["labels"]

        checkpoint = torch.load(cnn_path, map_location="cpu")
        cnn_model = AudioCNN(n_classes=checkpoint["n_classes"])
        cnn_model.load_state_dict(checkpoint["model_state_dict"])
        cnn_model.eval()

        with torch.no_grad():
            tensor_specs = torch.tensor(specs, dtype=torch.float32).unsqueeze(1)
            logits = cnn_model(tensor_specs)
            y_pred_cnn = logits.argmax(dim=1).numpy()

        acc = accuracy_score(y_cnn_test, y_pred_cnn)
        prec, rec, f1, _ = precision_recall_fscore_support(y_cnn_test, y_pred_cnn, average="weighted")
        results_summary["2D AudioCNN"] = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        }

        print("\n[4] 2D AUDIO CONVOLUTIONAL NEURAL NETWORK (Log-Mel Spectrograms):")
        print(classification_report(y_cnn_test, y_pred_cnn, target_names=class_names, digits=4))

        if acc > best_acc:
            best_acc = acc
            best_cm = confusion_matrix(y_cnn_test, y_pred_cnn)
            best_model_name = "2D AudioCNN"

    # Save Best Confusion Matrix Plot
    if best_cm is not None:
        plot_confusion_matrix(
            best_cm,
            class_names=class_names,
            output_path=os.path.join(outputs_dir, "confusion_matrix.png"),
            title=f"Confusion Matrix - {best_model_name} (Acc: {best_acc*100:.1f}%)",
        )
        print(f"\nBest model confusion matrix ({best_model_name}) saved to {os.path.join(outputs_dir, 'confusion_matrix.png')}")

    # Generate Side-by-Side Model Comparison Chart
    if len(results_summary) > 1:
        plot_model_comparison(
            results_summary,
            output_path=os.path.join(outputs_dir, "model_comparison.png"),
        )
        print(f"Model comparison chart saved to {os.path.join(outputs_dir, 'model_comparison.png')}")

    # Print Benchmark Summary Table
    print("\n" + "=" * 65)
    print(f"{'Model Architecture':<28} | {'Accuracy':<10} | {'Weighted F1':<12}")
    print("-" * 65)
    for model_name, metrics in results_summary.items():
        print(f"{model_name:<28} | {metrics['accuracy']*100:>8.2f}% | {metrics['f1']*100:>10.2f}%")
    print("=" * 65 + "\n")

    return results_summary


if __name__ == "__main__":
    evaluate_all_models()
