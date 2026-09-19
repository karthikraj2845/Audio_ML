"""
Evaluation Module for Audio ML Pipeline
Evaluates Random Forest, SVM, and PyTorch MLP classifiers on the test split,
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


def evaluate_all_models(
    processed_dir: str = "data/processed",
    models_dir: str = "models",
    outputs_dir: str = "outputs",
) -> Dict[str, Any]:
    """Load serialized models and held-out test data, evaluate metrics, and produce visualizations."""
    os.makedirs(outputs_dir, exist_ok=True)

    # 1. Load test data and preprocessors
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
    print("\n" + "=" * 60)
    print("           AUDIO CLASSIFIER EVALUATION REPORT")
    print("=" * 60)

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

        print("\n[1] RANDOM FOREST CLASSIFIER:")
        print(classification_report(y_test, y_pred_rf, target_names=class_names, digits=4))

        # Save confusion matrix plot for primary model
        cm_rf = confusion_matrix(y_test, y_pred_rf)
        plot_confusion_matrix(
            cm_rf,
            class_names=class_names,
            output_path=os.path.join(outputs_dir, "confusion_matrix.png"),
            title="Confusion Matrix - Random Forest (Speech Commands)",
        )
        print(f"Confusion matrix saved to {os.path.join(outputs_dir, 'confusion_matrix.png')}")

    # 3. Evaluate SVM
    svm_path = os.path.join(models_dir, "svm_classifier.pkl")
    if os.path.exists(svm_path):
        svm_clf = joblib.load(svm_path)
        y_pred_svm = svm_clf.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred_svm)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred_svm, average="weighted")
        results_summary["Support Vector Machine"] = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        }

        print("\n[2] SUPPORT VECTOR MACHINE (RBF Kernel):")
        print(classification_report(y_test, y_pred_svm, target_names=class_names, digits=4))

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

        print("\n[3] PYTORCH MULTI-LAYER PERCEPTRON (MLP):")
        print(classification_report(y_test, y_pred_mlp, target_names=class_names, digits=4))

    # 5. Generate Side-by-Side Model Comparison Chart
    if len(results_summary) > 1:
        plot_model_comparison(
            results_summary,
            output_path=os.path.join(outputs_dir, "model_comparison.png"),
        )
        print(f"Model comparison chart saved to {os.path.join(outputs_dir, 'model_comparison.png')}")

    # 6. Print Benchmark Summary Table
    print("\n" + "=" * 60)
    print(f"{'Model':<25} | {'Accuracy':<10} | {'Weighted F1':<12}")
    print("-" * 60)
    for model_name, metrics in results_summary.items():
        print(f"{model_name:<25} | {metrics['accuracy']*100:>8.2f}% | {metrics['f1']*100:>10.2f}%")
    print("=" * 60 + "\n")

    return results_summary


if __name__ == "__main__":
    evaluate_all_models()
