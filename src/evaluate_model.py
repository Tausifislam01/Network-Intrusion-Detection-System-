from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
ARTIFACTS_DIR = Path("artifacts")
EVALUATION_DIR = ARTIFACTS_DIR / "evaluation"

MODEL_PATH = MODELS_DIR / "xgboost_model.joblib"
MODEL_NAME = "xgboost"


def ensure_directories():
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)


def load_artifacts():
    X_test = joblib.load(PROCESSED_DIR / "X_test.joblib")
    y_test = joblib.load(PROCESSED_DIR / "y_test.joblib")
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")

    return X_test, y_test, model, label_encoder


def calculate_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
    }


def save_metrics(metrics):
    path = EVALUATION_DIR / f"{MODEL_NAME}_evaluation_metrics.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    return path


def save_classification_report(y_true, y_pred, class_names):
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )

    path = EVALUATION_DIR / f"{MODEL_NAME}_classification_report.txt"

    with open(path, "w", encoding="utf-8") as f:
        f.write(report)

    return path


def save_confusion_matrix(y_true, y_pred, class_names):
    matrix = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )
    display.plot(ax=ax, xticks_rotation=45, values_format="d")
    plt.title("XGBoost Evaluation Confusion Matrix")
    plt.tight_layout()

    path = EVALUATION_DIR / f"{MODEL_NAME}_confusion_matrix.png"
    plt.savefig(path, dpi=300)
    plt.close(fig)

    return path


def evaluate_model():
    ensure_directories()

    X_test, y_test, model, label_encoder = load_artifacts()
    y_pred = model.predict(X_test)

    class_names = label_encoder.classes_.tolist()

    metrics = calculate_metrics(y_test, y_pred)
    metrics_path = save_metrics(metrics)
    report_path = save_classification_report(y_test, y_pred, class_names)
    matrix_path = save_confusion_matrix(y_test, y_pred, class_names)

    print("Evaluation completed.")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Classification report saved to: {report_path}")
    print(f"Confusion matrix saved to: {matrix_path}")

    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    evaluate_model()