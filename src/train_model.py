from pathlib import Path
import json
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier


RANDOM_STATE = 42

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
ARTIFACTS_DIR = Path("artifacts")
PLOTS_DIR = ARTIFACTS_DIR / "plots"
METRICS_DIR = ARTIFACTS_DIR / "metrics"

RF_MODEL_PATH = MODELS_DIR / "random_forest_model.joblib"
XGB_MODEL_PATH = MODELS_DIR / "xgboost_model.joblib"


def ensure_directories():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)


def load_processed_data():
    X_train = joblib.load(PROCESSED_DIR / "X_train.joblib")
    X_test = joblib.load(PROCESSED_DIR / "X_test.joblib")
    y_train = joblib.load(PROCESSED_DIR / "y_train.joblib")
    y_test = joblib.load(PROCESSED_DIR / "y_test.joblib")
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")

    return X_train, X_test, y_train, y_test, label_encoder


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


def save_classification_report(y_true, y_pred, class_names, model_name):
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )

    report_path = METRICS_DIR / f"{model_name}_classification_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    return report_path


def save_metrics(metrics, model_name):
    metrics_path = METRICS_DIR / f"{model_name}_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    return metrics_path


def save_confusion_matrix(y_true, y_pred, class_names, model_name):
    matrix = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )
    display.plot(ax=ax, xticks_rotation=45, values_format="d")
    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()

    plot_path = PLOTS_DIR / f"{model_name}_confusion_matrix.png"
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)

    return plot_path


def evaluate_and_save(model, X_test, y_test, label_encoder, model_name):
    y_pred = model.predict(X_test)
    class_names = label_encoder.classes_.tolist()

    metrics = calculate_metrics(y_test, y_pred)

    metrics_path = save_metrics(metrics, model_name)
    report_path = save_classification_report(
        y_test,
        y_pred,
        class_names,
        model_name,
    )
    confusion_matrix_path = save_confusion_matrix(
        y_test,
        y_pred,
        class_names,
        model_name,
    )

    return metrics, metrics_path, report_path, confusion_matrix_path


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, num_classes):
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    return model


def run_mlflow_training():
    ensure_directories()

    X_train, X_test, y_train, y_test, label_encoder = load_processed_data()
    num_classes = len(label_encoder.classes_)

    mlflow.set_experiment("nids_cicids2017")

    with mlflow.start_run(run_name="random_forest_baseline"):
        rf_model = train_random_forest(X_train, y_train)

        rf_metrics, rf_metrics_path, rf_report_path, rf_cm_path = evaluate_and_save(
            rf_model,
            X_test,
            y_test,
            label_encoder,
            "random_forest",
        )

        joblib.dump(rf_model, RF_MODEL_PATH)

        mlflow.log_params({
            "model": "RandomForestClassifier",
            "n_estimators": 150,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
        })
        mlflow.log_metrics(rf_metrics)
        mlflow.log_artifact(str(rf_metrics_path))
        mlflow.log_artifact(str(rf_report_path))
        mlflow.log_artifact(str(rf_cm_path))
        mlflow.sklearn.log_model(rf_model, "model")

    with mlflow.start_run(run_name="xgboost_main_model"):
        xgb_model = train_xgboost(X_train, y_train, num_classes)

        xgb_metrics, xgb_metrics_path, xgb_report_path, xgb_cm_path = evaluate_and_save(
            xgb_model,
            X_test,
            y_test,
            label_encoder,
            "xgboost",
        )

        joblib.dump(xgb_model, XGB_MODEL_PATH)

        mlflow.log_params({
            "model": "XGBClassifier",
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.08,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "tree_method": "hist",
            "random_state": RANDOM_STATE,
        })
        mlflow.log_metrics(xgb_metrics)
        mlflow.log_artifact(str(xgb_metrics_path))
        mlflow.log_artifact(str(xgb_report_path))
        mlflow.log_artifact(str(xgb_cm_path))
        mlflow.xgboost.log_model(xgb_model, "model")

    print("Training completed.")
    print(f"Random Forest model saved to: {RF_MODEL_PATH}")
    print(f"XGBoost model saved to: {XGB_MODEL_PATH}")
    print("Artifacts saved to artifacts/")
    print("MLflow runs saved to mlruns/")


if __name__ == "__main__":
    run_mlflow_training()