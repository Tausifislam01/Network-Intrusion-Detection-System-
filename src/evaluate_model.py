from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EVALUATION_DIR = ARTIFACTS_DIR / "evaluation"

def calculate_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)
    return {
        "accuracy": float(acc),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1)
    }

def save_metrics(metrics, model_name):
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    path = EVALUATION_DIR / f"{model_name}_metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    return path

def save_classification_report(y_true, y_pred, class_names, model_name):
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    report = classification_report(y_true, y_pred, target_names=class_names)
    path = EVALUATION_DIR / f"{model_name}_classification_report.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    return path

def save_confusion_matrix(y_true, y_pred, class_names, model_name):
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)
    display.plot(ax=ax, xticks_rotation=45, values_format="d", cmap="Blues")
    plt.title(f"{model_name.replace('_', ' ').title()} Matrix Results")
    plt.tight_layout()
    
    path = EVALUATION_DIR / f"{model_name}_confusion_matrix.png"
    plt.savefig(path, dpi=300)
    plt.close(fig)
    return path