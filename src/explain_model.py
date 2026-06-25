from pathlib import Path
import joblib
import shap
import matplotlib.pyplot as plt


PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
ARTIFACTS_DIR = Path("artifacts")
SHAP_DIR = ARTIFACTS_DIR / "shap"

MODEL_PATH = MODELS_DIR / "xgboost_model.joblib"
MAX_EXPLAIN_SAMPLES = 500


def ensure_directories():
    SHAP_DIR.mkdir(parents=True, exist_ok=True)


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    X_test = joblib.load(PROCESSED_DIR / "X_test.joblib")
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")

    return model, X_test, label_encoder


def sample_data(X_test):
    if len(X_test) > MAX_EXPLAIN_SAMPLES:
        return X_test.sample(MAX_EXPLAIN_SAMPLES, random_state=42)

    return X_test


def get_shap_values(model, X_sample):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    return shap_values


def save_shap_summary(shap_values, X_sample, class_names):
    if isinstance(shap_values, list):
        for index, class_name in enumerate(class_names):
            plt.figure()
            shap.summary_plot(
                shap_values[index],
                X_sample,
                show=False,
                max_display=20,
            )
            path = SHAP_DIR / f"shap_summary_{class_name}.png"
            plt.tight_layout()
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()

    elif len(shap_values.shape) == 3:
        for index, class_name in enumerate(class_names):
            plt.figure()
            shap.summary_plot(
                shap_values[:, :, index],
                X_sample,
                show=False,
                max_display=20,
            )
            path = SHAP_DIR / f"shap_summary_{class_name}.png"
            plt.tight_layout()
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()

    else:
        plt.figure()
        shap.summary_plot(
            shap_values,
            X_sample,
            show=False,
            max_display=20,
        )
        path = SHAP_DIR / "shap_summary.png"
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()


def save_single_bar_plot(values, feature_names, title, filename):
    mean_abs_values = abs(values).mean(axis=0)
    sorted_indices = mean_abs_values.argsort()[-20:]

    selected_features = [feature_names[i] for i in sorted_indices]
    selected_values = mean_abs_values[sorted_indices]

    plt.figure(figsize=(10, 8))
    plt.barh(selected_features, selected_values)
    plt.xlabel("Mean absolute SHAP value")
    plt.title(title)
    plt.tight_layout()

    path = SHAP_DIR / filename
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_shap_bar(shap_values, X_sample, class_names):
    feature_names = X_sample.columns.tolist()

    if isinstance(shap_values, list):
        for index, class_name in enumerate(class_names):
            save_single_bar_plot(
                shap_values[index],
                feature_names,
                f"SHAP Feature Importance - {class_name}",
                f"shap_feature_importance_bar_{class_name}.png",
            )

    elif len(shap_values.shape) == 3:
        for index, class_name in enumerate(class_names):
            save_single_bar_plot(
                shap_values[:, :, index],
                feature_names,
                f"SHAP Feature Importance - {class_name}",
                f"shap_feature_importance_bar_{class_name}.png",
            )

    else:
        save_single_bar_plot(
            shap_values,
            feature_names,
            "SHAP Feature Importance",
            "shap_feature_importance_bar.png",
        )


def explain_model():
    ensure_directories()

    model, X_test, label_encoder = load_artifacts()
    X_sample = sample_data(X_test)
    class_names = label_encoder.classes_.tolist()

    shap_values = get_shap_values(model, X_sample)

    save_shap_summary(shap_values, X_sample, class_names)
    save_shap_bar(shap_values, X_sample, class_names)

    print("SHAP explanation completed.")
    print(f"SHAP plots saved to: {SHAP_DIR}")
    print(f"Explained samples: {len(X_sample)}")
    print("Classes:")

    for class_name in class_names:
        print(f"  {class_name}")


if __name__ == "__main__":
    explain_model()