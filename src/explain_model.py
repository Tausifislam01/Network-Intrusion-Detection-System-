from pathlib import Path
import joblib
import shap
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
SHAP_DIR = PROJECT_ROOT / "artifacts" / "shap"

XGB_MODEL_PATH = MODELS_DIR / "xgboost_model.joblib"
MAX_EXPLAIN_SAMPLES = 200

def ensure_directories():
    SHAP_DIR.mkdir(parents=True, exist_ok=True)

def load_artifacts():
    model = joblib.load(XGB_MODEL_PATH)
    X_train = joblib.load(PROCESSED_DIR / "X_train.joblib")
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")
    feature_names = joblib.load(MODELS_DIR / "feature_names.joblib")
    return model, X_train, label_encoder, feature_names

def main():
    ensure_directories()
    model, X_train, label_encoder, feature_names = load_artifacts()
    
    X_sample = X_train.sample(n=min(len(X_train), MAX_EXPLAIN_SAMPLES), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    
    fig = plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("Production Network Flow SHAP Analysis")
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "shap_summary_plot.png", dpi=300)
    plt.close(fig)
    print("SHAP Attribution Plots Exported Successfully.")

if __name__ == "__main__":
    main()