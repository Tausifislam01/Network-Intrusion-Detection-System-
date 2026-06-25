from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from src.evaluate_model import calculate_metrics, save_metrics, save_classification_report, save_confusion_matrix

RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

RF_MODEL_PATH = MODELS_DIR / "random_forest_model.joblib"
XGB_MODEL_PATH = MODELS_DIR / "xgboost_model.joblib"

def ensure_directories():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def load_processed_data():
    X_train = joblib.load(PROCESSED_DIR / "X_train.joblib")
    X_test = joblib.load(PROCESSED_DIR / "X_test.joblib")
    y_train = joblib.load(PROCESSED_DIR / "y_train.joblib")
    y_test = joblib.load(PROCESSED_DIR / "y_test.joblib")
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")
    return X_train, X_test, y_train, y_test, label_encoder

def train_random_forest(X_train, y_train):
    print("Training Random Forest Baseline Model...")
    model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    print("Training Production XGBoost Ensemble Classifier...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def evaluate_and_save(model, X_test, y_test, label_encoder, model_name):
    y_pred = model.predict(X_test)
    class_names = label_encoder.classes_.tolist()
    
    metrics = calculate_metrics(y_test, y_pred)
    metrics_path = save_metrics(metrics, model_name)
    report_path = save_classification_report(y_test, y_pred, class_names, model_name)
    cm_path = save_confusion_matrix(y_test, y_pred, class_names, model_name)
    
    return metrics, metrics_path, report_path, cm_path

def main():
    ensure_directories()
    X_train, X_test, y_train, y_test, label_encoder = load_processed_data()
    
    mlflow.set_experiment("NIDS_Stage2_AllDays")
    
    with mlflow.start_run(run_name="random_forest_baseline"):
        rf_model = train_random_forest(X_train, y_train)
        joblib.dump(rf_model, RF_MODEL_PATH)
        
        rf_metrics, rf_metrics_path, rf_report_path, rf_cm_path = evaluate_and_save(
            rf_model, X_test, y_test, label_encoder, "random_forest"
        )
        
        mlflow.log_params({"model_type": "RandomForestClassifier", "n_estimators": 100, "max_depth": 15})
        mlflow.log_metrics(rf_metrics)
        mlflow.log_artifact(str(rf_metrics_path))
        mlflow.log_artifact(str(rf_report_path))
        mlflow.log_artifact(str(rf_cm_path))
        mlflow.sklearn.log_model(rf_model, "model")
        print("Random Forest run logged successfully to MLflow tracking engine.")

    with mlflow.start_run(run_name="xgboost_production_model"):
        xgb_model = train_xgboost(X_train, y_train)
        joblib.dump(xgb_model, XGB_MODEL_PATH)
        
        xgb_metrics, xgb_metrics_path, xgb_report_path, xgb_cm_path = evaluate_and_save(
            xgb_model, X_test, y_test, label_encoder, "xgboost"
        )
        
        mlflow.log_params({"model_type": "XGBClassifier", "n_estimators": 200, "max_depth": 6, "learning_rate": 0.08})
        mlflow.log_metrics(xgb_metrics)
        mlflow.log_artifact(str(xgb_metrics_path))
        mlflow.log_artifact(str(xgb_report_path))
        mlflow.log_artifact(str(xgb_cm_path))
        mlflow.xgboost.log_model(xgb_model, "model")
        print("Production XGBoost run logged successfully to MLflow tracking engine.")

if __name__ == "__main__":
    main()