from pathlib import Path
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODELS_DIR / "xgboost_model.joblib"
ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.joblib"

def load_prediction_artifacts():
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    return model, label_encoder, feature_names

def clean_input_features(df, feature_names):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    
    if "label" in [c.lower() for c in df.columns]:
        df = df.drop(columns=[c for c in df.columns if c.lower() == "label"])
        
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    df = df.replace([np.inf, -np.inf], np.nan)
    
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0.0
            
    return df[feature_names]

def predict_dataframe(df):
    model, label_encoder, feature_names = load_prediction_artifacts()
    aligned_df = clean_input_features(df, feature_names)
    aligned_df = aligned_df.apply(lambda x: x.fillna(x.median()) if x.isnull().any() else x)
    
    encoded_preds = model.predict(aligned_df)
    predicted_labels = label_encoder.inverse_transform(encoded_preds)
    
    result = df.copy()
    result["prediction"] = predicted_labels
    if hasattr(model, "predict_proba"):
        result["confidence"] = model.predict_proba(aligned_df).max(axis=1)
    return result

if __name__ == "__main__":
    test_path = PROJECT_ROOT / "demo_unseen_sample.csv"
    if test_path.exists():
        input_df = pd.read_csv(test_path, low_memory=False)
        out = predict_dataframe(input_df)
        print("Predictions Summary on Held-Out Validation Slice:")
        print(out["prediction"].value_counts())