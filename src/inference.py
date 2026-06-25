from pathlib import Path
import joblib
import numpy as np
import pandas as pd


MODELS_DIR = Path("models")

MODEL_PATH = MODELS_DIR / "xgboost_model.joblib"
ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.joblib"


def load_prediction_artifacts():
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)

    return model, label_encoder, feature_names


def normalize_columns(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    df.columns = df.columns.str.replace(r"\s+", " ", regex=True)
    return df


def remove_label_column(df):
    df = df.copy()

    for column in df.columns:
        if column.strip().lower() == "label":
            df = df.drop(columns=[column])
            break

    return df


def clean_input_features(df):
    df = df.copy()
    df = normalize_columns(df)
    df = remove_label_column(df)

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    return df


def align_features(df, feature_names):
    df = df.copy()

    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0

    df = df[feature_names]

    return df


def predict_dataframe(df):
    model, label_encoder, feature_names = load_prediction_artifacts()

    cleaned_df = clean_input_features(df)
    aligned_df = align_features(cleaned_df, feature_names)

    encoded_predictions = model.predict(aligned_df)
    predicted_labels = label_encoder.inverse_transform(encoded_predictions)

    result = df.copy()
    result["prediction"] = predicted_labels

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(aligned_df)
        confidence = probabilities.max(axis=1)
        result["confidence"] = confidence

    return result


def predict_csv(input_path, output_path=None):
    input_path = Path(input_path)
    df = pd.read_csv(input_path, low_memory=False)

    result = predict_dataframe(df)

    if output_path is not None:
        output_path = Path(output_path)
        result.to_csv(output_path, index=False)

    return result


if __name__ == "__main__":
    sample_path = Path("data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")

    if sample_path.exists():
        predictions = predict_csv(sample_path)
        print(predictions[["prediction", "confidence"]].head())
        print(predictions["prediction"].value_counts())
    else:
        print("Sample raw CSV not found.")