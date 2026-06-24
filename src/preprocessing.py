from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler


RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_SAMPLES_PER_CLASS = 10000
MIN_SAMPLES_PER_CLASS = 2

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
ARTIFACTS_DIR = Path("artifacts")


ATTACK_GROUPS = {
    "BENIGN": "BENIGN",
    "DDoS": "DDoS",
    "PortScan": "PortScan",
    "Bot": "Bot",
    "Infiltration": "Infiltration",

    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",

    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",

    "Heartbleed": "Heartbleed",

    "Web Attack � Brute Force": "WebAttack",
    "Web Attack � XSS": "WebAttack",
    "Web Attack � Sql Injection": "WebAttack",
    "Web Attack - Brute Force": "WebAttack",
    "Web Attack - XSS": "WebAttack",
    "Web Attack - Sql Injection": "WebAttack",
    "Web Attack Brute Force": "WebAttack",
    "Web Attack XSS": "WebAttack",
    "Web Attack Sql Injection": "WebAttack",
}


def ensure_directories():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_data(raw_dir=RAW_DATA_DIR):
    raw_dir = Path(raw_dir)
    csv_files = sorted(raw_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {raw_dir}")

    dataframes = []

    for file_path in csv_files:
        print(f"Loading: {file_path.name}")
        df = pd.read_csv(file_path, low_memory=False)
        dataframes.append(df)

    return pd.concat(dataframes, ignore_index=True)


def normalize_columns(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    df.columns = df.columns.str.replace(r"\s+", " ", regex=True)
    return df


def find_label_column(df):
    for col in df.columns:
        if col.strip().lower() == "label":
            return col

    raise ValueError("Label column not found.")


def normalize_labels(y):
    y = y.astype(str).str.strip()
    y = y.str.replace("–", "-", regex=False)
    y = y.str.replace("—", "-", regex=False)
    y = y.str.replace("�", "-", regex=False)
    y = y.str.replace(r"\s+", " ", regex=True)
    y = y.replace(ATTACK_GROUPS)
    return y


def clean_numeric_features(X):
    X = X.copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    return X


def remove_invalid_rows(X, y):
    mask = X.notna().all(axis=1)
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    return X, y


def remove_duplicate_rows(X, y):
    df = X.copy()
    df["__target__"] = y.values
    df = df.drop_duplicates().reset_index(drop=True)

    y_clean = df["__target__"].copy()
    X_clean = df.drop(columns=["__target__"])

    return X_clean, y_clean


def remove_constant_columns(X):
    nunique = X.nunique(dropna=False)
    keep_cols = nunique[nunique > 1].index.tolist()
    removed_cols = sorted(set(X.columns) - set(keep_cols))

    return X[keep_cols].copy(), removed_cols


def remove_rare_classes(X, y, min_samples=MIN_SAMPLES_PER_CLASS):
    counts = y.value_counts()
    valid_classes = counts[counts >= min_samples].index

    mask = y.isin(valid_classes)

    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    return X, y


def cap_classes(X, y, max_samples=MAX_SAMPLES_PER_CLASS):
    counts = y.value_counts()

    strategy = {
        label: min(count, max_samples)
        for label, count in counts.items()
    }

    sampler = RandomUnderSampler(
        sampling_strategy=strategy,
        random_state=RANDOM_STATE
    )

    X_resampled, y_resampled = sampler.fit_resample(X, y)

    X_resampled = X_resampled.reset_index(drop=True)
    y_resampled = pd.Series(y_resampled).reset_index(drop=True)

    return X_resampled, y_resampled


def split_data(X, y):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    return X_train, X_test, y_train, y_test, encoder


def save_outputs(
    X_train,
    X_test,
    y_train,
    y_test,
    encoder,
    feature_names,
    removed_columns,
    class_counts_before_resampling,
    class_counts_after_resampling,
):
    joblib.dump(X_train, PROCESSED_DIR / "X_train.joblib")
    joblib.dump(X_test, PROCESSED_DIR / "X_test.joblib")
    joblib.dump(y_train, PROCESSED_DIR / "y_train.joblib")
    joblib.dump(y_test, PROCESSED_DIR / "y_test.joblib")

    joblib.dump(encoder, MODELS_DIR / "label_encoder.joblib")
    joblib.dump(feature_names, MODELS_DIR / "feature_names.joblib")

    metadata = {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "max_samples_per_class": MAX_SAMPLES_PER_CLASS,
        "min_samples_per_class": MIN_SAMPLES_PER_CLASS,
        "features": len(feature_names),
        "removed_columns": removed_columns,
        "classes": encoder.classes_.tolist(),
        "class_counts_before_resampling": class_counts_before_resampling,
        "class_counts_after_resampling": class_counts_after_resampling,
    }

    with open(
        ARTIFACTS_DIR / "preprocessing_metadata.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(metadata, f, indent=4)


def preprocess_dataset(raw_dir=RAW_DATA_DIR):
    ensure_directories()

    df = load_raw_data(raw_dir)
    df = normalize_columns(df)

    label_col = find_label_column(df)

    y = normalize_labels(df[label_col])
    X = df.drop(columns=[label_col])

    print(f"Initial rows: {len(X)}")
    print(f"Initial features: {X.shape[1]}")

    X = clean_numeric_features(X)
    X, y = remove_invalid_rows(X, y)
    X, y = remove_duplicate_rows(X, y)
    X, removed_columns = remove_constant_columns(X)
    X, y = remove_rare_classes(X, y)

    class_counts_before_resampling = y.value_counts().sort_index().to_dict()

    X, y = cap_classes(X, y)

    class_counts_after_resampling = y.value_counts().sort_index().to_dict()

    X_train, X_test, y_train, y_test, encoder = split_data(X, y)

    feature_names = X_train.columns.tolist()

    save_outputs(
        X_train,
        X_test,
        y_train,
        y_test,
        encoder,
        feature_names,
        removed_columns,
        class_counts_before_resampling,
        class_counts_after_resampling,
    )

    print("Preprocessing completed.")
    print(f"Rows after preprocessing: {len(X)}")
    print(f"Features: {len(feature_names)}")
    print("Classes:")

    for label, count in class_counts_after_resampling.items():
        print(f"  {label}: {count}")

    return X_train, X_test, y_train, y_test, encoder


if __name__ == "__main__":
    preprocess_dataset()