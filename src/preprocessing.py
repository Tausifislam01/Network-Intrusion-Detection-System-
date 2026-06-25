from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_SAMPLES_ATTACK = 15000
MAX_SAMPLES_BENIGN = 250000  # Increased to keep normal background noise

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

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
    "Web Attack - Brute Force": "WebAttack",
    "Web Attack - XSS": "WebAttack",
    "Web Attack - Sql Injection": "WebAttack",
}

HELD_OUT_FILE = "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"

# Identifiers that cause data leakage
IDENTIFIER_COLS = [
    "flow id", "source ip", "src ip", "source port", "src port",
    "destination ip", "dst ip", "destination port", "dst port", "timestamp"
]

def ensure_directories():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def load_all_days_data(raw_dir):
    all_files = list(raw_dir.glob("*.csv"))
    train_dfs = []
    
    print(f"Strictly holding out {HELD_OUT_FILE} from training workflow.")
    
    for f_path in all_files:
        if f_path.name == HELD_OUT_FILE:
            continue
        print(f"Ingesting training data day: {f_path.name}")
        df = pd.read_csv(f_path, low_memory=False)
        train_dfs.append(df)
        
    if not train_dfs:
        raise FileNotFoundError("No training files discovered in data/raw/.")
    return pd.concat(train_dfs, ignore_index=True)

def normalize_columns(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    return df

def drop_identifiers(df):
    """Drop columns that can cause the model to memorize specific attacks."""
    cols_to_drop = [c for c in df.columns if c.lower() in IDENTIFIER_COLS]
    print(f"Dropping identifier columns to prevent leakage: {cols_to_drop}")
    return df.drop(columns=cols_to_drop)

def find_label_column(df):
    for col in df.columns:
        if col.lower() == "label":
            return col
    raise KeyError("No network intrusion label column detected.")

def normalize_labels(label_series):
    cleaned = label_series.astype(str).str.strip()
    cleaned = cleaned.str.replace(r"\s+", " ", regex=True)
    return cleaned.map(lambda x: ATTACK_GROUPS.get(x, "Other"))

def clean_numeric_features(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Calculate and save medians for imputation
    medians = df.median(numeric_only=True)
    df = df.fillna(medians)
    return df, medians

def remove_invalid_rows(X, y):
    keep_mask = ~(X.isnull().any(axis=1) | y.isnull())
    return X[keep_mask], y[keep_mask]

def remove_duplicate_rows(X, y):
    combined = pd.concat([X, y], axis=1)
    deduped = combined.drop_duplicates()
    label_col = y.name
    return deduped.drop(columns=[label_col]), deduped[label_col]

def remove_constant_columns(X):
    constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
    return X.drop(columns=constant_cols), constant_cols

def cap_classes(X, y):
    combined = pd.concat([X, y], axis=1)
    label_col = y.name
    
    capped_chunks = []
    for cls, group in combined.groupby(label_col):
        limit = MAX_SAMPLES_BENIGN if cls == "BENIGN" else MAX_SAMPLES_ATTACK
        if len(group) > limit:
            group = group.sample(n=limit, random_state=RANDOM_STATE)
        capped_chunks.append(group)
        
    final_df = pd.concat(capped_chunks, ignore_index=True)
    return final_df.drop(columns=[label_col]), final_df[label_col]

def save_outputs(X_train, X_test, y_train, y_test, encoder, feature_names, medians, metadata):
    joblib.dump(X_train, PROCESSED_DIR / "X_train.joblib")
    joblib.dump(X_test, PROCESSED_DIR / "X_test.joblib")
    joblib.dump(y_train, PROCESSED_DIR / "y_train.joblib")
    joblib.dump(y_test, PROCESSED_DIR / "y_test.joblib")
    joblib.dump(encoder, MODELS_DIR / "label_encoder.joblib")
    joblib.dump(feature_names, MODELS_DIR / "feature_names.joblib")
    joblib.dump(medians, MODELS_DIR / "feature_medians.joblib")
    
    with open(ARTIFACTS_DIR / "preprocessing_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

def generate_held_out_demo_sample():
    held_out_path = RAW_DATA_DIR / HELD_OUT_FILE
    if not held_out_path.exists():
        return
    df = pd.read_csv(held_out_path, low_memory=False)
    df = normalize_columns(df)
    sample_df = df.sample(n=2000, random_state=RANDOM_STATE)
    sample_df.to_csv(PROJECT_ROOT / "demo_unseen_sample.csv", index=False)

def preprocess_dataset():
    ensure_directories()
    df = load_all_days_data(RAW_DATA_DIR)
    df = normalize_columns(df)
    df = drop_identifiers(df)
    
    label_col = find_label_column(df)
    y = normalize_labels(df[label_col])
    X = df.drop(columns=[label_col])
    
    X, medians = clean_numeric_features(X)
    X, y = remove_invalid_rows(X, y)
    X, y = remove_duplicate_rows(X, y)
    X, removed_columns = remove_constant_columns(X)
    
    class_counts_before = y.value_counts().to_dict()
    X, y = cap_classes(X, y)
    class_counts_after = y.value_counts().to_dict()
    
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_split=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
    )
    
    feature_names = X_train.columns.tolist()
    
    metadata = {
        "features_after_preprocessing": len(feature_names),
        "removed_constant_columns": removed_columns,
        "class_counts_before_resampling": class_counts_before,
        "class_counts_after_resampling": class_counts_after,
        "held_out_file_identity": HELD_OUT_FILE
    }
    
    save_outputs(X_train, X_test, y_train, y_test, encoder, feature_names, medians, metadata)
    generate_held_out_demo_sample()
    print("Stage 2 Preprocessing Completed Successfully.")

if __name__ == "__main__":
    preprocess_dataset()