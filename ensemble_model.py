# ======================================
# RAINFALL ENSEMBLE CLASSIFIER
# ======================================

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

# ======================================
# CONFIG
# ======================================
MODEL_PATH = "rainfall_ensemble_model.pkl"

# ======================================
# CLASSIFICATION LOGIC
# ======================================
def classify_rainfall(dep):
    if dep >= 20:
        return "Above Normal"
    elif dep >= -20:
        return "Normal"
    else:
        return "Deficit"

# ======================================
# FEATURE ENGINEERING
# ======================================
def prepare_features(df, is_predict=False):
    df = df.copy()

    # --- target (only during training) ---
    if not is_predict:
        df["rainfall_class"] = df["departure_pct"].apply(classify_rainfall)
        le_target = LabelEncoder()
        df["target"] = le_target.fit_transform(df["rainfall_class"])
    else:
        le_target = None

    # --- time features ---
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # --- encode categoricals ---
    le_district = LabelEncoder()
    le_state = LabelEncoder()

    df["district_name_enc"] = le_district.fit_transform(df["district_name"].astype(str))
    df["state_name_enc"] = le_state.fit_transform(df["state_name"].astype(str))

    # --- DROP ALL NON-NUMERIC / UNUSED ---
    drop_cols = [
        "rainfall_class",
        "date",
        "Year",
        "Month",
        "season",
        "anomaly_category",
        "state_name",
        "district_name",
        "departure_pct",  # CRITICAL: drop this so it doesn't reach the model
    ]
    df = df.drop(columns=drop_cols, errors="ignore")

    return df, le_target

# ======================================
# TRAIN MODEL
# ======================================
def train_model(df):

    df, le_target = prepare_features(df, is_predict=False)

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- MODELS ---
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42
    )

    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )

    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42
    )

    # --- ENSEMBLE ---
    model = VotingClassifier(
        estimators=[
            ("lgb", lgb_model),
            ("xgb", xgb_model),
            ("rf", rf_model)
        ],
        voting="soft"
    )

    print("\nTraining Ensemble Model...")
    model.fit(X_train, y_train)

    # --- EVALUATION ---
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("\n" + "="*45)
    print("ENSEMBLE MODEL PERFORMANCE")
    print("="*45)
    print(f"Accuracy: {acc * 100:.2f}%")

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    # --- SAVE ---
    joblib.dump({
        "model": model,
        "label_encoder": le_target
    }, MODEL_PATH)

    print(f"\n✅ Model saved → {MODEL_PATH}")

    return model

# ======================================
# LOAD + PREDICT
# ======================================
def load_model():
    return joblib.load(MODEL_PATH)

def predict(model_dict, df):
    model = model_dict["model"]
    le = model_dict["label_encoder"]

    df, _ = prepare_features(df, is_predict=True)
    df = df.drop(columns=["target"], errors="ignore")

    preds = model.predict(df)
    labels = le.inverse_transform(preds)

    probs = model.predict_proba(df)

    return labels, probs

# ======================================
# ENTRY POINT
# ======================================
if __name__ == "__main__":
    from rainfall_pipeline import run_pipeline

    print("=" * 60)
    print("ENSEMBLE RAINFALL MODEL — START")
    print("=" * 60)

    results = run_pipeline()

    # IMPORTANT: use monthly data (has departure_pct)
    df = results["monthly"]

    train_model(df)

    print("\n✅ Ensemble training complete.")