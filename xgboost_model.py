# ======================================
# RAINFALL MODEL - XGBOOST
# ======================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib

# ======================================
# CONFIG
# ======================================
TARGET_COLUMN = 'rainfall_mm'      # matches pipeline output
MODEL_PATH    = 'rainfall_xgb_model.pkl'

# ======================================
# FEATURE ENGINEERING
# ======================================
def prepare_features(df: pd.DataFrame):
    df = df.copy()

    # ── Time features ──────────────────────────────────────────
    df['day']         = df['date'].dt.day
    df['month']       = df['date'].dt.month
    df['year']        = df['date'].dt.year
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week']        = df['date'].dt.isocalendar().week.astype(int)

    # Cyclical encoding for month and day_of_year so XGBoost
    # understands Jan (1) and Dec (12) are close in the calendar cycle
    df['month_sin']      = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos']      = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_yr_sin']  = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_yr_cos']  = np.cos(2 * np.pi * df['day_of_year'] / 365)

    # ── Encode season as ordinal ────────────────────────────────
    season_map = {'Kharif': 0, 'Rabi': 1, 'Zaid': 2}
    df['season_enc'] = df['season'].map(season_map).fillna(-1).astype(int)

    # ── Label-encode categoricals (XGBoost prefers integers) ───
    for col in ['state_name', 'district_name']:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col].astype(str))

    # ── Drop columns not used as features ──────────────────────
    drop_cols = [
        'date', 'Year', 'Month', 'week_number',
        'state_name', 'district_name',
        'Agency_name', 'source_file',
        'season', 'anomaly_category',
    ]
    df = df.drop(columns=drop_cols, errors='ignore')

    return df


# ======================================
# TRAIN MODEL
# ======================================
def train_model(df: pd.DataFrame, tune_hyperparams: bool = False):
    """
    Train an XGBoost regressor on the cleaned pipeline dataframe.

    Parameters
    ----------
    df               : clean_df returned by run_pipeline()
    tune_hyperparams : if True, runs a small GridSearchCV to find better
                       hyperparameters (slower but often improves R²)
    """
    df = prepare_features(df)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── Model definition ───────────────────────────────────────
    base_params = dict(
        n_estimators      = 500,
        learning_rate     = 0.05,
        max_depth         = 6,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        min_child_weight  = 3,
        gamma             = 0.1,
        reg_alpha         = 0.1,      # L1 regularisation
        reg_lambda        = 1.0,      # L2 regularisation
        random_state      = 42,
        n_jobs            = -1,
        early_stopping_rounds = 30,   # stop if no improvement for 30 rounds
    )

    model = xgb.XGBRegressor(**base_params)

    if tune_hyperparams:
        print("\nRunning GridSearchCV (this may take a few minutes)...")
        param_grid = {
            'max_depth'    : [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators' : [300, 500],
        }
        grid = GridSearchCV(
            xgb.XGBRegressor(random_state=42, n_jobs=-1),
            param_grid,
            cv=3,
            scoring='r2',
            verbose=1,
        )
        grid.fit(X_train, y_train)
        print(f"Best params : {grid.best_params_}")
        model = grid.best_estimator_
    else:
        print("\nTraining XGBoost...")
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=50,             # print loss every 50 rounds
        )

    # ── Evaluation ─────────────────────────────────────────────
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2   = r2_score(y_test, y_pred)

    print("\n" + "="*45)
    print("  MODEL PERFORMANCE")
    print("="*45)
    print(f"  MAE  : {mae:.4f} mm")
    print(f"  RMSE : {rmse:.4f} mm")
    print(f"  R²   : {r2:.4f}")
    print("="*45)

    # ── Plots ──────────────────────────────────────────────────
    _plot_actual_vs_predicted(y_test, y_pred, mae, rmse, r2)
    _plot_feature_importance(model, X_train)

    # ── Save ───────────────────────────────────────────────────
    joblib.dump(model, MODEL_PATH)
    print(f"\n✅ Model saved → {MODEL_PATH}")

    return model


# ======================================
# PLOTS
# ======================================
def _plot_actual_vs_predicted(y_test, y_pred, mae, rmse, r2):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.25, edgecolors='k',
                linewidths=0.3, color='steelblue', label='Predictions')
    lims = [min(y_test.min(), y_pred.min()),
            max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--', lw=2, label='Perfect fit')
    plt.xlabel("Actual Rainfall (mm)")
    plt.ylabel("Predicted Rainfall (mm)")
    plt.title(
        f"XGBoost — Actual vs Predicted\n"
        f"R²={r2:.3f}   MAE={mae:.2f} mm   RMSE={rmse:.2f} mm"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig("xgb_actual_vs_predicted.png", dpi=150)
    plt.show()
    print("  📊 Saved → xgb_actual_vs_predicted.png")


def _plot_feature_importance(model, X_train):
    feat_imp = pd.Series(model.feature_importances_, index=X_train.columns)
    feat_imp.nlargest(15).sort_values().plot(
        kind='barh', figsize=(8, 6), color='steelblue'
    )
    plt.title("XGBoost — Top 15 Feature Importances")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("xgb_feature_importance.png", dpi=150)
    plt.show()
    print("  📊 Saved → xgb_feature_importance.png")


# ======================================
# LOAD + PREDICT
# ======================================
def load_model():
    return joblib.load(MODEL_PATH)


def predict(model, df: pd.DataFrame):
    df = prepare_features(df)
    df = df.drop(columns=[TARGET_COLUMN], errors='ignore')
    return model.predict(df)


# ======================================
# ENTRY POINT
# ======================================
if __name__ == "__main__":
    from rainfall_pipeline import run_pipeline

    print("=" * 60)
    print("  XGBOOST RAINFALL MODEL — START")
    print("=" * 60)

    clean_df = run_pipeline()

    # Set tune_hyperparams=True for a slower but better-tuned model
    model = train_model(clean_df, tune_hyperparams=False)

    print("\n✅ XGBoost training complete.")