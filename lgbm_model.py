# ======================================
# RAINFALL MODEL - LIGHTGBM
# ======================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import joblib

# ======================================
# CONFIG
# ======================================
TARGET_COLUMN = 'rainfall_mm'
MODEL_PATH    = 'rainfall_lgbm_model.pkl'

# ======================================
# FEATURE ENGINEERING
# ======================================
def prepare_features(df: pd.DataFrame):
    df = df.copy()

    df['day']         = df['date'].dt.day
    df['month']       = df['date'].dt.month
    df['year']        = df['date'].dt.year
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week']        = df['date'].dt.isocalendar().week.astype(int)

    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin']   = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos']   = np.cos(2 * np.pi * df['day_of_year'] / 365)

    season_map = {'Kharif': 0, 'Rabi': 1, 'Zaid': 2}
    df['season_enc'] = df['season'].map(season_map).fillna(-1).astype(int)

    for col in ['state_name', 'district_name']:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col].astype(str))

    df = df.drop(columns=[
        'date', 'Year', 'Month', 'week_number',
        'state_name', 'district_name',
        'Agency_name', 'source_file',
        'season', 'anomaly_category'
    ], errors='ignore')

    return df

# ======================================
# TRAIN MODEL
# ======================================
def train_model(df: pd.DataFrame, tune_hyperparams: bool = False):

    df = prepare_features(df)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )

    if tune_hyperparams:
        print("\nRunning GridSearchCV...")

        param_grid = {
            'num_leaves': [20, 31, 50],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [300, 500],
        }

        grid = GridSearchCV(
            lgb.LGBMRegressor(random_state=42),
            param_grid,
            cv=3,
            scoring='r2',
            verbose=1,
        )

        grid.fit(X_train, y_train)
        print(f"Best params: {grid.best_params_}")
        model = grid.best_estimator_

    else:
        print("\nTraining LightGBM...")

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='rmse',
            callbacks=[
                lgb.early_stopping(30),
                lgb.log_evaluation(50)
            ]
        )

    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2   = r2_score(y_test, y_pred)

    print("\n" + "="*45)
    print("  MODEL PERFORMANCE (LightGBM)")
    print("="*45)
    print(f"  MAE  : {mae:.4f} mm")
    print(f"  RMSE : {rmse:.4f} mm")
    print(f"  R²   : {r2:.4f}")
    print("="*45)

    _plot_actual_vs_predicted(y_test, y_pred, mae, rmse, r2)
    _plot_feature_importance(model, X_train)

    joblib.dump(model, MODEL_PATH)
    print(f"\n✅ Model saved → {MODEL_PATH}")

    return model

# ======================================
# PLOTS
# ======================================
def _plot_actual_vs_predicted(y_test, y_pred, mae, rmse, r2):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.25)
    lims = [min(y_test.min(), y_pred.min()),
            max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--')
    plt.title(f"LightGBM — R²={r2:.3f}")
    plt.tight_layout()
    plt.show()

def _plot_feature_importance(model, X_train):
    feat_imp = pd.Series(model.feature_importances_, index=X_train.columns)
    feat_imp.nlargest(15).sort_values().plot(kind='barh')
    plt.title("LightGBM Feature Importance")
    plt.tight_layout()
    plt.show()

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
    print("  LIGHTGBM RAINFALL MODEL — START")
    print("=" * 60)

    results = run_pipeline()

    # ✅ FIX: use correct dataframe
    clean_df = results["clean_df"]

    model = train_model(clean_df)

    print("\n✅ LightGBM training complete.")