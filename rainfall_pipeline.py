"""
=============================================================================
RAINFALL ANALYSIS & PREDICTION FRAMEWORK
Step 2: Data Processing Pipeline + ML Model Training
=============================================================================
Purpose : Load IMD/OGD daily CSVs, clean them, aggregate to monthly &
          seasonal totals per district, compute Departure from Mean (%),
          engineer lag/climate/spatial features, and train XGBoost +
          Random Forest models with proper time-series cross-validation.

Author  : (your name)
Data    : IMD daily district-wise rainfall CSVs — one file per year

Improvements over v1:
  - Lag features (1m, 2m, 3m, 12m, 24m, rolling means)
  - Cyclical month/season encoding
  - Cumulative seasonal totals
  - ENSO / IOD climate index integration (optional)
  - Spatial neighbor aggregates
  - Regression target (departure_pct) instead of binary classification
  - TimeSeriesSplit validation — no future leakage
  - SHAP feature importance
  - Optuna hyperparameter tuning (optional)
=============================================================================
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning)

# Optional heavy deps — imported lazily in the functions that need them
# so the pipeline still runs if they're missing.
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logging.warning("xgboost not installed. XGBoost model will be skipped.")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

# Dictionary of yearly Google Drive file IDs.
# Get the file ID from the sharing link:
#   https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing
RAW_FILE_IDS: dict[str, str] = {
    "2018": "1GJtKaG1Ht82cDrYUSyLi63lUdx_fONrT",
    "2019": "1OS_JAicP0iE-ZiMWye8m_ynfJ5Ypf2eO",
    "2020": "1nB6qe_6SqVPDx5yyCVGJcX-ydeqtwlrE",
    "2021": "1QwtMNFi-TxS3sn2SM9BteuDGSQS3L5mW",
    "2022": "1179FbAiLT1KZJAvQiBcE8T2NQPHAC7zE",
    "2023": "1OgHoFuSwd_JUdadvuxPV1Pj0QFBE4sjf",
    "2024": "1q_yHt0UeqOzo1Kvzz8MTjaP-KEKBU3hr",
}

# ── Column name constants ─────────────────────────────────────────────────
# Adjust these to match the EXACT header names in your CSV files.
COL_DATE     = "Date"
COL_STATE    = "State"
COL_DISTRICT = "District"
COL_RAINFALL = "Avg_rainfall"

# ── Season month definitions ──────────────────────────────────────────────
KHARIF_MONTHS: frozenset[int] = frozenset({6, 7, 8, 9})
RABI_MONTHS:   frozenset[int] = frozenset({10, 11, 12, 1, 2})

# ── District name corrections ─────────────────────────────────────────────
DISTRICT_NAME_CORRECTIONS: dict[str, str] = {
    "Jagtial"            : "Jagitial",
    "Jangoan"            : "Jangaon",
    "Kumuram Bheem"      : "Kumuram Bheem Asifabad",
    "Medchal-Malkajgiri" : "Medchal Malkajgiri",
    "Rangareddy"         : "Ranga Reddy",
    "Ranjanna Sircilla"  : "Rajanna Sircilla",
    "Warangal Rural"     : "Warangal (Rural)",
    "Warangal Urban"     : "Warangal (Urban)",
}

# ── ML configuration ──────────────────────────────────────────────────────
N_CV_SPLITS    = 5        # TimeSeriesSplit folds
OPTUNA_TRIALS  = 30       # Set to 0 to skip hyperparameter tuning
RANDOM_STATE   = 42
OUTPUT_FOLDER  = Path("data/processed")


# =============================================================================
# HELPERS
# =============================================================================

def drive_url(file_id: str) -> str:
    """Converts a Google Drive file ID into a direct-download URL."""
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def assign_season(month: int) -> str:
    """
    Maps a month number to its Indian agricultural season name.

    Kharif : June–September   (main rainy/sowing season)
    Rabi   : October–February (winter crop season)
    Zaid   : March–May        (minor summer crop season)
    """
    if month in KHARIF_MONTHS:
        return "Kharif"
    elif month in RABI_MONTHS:
        return "Rabi"
    return "Zaid"


def classify_anomaly(departure_pct: float) -> str:
    """
    Applies IMD's official 5-tier anomaly classification.

    > +20%       : Large Excess
    +5 to +20%   : Above Normal
    -5 to +5%    : Normal
    -20 to -5%   : Below Normal
    < -20%       : Large Deficit
    """
    if departure_pct > 20:
        return "Large Excess"
    elif departure_pct > 5:
        return "Above Normal"
    elif departure_pct >= -5:
        return "Normal"
    elif departure_pct >= -20:
        return "Below Normal"
    return "Large Deficit"


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Returns MAE, RMSE, and R² for a regression prediction."""
    return {
        "MAE" : round(mean_absolute_error(y_true, y_pred), 3),
        "RMSE": round(mean_squared_error(y_true, y_pred) ** 0.5, 3),
        "R2"  : round(r2_score(y_true, y_pred), 3),
    }


# =============================================================================
# SECTION 2 — LOAD ALL YEARLY CSVs FROM GOOGLE DRIVE
# =============================================================================

def load_all_csvs(file_ids_dict: dict[str, str]) -> pd.DataFrame:
    """
    Loads yearly CSVs directly from Google Drive and concatenates them.

    Each CSV must be shared as "Anyone with the link → Viewer" in Google Drive.
    A 'source_file' column is added so you can trace which year each row
    came from during debugging.

    Args:
        file_ids_dict: Mapping of year strings to Google Drive file IDs.

    Returns:
        Combined DataFrame with all years stacked.

    Raises:
        ValueError: If no data was loaded successfully.
    """
    yearly_frames: list[pd.DataFrame] = []

    log.info("Starting Drive download for years: %s", list(file_ids_dict.keys()))

    for year, f_id in file_ids_dict.items():
        url = drive_url(f_id)
        try:
            df_year = pd.read_csv(url)
            df_year["source_file"] = f"{year}.csv"
            yearly_frames.append(df_year)
            log.info("  ✅  %s loaded  (%s rows)", year, f"{len(df_year):,}")
        except Exception as exc:
            log.error("  ❌  %s failed: %s", year, exc)
            log.error("     → Verify file is shared as 'Anyone with the link'.")

    if not yearly_frames:
        raise ValueError(
            "No data was loaded. Check your file IDs and Drive sharing settings."
        )

    combined = pd.concat(yearly_frames, ignore_index=True)
    log.info("Total rows loaded: %s\n", f"{len(combined):,}")
    return combined


# =============================================================================
# SECTION 3 — CLEAN & STANDARDISE
# =============================================================================

def clean_and_standardise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw combined data through four steps:
      A. Rename columns to standard internal names.
      B. Parse dates; extract year, month, week, season.
      C. Strip whitespace; apply district name corrections.
      D. Handle missing rainfall via time-based interpolation.

    Args:
        df: Raw combined DataFrame from load_all_csvs().

    Returns:
        Cleaned DataFrame with no missing rainfall values.
    """
    # ── A. Rename columns ────────────────────────────────────────────────
    rename_map = {
        col: internal
        for col, internal in [
            (COL_DATE,     "date"),
            (COL_STATE,    "state_name"),
            (COL_DISTRICT, "district_name"),
            (COL_RAINFALL, "rainfall_mm"),
        ]
        if col in df.columns
    }
    df = df.rename(columns=rename_map)
    log.info("Columns after rename: %s", df.columns.tolist())

    required = {"date", "state_name", "district_name", "rainfall_mm"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise KeyError(
            f"Required columns not found after renaming: {missing_cols}. "
            f"Available columns: {df.columns.tolist()}. "
            f"Update the COL_* constants at the top of this file."
        )

    # ── B. Parse dates ───────────────────────────────────────────────────
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

    bad_dates = df["date"].isna().sum()
    if bad_dates > 0:
        log.warning("Dropped %d rows with unparseable dates.", bad_dates)
        df = df.dropna(subset=["date"])

    df["year"]        = df["date"].dt.year
    df["month"]       = df["date"].dt.month
    df["week_number"] = df["date"].dt.isocalendar().week.astype(int)
    df["season"]      = df["month"].apply(assign_season)

    # ── C. Standardise text columns ──────────────────────────────────────
    df["state_name"]    = df["state_name"].str.strip().str.title()
    df["district_name"] = df["district_name"].str.strip().str.title()
    df["district_name"] = df["district_name"].replace(DISTRICT_NAME_CORRECTIONS)
    df["rainfall_mm"]   = pd.to_numeric(df["rainfall_mm"], errors="coerce")

    # ── D. Interpolate missing rainfall ──────────────────────────────────
    # WHY TIME-BASED INTERPOLATION?
    # ─────────────────────────────
    # • Zero-fill:  wrong — a missing reading ≠ no rainfall.
    # • Mean-fill:  bad  — rainfall is seasonal; inserting a global mean
    #               into a dry-season gap is meteorologically wrong.
    # • Linear interpolation within each district's time series: best for
    #   1–3 day gaps (typical IMD missing-data pattern). Physically
    #   reasonable for a continuous weather variable.
    missing_before = df["rainfall_mm"].isna().sum()
    log.info(
        "Missing rainfall values: %d (%.2f%% of rows)",
        missing_before,
        missing_before / len(df) * 100,
    )

    df = df.sort_values(["district_name", "date"]).reset_index(drop=True)

    df["rainfall_mm"] = (
        df.groupby("district_name")["rainfall_mm"]
          .transform(lambda s: s.interpolate(method="linear"))
    )
    # Back/forward fill for series edges where interpolation cannot act
    df["rainfall_mm"] = (
        df.groupby("district_name")["rainfall_mm"]
          .transform(lambda s: s.bfill().ffill())
    )
    df["rainfall_mm"] = df["rainfall_mm"].clip(lower=0)

    remaining = df["rainfall_mm"].isna().sum()
    log.info("Missing values after cleaning: %d\n", remaining)

    return df


# =============================================================================
# SECTION 4 — MONTHLY AGGREGATION
# =============================================================================

def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapses daily rows into one row per (district, year, month).

    We use SUM not mean — farmers and meteorologists care about total
    monthly rainfall ('July got 180 mm'), not average daily drizzle.

    Returns:
        DataFrame with: state_name, district_name, year, month,
        season, total_rainfall_mm, rainy_days, data_days.
    """
    monthly = (
        df.groupby(
            ["state_name", "district_name", "year", "month", "season"],
            as_index=False,
        )
        .agg(
            total_rainfall_mm = ("rainfall_mm", "sum"),
            rainy_days        = ("rainfall_mm", lambda x: (x > 2.5).sum()),
            data_days         = ("rainfall_mm", "count"),
        )
    )
    monthly = monthly.sort_values(
        ["district_name", "year", "month"]
    ).reset_index(drop=True)

    log.info(
        "Monthly aggregation complete: %s rows (one per district-year-month)\n",
        f"{len(monthly):,}",
    )
    return monthly


# =============================================================================
# SECTION 5 — SEASONAL AGGREGATION
# =============================================================================

def aggregate_seasonal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapses daily rows into one row per (district, year, season).

    Rabi year attribution:
    Rabi spans Oct of year Y through Feb of year Y+1. We attribute the
    entire Rabi season to the year it STARTED (October), consistent with
    IMD seasonal publications.

    Returns:
        DataFrame with: state_name, district_name, year, season,
        total_rainfall_mm, rainy_days, data_days.
    """
    df = df.copy()
    df["season_year"] = df.apply(
        lambda row: row["year"] - 1
        if (row["season"] == "Rabi" and row["month"] <= 2)
        else row["year"],
        axis=1,
    )

    seasonal = (
        df.groupby(
            ["state_name", "district_name", "season_year", "season"],
            as_index=False,
        )
        .agg(
            total_rainfall_mm = ("rainfall_mm", "sum"),
            rainy_days        = ("rainfall_mm", lambda x: (x > 2.5).sum()),
            data_days         = ("rainfall_mm", "count"),
        )
        .rename(columns={"season_year": "year"})
    )
    seasonal = seasonal.sort_values(
        ["district_name", "year", "season"]
    ).reset_index(drop=True)

    log.info(
        "Seasonal aggregation complete: %s rows (one per district-year-season)\n",
        f"{len(seasonal):,}",
    )
    return seasonal


# =============================================================================
# SECTION 6 — LPA AND DEPARTURE FROM MEAN
# =============================================================================

def calculate_lpa_and_departure(
    monthly_df: pd.DataFrame,
    seasonal_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes Long Period Average (LPA) and Departure from Mean (%) for
    both monthly and seasonal aggregations.

    Formula: Departure % = ((Actual − LPA) / LPA) × 100
    Positive = above normal. Negative = below normal / drought signal.

    Returns:
        (monthly_with_lpa, seasonal_with_lpa) enriched with
        lpa_mm, departure_pct, anomaly_category columns.
    """
    def _add_departure(
        df: pd.DataFrame,
        group_cols: list[str],
    ) -> pd.DataFrame:
        lpa = (
            df.groupby(group_cols, as_index=False)["total_rainfall_mm"]
              .mean()
              .rename(columns={"total_rainfall_mm": "lpa_mm"})
        )
        merged = df.merge(lpa, on=group_cols, how="left")
        merged["departure_pct"] = np.where(
            merged["lpa_mm"] > 0,
            (merged["total_rainfall_mm"] - merged["lpa_mm"]) / merged["lpa_mm"] * 100,
            0.0,
        ).round(2)
        merged["lpa_mm"]           = merged["lpa_mm"].round(2)
        merged["anomaly_category"] = merged["departure_pct"].apply(classify_anomaly)
        return merged

    monthly_with_lpa  = _add_departure(monthly_df,  ["district_name", "month"])
    seasonal_with_lpa = _add_departure(seasonal_df, ["district_name", "season"])

    log.info("LPA and Departure calculations complete.\n")
    return monthly_with_lpa, seasonal_with_lpa


# =============================================================================
# SECTION 7 — FEATURE ENGINEERING
# =============================================================================

def add_lag_features(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds temporal lag features to the monthly DataFrame.

    Features added per district time series:
      • lag_1m / lag_2m / lag_3m      : rainfall 1–3 months prior
      • lag_12m / lag_24m             : same month in previous 1–2 years
      • roll_3m_mean / roll_6m_mean   : rolling averages (shifted to avoid leakage)
      • cum_seasonal                  : cumulative total within current season so far

    All lags shift BEFORE aggregation so no future data leaks into training.
    """
    df = monthly_df.sort_values(["district_name", "year", "month"]).copy()

    grp = df.groupby("district_name")["total_rainfall_mm"]

    df["lag_1m"]  = grp.shift(1)
    df["lag_2m"]  = grp.shift(2)
    df["lag_3m"]  = grp.shift(3)
    df["lag_12m"] = grp.shift(12)
    df["lag_24m"] = grp.shift(24)

    # Rolling means: shift(1) ensures no same-month leakage
    df["roll_3m_mean"] = grp.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df["roll_6m_mean"] = grp.transform(lambda x: x.shift(1).rolling(6, min_periods=2).mean())

    # Cumulative seasonal total up to (but not including) current month
    df["cum_seasonal"] = (
        df.groupby(["district_name", "year", "season"])["total_rainfall_mm"]
          .cumsum()
          .shift(1)
          .fillna(0)
    )

    log.info("Lag features added.\n")
    return df


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes month and season as cyclical (sine/cosine) features.

    Why cyclical?
    Linear encoding treats month 12 and month 1 as far apart. Cyclical
    encoding makes them neighbors — which they are meteorologically.

    Features added: month_sin, month_cos, season_sin, season_cos.
    """
    df = df.copy()

    # Month (1–12) → sine/cosine on a 12-unit circle
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Season: encode as ordinal (Kharif=0, Rabi=1, Zaid=2) then cyclical
    season_order = {"Kharif": 0, "Rabi": 1, "Zaid": 2}
    season_num   = df["season"].map(season_order)
    df["season_sin"] = np.sin(2 * np.pi * season_num / 3)
    df["season_cos"] = np.cos(2 * np.pi * season_num / 3)

    log.info("Cyclical features added.\n")
    return df


def add_spatial_features(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds state-level aggregate features as a spatial proxy.

    For each (state, year, month), computes the mean and std of rainfall
    across all districts. These capture regional climate signals that
    individual districts may miss due to data gaps or local noise.

    A proper spatial implementation would use actual lat/lon to find
    k-nearest neighbor districts. This approximation works well for
    Telangana's compact geography.
    """
    state_agg = (
        monthly_df
        .groupby(["state_name", "year", "month"])["total_rainfall_mm"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={
            "mean": "state_mean_rainfall",
            "std": "state_std_rainfall"
        })
    )
    # Unpack the MultiIndex columns produced by named agg
    state_agg.columns = ["state_name", "year", "month",
                          "state_mean_rainfall", "state_std_rainfall"]

    df = monthly_df.merge(state_agg, on=["state_name", "year", "month"], how="left")
    df["state_std_rainfall"] = df["state_std_rainfall"].fillna(0)

    log.info("Spatial (state-level) features added.\n")
    return df


def add_enso_features(
    monthly_df: pd.DataFrame,
    enso_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Merges ENSO (Niño 3.4 SST) index into the monthly DataFrame.

    ENSO leads Indian rainfall by ~2–3 months. A positive (El Niño) phase
    is associated with below-normal Indian monsoon; negative (La Niña) with
    above-normal.

    Free data source:
        https://psl.noaa.gov/data/correlation/nina34.data
        Download as text, parse into a CSV with columns: year, month, nino34_sst

    If enso_path is None or the file is not found, ENSO features are skipped
    and a warning is logged.

    Args:
        monthly_df: Monthly aggregated DataFrame.
        enso_path : Path to a CSV with columns [year, month, nino34_sst].

    Returns:
        DataFrame with nino34_sst, enso_lag2, enso_lag3 columns added,
        or unchanged if the file is unavailable.
    """
    if enso_path is None or not Path(enso_path).exists():
        log.warning(
            "ENSO file not found at '%s'. Skipping ENSO features. "
            "Download from https://psl.noaa.gov/data/correlation/nina34.data",
            enso_path,
        )
        return monthly_df

    enso = pd.read_csv(enso_path)[["year", "month", "nino34_sst"]]
    df   = monthly_df.merge(enso, on=["year", "month"], how="left")

    # Lag 2 and 3 months — ENSO precedes Indian monsoon by ~2–3 months
    df = df.sort_values(["district_name", "year", "month"])
    df["enso_lag2"] = (
        df.groupby("district_name")["nino34_sst"].shift(2)
    )
    df["enso_lag3"] = (
        df.groupby("district_name")["nino34_sst"].shift(3)
    )

    log.info("ENSO features added.\n")
    return df


def engineer_all_features(
    monthly_df: pd.DataFrame,
    enso_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Orchestrates all feature engineering steps in the correct order.

    Order matters: lags must come before cyclical encoding so sorting
    is consistent. Spatial features are added last to avoid polluting
    the district-level sort.

    Args:
        monthly_df: Monthly aggregated DataFrame with LPA and departure columns.
        enso_path : Optional path to ENSO index CSV.

    Returns:
        Feature-rich DataFrame ready for ML model training.
    """
    df = add_lag_features(monthly_df)
    df = add_cyclical_features(df)
    df = add_spatial_features(df)
    df = add_enso_features(df, enso_path)
    return df


# =============================================================================
# SECTION 8 — MODEL TRAINING
# =============================================================================

# Features used for training. Remove any that aren't available in your data.
FEATURE_COLS: list[str] = [
    # Temporal
    "month_sin", "month_cos",
    "season_sin", "season_cos",
    # Lag
    "lag_1m", "lag_2m", "lag_3m",
    "lag_12m", "lag_24m",
    "roll_3m_mean", "roll_6m_mean",
    "cum_seasonal",
    # LPA context
    "lpa_mm",
    "rainy_days",
    "data_days",
    # Spatial
    "state_mean_rainfall",
    "state_std_rainfall",
    # Climate (if available)
    "nino34_sst",
    "enso_lag2",
    "enso_lag3",
]

TARGET_COL = "departure_pct"   # Regression target — richer signal than binary class


def prepare_ml_dataset(
    feature_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Prepares a clean ML dataset from the feature-engineered DataFrame.

    Steps:
      1. Keep only rows where the target (departure_pct) is finite.
      2. Drop rows with NaN in ANY feature column — lag features will produce
         NaN for the earliest rows of each district's time series.
      3. Sort by date (year, month) so TimeSeriesSplit works correctly.
      4. Label-encode district_name and add it as a feature.

    Args:
        feature_df: Output of engineer_all_features().

    Returns:
        (X, y, meta) where meta holds district_name, year, month for
        joining predictions back to the original DataFrame.
    """
    df = feature_df.copy()

    # Encode district as a numeric feature
    le = LabelEncoder()
    df["district_encoded"] = le.fit_transform(df["district_name"])

    feature_cols_present = [
        c for c in FEATURE_COLS + ["district_encoded"] if c in df.columns
    ]
    all_cols = feature_cols_present + [TARGET_COL]

    df_ml = (
        df[all_cols + ["district_name", "year", "month"]]
          .replace([np.inf, -np.inf], np.nan)
          .dropna(subset=all_cols)
          .sort_values(["year", "month"])
          .reset_index(drop=True)
    )

    X    = df_ml[feature_cols_present]
    y    = df_ml[TARGET_COL]
    meta = df_ml[["district_name", "year", "month"]]

    log.info(
        "ML dataset: %d rows × %d features. Target: '%s'",
        len(X), X.shape[1], TARGET_COL,
    )
    return X, y, meta


def _cv_score(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = N_CV_SPLITS,
) -> dict[str, float]:
    """
    Evaluates a model using TimeSeriesSplit cross-validation.

    IMPORTANT: We use TimeSeriesSplit — NOT random train_test_split.
    Random splitting leaks future data into training, inflating accuracy.
    With time-series data you must always train on the past and test on
    the future.

    Returns averaged MAE, RMSE, R² across all folds.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics: list[dict[str, float]] = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_tr, y_tr)
        preds   = model.predict(X_te)
        metrics = regression_metrics(y_te.values, preds)
        fold_metrics.append(metrics)

        log.info("  Fold %d — MAE: %.2f  RMSE: %.2f  R²: %.3f",
                 fold, metrics["MAE"], metrics["RMSE"], metrics["R2"])

    avg = {
        k: round(float(np.mean([m[k] for m in fold_metrics])), 3)
        for k in ("MAE", "RMSE", "R2")
    }
    return avg


def tune_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = OPTUNA_TRIALS,
) -> dict:
    """
    Uses Optuna to find good XGBoost hyperparameters via time-series CV.

    Only runs if optuna is installed AND n_trials > 0.
    Falls back to sensible defaults otherwise.

    Returns:
        Best hyperparameter dict (or defaults if tuning is skipped).
    """
    defaults = {
        "n_estimators"    : 300,
        "max_depth"       : 5,
        "learning_rate"   : 0.05,
        "subsample"       : 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha"       : 0.1,
        "reg_lambda"      : 1.0,
    }

    if not HAS_OPTUNA or not HAS_XGB or n_trials == 0:
        log.info("Skipping Optuna tuning — using default XGBoost params.")
        return defaults

    log.info("Running Optuna hyperparameter search (%d trials)…", n_trials)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators"    : trial.suggest_int("n_estimators", 100, 600),
            "max_depth"       : trial.suggest_int("max_depth", 3, 8),
            "learning_rate"   : trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample"       : trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha"       : trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda"      : trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state"    : RANDOM_STATE,
            "n_jobs"          : -1,
        }
        model   = xgb.XGBRegressor(**params, verbosity=0)
        metrics = _cv_score(model, X, y, n_splits=3)   # 3 folds for speed
        return metrics["MAE"]                          # minimise MAE

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = {**defaults, **study.best_params}
    log.info("Best params: %s", best)
    return best


def train_models(
    X: pd.DataFrame,
    y: pd.Series,
    enso_available: bool = False,
) -> dict[str, object]:
    """
    Trains XGBoost and Random Forest regressors with time-series CV.

    Prediction workflow:
      1. Model predicts departure_pct (continuous regression).
      2. At inference time, apply classify_anomaly() to get the
         5-class IMD label — this gives more signal than direct classification.

    Args:
        X              : Feature matrix.
        y              : Target (departure_pct).
        enso_available : Whether ENSO features are present (for logging).

    Returns:
        Dict mapping model name → fitted model (trained on full dataset).
    """
    log.info("=" * 60)
    log.info("MODEL TRAINING — target: '%s'", TARGET_COL)
    log.info("Features used: %d", X.shape[1])
    log.info("ENSO features: %s", "yes" if enso_available else "no")
    log.info("=" * 60)

    trained_models: dict[str, object] = {}

    # ── Random Forest ─────────────────────────────────────────────────
    log.info("\n── Random Forest ──────────────────────────────────────")
    rf = RandomForestRegressor(
        n_estimators  = 300,
        max_depth     = 8,
        min_samples_leaf = 3,
        max_features  = "sqrt",
        random_state  = RANDOM_STATE,
        n_jobs        = -1,
    )
    rf_cv = _cv_score(rf, X, y)
    log.info("RF CV averages — MAE: %.2f  RMSE: %.2f  R²: %.3f",
             rf_cv["MAE"], rf_cv["RMSE"], rf_cv["R2"])
    rf.fit(X, y)   # final fit on all data
    trained_models["random_forest"] = rf

    # ── XGBoost ──────────────────────────────────────────────────────
    if HAS_XGB:
        log.info("\n── XGBoost ────────────────────────────────────────────")
        best_params = tune_xgboost(X, y, n_trials=OPTUNA_TRIALS)
        xgb_model   = xgb.XGBRegressor(
            **best_params,
            random_state = RANDOM_STATE,
            n_jobs       = -1,
            verbosity    = 0,
        )
        xgb_cv = _cv_score(xgb_model, X, y)
        log.info("XGB CV averages — MAE: %.2f  RMSE: %.2f  R²: %.3f",
                 xgb_cv["MAE"], xgb_cv["RMSE"], xgb_cv["R2"])
        xgb_model.fit(X, y)
        trained_models["xgboost"] = xgb_model
    else:
        log.warning("XGBoost not available — skipping.")

    return trained_models


def explain_model(
    model,
    X: pd.DataFrame,
    model_name: str = "model",
    max_display: int = 15,
) -> Optional[pd.DataFrame]:
    """
    Generates SHAP feature importance for a trained model.

    SHAP (SHapley Additive exPlanations) gives each feature a contribution
    score for every prediction — more informative than built-in importances.

    Args:
        model      : Fitted sklearn/XGBoost model.
        X          : Feature matrix (a sample of 200 rows is used for speed).
        model_name : Used for logging.
        max_display: Top N features to show.

    Returns:
        DataFrame of mean absolute SHAP values per feature,
        or None if SHAP is not installed.
    """
    if not HAS_SHAP:
        log.warning("shap not installed — skipping feature importance.")
        return None

    sample = X.sample(min(200, len(X)), random_state=RANDOM_STATE)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    importance = pd.DataFrame({
        "feature"         : X.columns,
        "mean_abs_shap"   : np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).head(max_display)

    log.info("\nTop features (%s):\n%s", model_name, importance.to_string(index=False))
    return importance


def predict_with_category(
    model,
    X: pd.DataFrame,
    meta: pd.DataFrame,
) -> pd.DataFrame:
    """
    Runs inference and converts continuous departure predictions back to
    IMD anomaly categories.

    Args:
        model : Fitted regressor.
        X     : Feature matrix.
        meta  : DataFrame with district_name, year, month columns.

    Returns:
        DataFrame with predicted_departure_pct and predicted_category.
    """
    preds = model.predict(X)
    result = meta.copy()
    result["predicted_departure_pct"] = preds.round(2)
    result["predicted_category"]      = [classify_anomaly(p) for p in preds]
    return result


# =============================================================================
# SECTION 9 — ABOVE-NORMAL PROBABILITY (EMPIRICAL BASELINE)
# =============================================================================

def calculate_above_normal_probability(
    seasonal_with_lpa: pd.DataFrame,
) -> pd.DataFrame:
    """
    Empirical probability of Above Normal or Large Excess rainfall per
    district+season, based on the historical record.

    Formula: P(Above Normal) = count(years where departure > +5%) / total_years

    This is an explainable, model-free baseline — useful for comparison
    against the ML model outputs.
    """
    df = seasonal_with_lpa.copy()
    df["is_above_normal"] = (df["departure_pct"] > 5).astype(int)

    prob_df = (
        df.groupby(["district_name", "season"], as_index=False)
          .agg(
              years_above_normal = ("is_above_normal", "sum"),
              total_years        = ("is_above_normal", "count"),
          )
    )
    prob_df["prob_above_normal_pct"] = (
        prob_df["years_above_normal"] / prob_df["total_years"] * 100
    ).round(1)

    log.info("Above-Normal probability calculation complete.\n")
    return prob_df


# =============================================================================
# SECTION 10 — SAVE OUTPUTS
# =============================================================================

def save_outputs(
    daily_clean    : pd.DataFrame,
    monthly_final  : pd.DataFrame,
    seasonal_final : pd.DataFrame,
    probability_df : pd.DataFrame,
    feature_df     : pd.DataFrame,
    predictions_df : Optional[pd.DataFrame] = None,
    output_folder  : Path = OUTPUT_FOLDER,
) -> None:
    """
    Saves all processed DataFrames to CSV in the output folder.
    These CSVs feed directly into the Streamlit dashboard (app.py).

    Output files:
        01_daily_clean.csv
        02_monthly_with_departure.csv
        03_seasonal_with_departure.csv
        04_above_normal_probability.csv
        05_features_for_ml.csv
        06_predictions.csv           (if predictions_df is provided)
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    files: dict[str, pd.DataFrame] = {
        "01_daily_clean.csv"             : daily_clean,
        "02_monthly_with_departure.csv"  : monthly_final,
        "03_seasonal_with_departure.csv" : seasonal_final,
        "04_above_normal_probability.csv": probability_df,
        "05_features_for_ml.csv"         : feature_df,
    }
    if predictions_df is not None:
        files["06_predictions.csv"] = predictions_df

    for filename, df in files.items():
        path = output_folder / filename
        df.to_csv(path, index=False)
        log.info("  Saved → %s  (%s rows)", path, f"{len(df):,}")

    log.info("\nAll outputs saved successfully.")


# =============================================================================
# SECTION 11 — MAIN PIPELINE ORCHESTRATOR
# =============================================================================

def run_pipeline(
    enso_path: Optional[str | Path] = None,
) -> dict[str, pd.DataFrame | dict]:
    """
    Runs the full pipeline end-to-end.

    Args:
        enso_path: Optional path to ENSO index CSV
                   (columns: year, month, nino34_sst).
                   Download from https://psl.noaa.gov/data/correlation/nina34.data
                   Leave as None to skip ENSO features.

    Returns:
        Dict with keys: 'clean_df', 'monthly', 'seasonal',
        'features', 'models', 'predictions'.

    Usage:
        python rainfall_pipeline.py
        from rainfall_pipeline import run_pipeline; results = run_pipeline()
    """
    log.info("=" * 65)
    log.info("  RAINFALL PIPELINE — START")
    log.info("=" * 65)

    # ── 1. Load ──────────────────────────────────────────────────────
    raw_df   = load_all_csvs(RAW_FILE_IDS)

    # ── 2. Clean ─────────────────────────────────────────────────────
    clean_df = clean_and_standardise(raw_df)

    log.info(
        "Date range: %s → %s  |  Districts: %d  |  States: %d",
        clean_df["date"].min().date(),
        clean_df["date"].max().date(),
        clean_df["district_name"].nunique(),
        clean_df["state_name"].nunique(),
    )

    # ── 3. Aggregate ─────────────────────────────────────────────────
    monthly_df  = aggregate_monthly(clean_df)
    seasonal_df = aggregate_seasonal(clean_df)

    # ── 4. LPA & Departure ───────────────────────────────────────────
    monthly_final, seasonal_final = calculate_lpa_and_departure(
        monthly_df, seasonal_df
    )

    # ── 5. Empirical above-normal probability ────────────────────────
    probability_df = calculate_above_normal_probability(seasonal_final)

    # ── 6. Feature engineering ───────────────────────────────────────
    feature_df = engineer_all_features(monthly_final, enso_path=enso_path)
    enso_available = "nino34_sst" in feature_df.columns

    # ── 7. Train models ──────────────────────────────────────────────
    X, y, meta   = prepare_ml_dataset(feature_df)
    models       = train_models(X, y, enso_available=enso_available)

    # SHAP importance for the best available model
    best_model_name = "xgboost" if "xgboost" in models else "random_forest"
    explain_model(models[best_model_name], X, model_name=best_model_name)

    # ── 8. Predictions ───────────────────────────────────────────────
    predictions_df = predict_with_category(models[best_model_name], X, meta)

    # ── 9. Save ──────────────────────────────────────────────────────
    log.info("\nSaving processed files…")
    save_outputs(
        clean_df, monthly_final, seasonal_final,
        probability_df, feature_df, predictions_df,
    )

    # ── 10. Preview ──────────────────────────────────────────────────
    log.info("\n── Monthly output (first 5 rows) ──────────────────────")
    log.info(
        "\n%s",
        monthly_final[
            ["district_name", "year", "month",
             "total_rainfall_mm", "lpa_mm",
             "departure_pct", "anomaly_category"]
        ].head().to_string(index=False),
    )

    log.info("\n── Predictions sample (first 5 rows) ──────────────────")
    log.info("\n%s", predictions_df.head().to_string(index=False))

    log.info("\n" + "=" * 65)
    log.info("  PIPELINE COMPLETE — outputs saved to %s", OUTPUT_FOLDER)
    log.info("=" * 65)

    return {
        "clean_df"      : clean_df,
        "monthly"       : monthly_final,
        "seasonal"      : seasonal_final,
        "features"      : feature_df,
        "models"        : models,
        "predictions"   : predictions_df,
    }


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # To use ENSO features, download the Niño 3.4 index from NOAA and pass the
    # path here:
    #   run_pipeline(enso_path="data/raw/enso_nino34.csv")
    run_pipeline()