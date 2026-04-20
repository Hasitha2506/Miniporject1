"""
=============================================================================
RAINFALL ANALYSIS & PREDICTION FRAMEWORK
Step 2: Data Processing Pipeline + ML Model Training — v5 (merged)
=============================================================================
Purpose : Load IMD/OGD daily CSVs, clean them, aggregate to monthly &
          seasonal totals per district, compute Departure from Mean (%),
          engineer lag/climate/spatial/GEE features, and train XGBoost +
          Random Forest models with proper time-series cross-validation.

Authors : Hasi (ML pipeline, TimeSeriesSplit, lag/cyclical/spatial features,
                XGBoost + RF with Optuna, SHAP)
          Teammate (GEE soil/temperature enrichment, ENSO lookup table,
                    SPI computation, gee_gateway integration)

Merge notes (v5):
  FROM HASI   — all ML code: FEATURE_COLS, prepare_ml_dataset,
                _cv_score (TimeSeriesSplit), tune_xgboost (Optuna),
                train_models, explain_model, predict_with_category,
                add_lag_features, add_cyclical_features, add_spatial_features,
                save_model_artifacts, load_and_predict
  FROM TEAMMATE — ENSO_LOOKUP, add_enso_context, calculate_spi,
                  enrich_with_gee_features, _add_empty_gee_columns,
                  SPI passthrough in aggregate_monthly / aggregate_seasonal
=============================================================================
"""

from __future__ import annotations

import joblib
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

RAW_FILE_IDS: dict[str, str] = {
    "2018": "1GJtKaG1Ht82cDrYUSyLi63lUdx_fONrT",
    "2019": "1OS_JAicP0iE-ZiMWye8m_ynfJ5Ypf2eO",
    "2020": "1nB6qe_6SqVPDx5yyCVGJcX-ydeqtwlrE",
    "2021": "1QwtMNFi-TxS3sn2SM9BteuDGSQS3L5mW",
    "2022": "1179FbAiLT1KZJAvQiBcE8T2NQPHAC7zE",
    "2023": "1OgHoFuSwd_JUdadvuxPV1Pj0QFBE4sjf",
    "2024": "1q_yHt0UeqOzo1Kvzz8MTjaP-KEKBU3hr",
}

COL_DATE     = "Date"
COL_STATE    = "State"
COL_DISTRICT = "District"
COL_RAINFALL = "Avg_rainfall"

KHARIF_MONTHS: frozenset[int] = frozenset({6, 7, 8, 9})
RABI_MONTHS:   frozenset[int] = frozenset({10, 11, 12, 1, 2})

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

# ── ENSO lookup (from teammate) ───────────────────────────────────────────
# Source: NOAA / IMD seasonal reports
# enso_code: 0 = La Niña, 1 = Neutral, 2 = El Niño
ENSO_LOOKUP: dict[int, dict] = {
    2010: {"enso_state": "La Niña",       "enso_code": 0},
    2011: {"enso_state": "La Niña",       "enso_code": 0},
    2012: {"enso_state": "Neutral",       "enso_code": 1},
    2013: {"enso_state": "Neutral",       "enso_code": 1},
    2014: {"enso_state": "Neutral",       "enso_code": 1},
    2015: {"enso_state": "Strong El Niño","enso_code": 2},
    2016: {"enso_state": "El Niño",       "enso_code": 2},
    2017: {"enso_state": "Neutral",       "enso_code": 1},
    2018: {"enso_state": "Neutral",       "enso_code": 1},
    2019: {"enso_state": "El Niño",       "enso_code": 2},
    2020: {"enso_state": "La Niña",       "enso_code": 0},
    2021: {"enso_state": "La Niña",       "enso_code": 0},
    2022: {"enso_state": "La Niña",       "enso_code": 0},
    2023: {"enso_state": "El Niño",       "enso_code": 2},
    2024: {"enso_state": "Neutral",       "enso_code": 1},
    2025: {"enso_state": "La Niña",       "enso_code": 0},
}

# ── ML configuration ──────────────────────────────────────────────────────
N_CV_SPLITS   = 5
OPTUNA_TRIALS = 30       # Set to 0 to skip hyperparameter tuning
RANDOM_STATE  = 42
OUTPUT_FOLDER = Path("data/processed")
MODELS_FOLDER = Path("models")


# =============================================================================
# HELPERS
# =============================================================================

def drive_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def assign_season(month: int) -> str:
    if month in KHARIF_MONTHS:
        return "Kharif"
    elif month in RABI_MONTHS:
        return "Rabi"
    return "Zaid"


def classify_anomaly(departure_pct: float) -> str:
    if departure_pct > 20:     return "Large Excess"
    elif departure_pct > 5:    return "Above Normal"
    elif departure_pct >= -5:  return "Normal"
    elif departure_pct >= -20: return "Below Normal"
    return "Large Deficit"


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE" : round(mean_absolute_error(y_true, y_pred), 3),
        "RMSE": round(mean_squared_error(y_true, y_pred) ** 0.5, 3),
        "R2"  : round(r2_score(y_true, y_pred), 3),
    }


# =============================================================================
# SECTION 2 — LOAD ALL YEARLY CSVs
# =============================================================================

def load_all_csvs(file_ids_dict: dict[str, str]) -> pd.DataFrame:
    """Loads yearly CSVs from Google Drive and concatenates them."""
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
        raise ValueError("No data loaded. Check file IDs and Drive sharing settings.")

    combined = pd.concat(yearly_frames, ignore_index=True)
    log.info("Total rows loaded: %s\n", f"{len(combined):,}")
    return combined


# =============================================================================
# SECTION 3 — CLEAN & STANDARDISE
# =============================================================================

def clean_and_standardise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw combined data:
      A. Rename columns to standard internal names.
      B. Parse dates; extract year, month, week, season.
      C. Strip whitespace; apply district name corrections.
      D. Handle missing rainfall via time-based interpolation.
    """
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
            f"Available: {df.columns.tolist()}. Update COL_* constants."
        )

    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    bad_dates = df["date"].isna().sum()
    if bad_dates > 0:
        log.warning("Dropped %d rows with unparseable dates.", bad_dates)
        df = df.dropna(subset=["date"])

    df["year"]        = df["date"].dt.year
    df["month"]       = df["date"].dt.month
    df["week_number"] = df["date"].dt.isocalendar().week.astype(int)
    df["season"]      = df["month"].apply(assign_season)

    df["state_name"]    = df["state_name"].str.strip().str.title()
    df["district_name"] = df["district_name"].str.strip().str.title()
    df["district_name"] = df["district_name"].replace(DISTRICT_NAME_CORRECTIONS)
    df["rainfall_mm"]   = pd.to_numeric(df["rainfall_mm"], errors="coerce")

    missing_before = df["rainfall_mm"].isna().sum()
    log.info("Missing rainfall values: %d (%.2f%%)", missing_before,
             missing_before / len(df) * 100)

    df = df.sort_values(["district_name", "date"]).reset_index(drop=True)
    df["rainfall_mm"] = (
        df.groupby("district_name")["rainfall_mm"]
          .transform(lambda s: s.interpolate(method="linear"))
    )
    df["rainfall_mm"] = (
        df.groupby("district_name")["rainfall_mm"]
          .transform(lambda s: s.bfill().ffill())
    )
    df["rainfall_mm"] = df["rainfall_mm"].clip(lower=0)

    log.info("Missing values after cleaning: %d\n", df["rainfall_mm"].isna().sum())
    return df


# =============================================================================
# SECTION 4 — SPI (from teammate)
# =============================================================================

def calculate_spi(df: pd.DataFrame, window_days: int = 30) -> pd.DataFrame:
    """
    Computes a rolling Standardised Precipitation Index (SPI) on daily data.

    SPI = (rolling_sum - rolling_mean) / rolling_std

    A positive SPI means wetter than the district's historical norm over the
    same window; negative means drier. This gives the model a drought/surplus
    signal that departure_pct alone doesn't capture at daily resolution.

    Args:
        df          : Cleaned daily DataFrame.
        window_days : Rolling window in days (default 30 = SPI-30).

    Returns:
        DataFrame with new 'spi_30d' column (or spi_{window_days}d).
    """
    col_name = f"spi_{window_days}d"
    df = df.sort_values(["district_name", "date"]).copy()

    rolling = (
        df.groupby("district_name")["rainfall_mm"]
          .transform(lambda s: s.rolling(window_days, min_periods=max(1, window_days // 2)).sum())
    )
    roll_mean = (
        df.groupby("district_name")["rainfall_mm"]
          .transform(lambda s: s.rolling(window_days, min_periods=max(1, window_days // 2))
                                .sum().expanding().mean())
    )
    roll_std = (
        df.groupby("district_name")["rainfall_mm"]
          .transform(lambda s: s.rolling(window_days, min_periods=max(1, window_days // 2))
                                .sum().expanding().std())
    )

    df[col_name] = np.where(
        roll_std > 0,
        (rolling - roll_mean) / roll_std,
        0.0,
    )
    log.info("SPI-%d computed.\n", window_days)
    return df


# =============================================================================
# SECTION 5 — ENSO CONTEXT (from teammate)
# =============================================================================

def add_enso_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges ENSO state from the hard-coded ENSO_LOOKUP table into any
    DataFrame that has a 'year' column (works on daily, monthly, seasonal).

    Columns added:
      enso_state : human-readable label (e.g. "La Niña", "Strong El Niño")
      enso_code  : integer (0 = La Niña, 1 = Neutral, 2 = El Niño)

    WHY ENSO MATTERS FOR KHARIF PREDICTION:
    The Indian Summer Monsoon is strongly anti-correlated with El Niño.
    During El Niño years, the Walker Circulation weakens, reducing moisture
    transport to South Asia. Adding ENSO gives the model the large-scale
    climate context that explains why some years are systematically wetter
    or drier — independent of district-level patterns.
    """
    enso_df = (
        pd.DataFrame.from_dict(ENSO_LOOKUP, orient="index")
          .reset_index()
          .rename(columns={"index": "year"})
    )
    enso_df["year"] = enso_df["year"].astype(int)

    df = df.merge(enso_df, on="year", how="left")
    df["enso_state"] = df["enso_state"].fillna("Neutral")
    df["enso_code"]  = df["enso_code"].fillna(1).astype(int)

    log.info(
        "ENSO context added. Distribution:\n%s\n",
        df.groupby("enso_state")["year"].nunique().rename("years").to_string(),
    )
    return df


# =============================================================================
# SECTION 6 — GEE FEATURE ENRICHMENT (from teammate)
# =============================================================================

def enrich_with_gee_features(
    seasonal_df: pd.DataFrame,
    district_geometries: dict,
    use_gee: bool = False,
) -> pd.DataFrame:
    """
    Merges GEE-derived soil moisture and temperature features into the
    seasonal DataFrame. One row per (district, year) is added.

    Args:
        seasonal_df         : Seasonal aggregates DataFrame.
        district_geometries : Dict of {district_name: geojson_dict}.
                              Pre-computed from GeoJSON shapefile via
                              run_once_build_geometries.py.
        use_gee             : Set True only when GEE is authenticated and
                              you want to re-fetch. False uses cached values
                              from data/external/gee_features.csv if present.

    Returns:
        seasonal_df with added columns:
          susm_may_mean         — pre-monsoon sub-surface soil moisture (mm)
          susm_may_max          — peak pre-monsoon soil moisture (mm)
          temp_june_mean        — mean June temperature (°C)
          temp_june_stress_days — days > 35°C in June
    """
    GEE_CACHE_PATH = "data/external/gee_features.csv"

    # ── Use cached GEE data if available ─────────────────────────────────
    if not use_gee and os.path.exists(GEE_CACHE_PATH):
        log.info("Loading cached GEE features from %s", GEE_CACHE_PATH)
        gee_df = pd.read_csv(GEE_CACHE_PATH)
        seasonal_df = seasonal_df.merge(
            gee_df, on=["district_name", "year"], how="left"
        )
        log.info(
            "GEE features merged. Coverage: %d districts\n",
            gee_df["district_name"].nunique(),
        )
        return seasonal_df

    # ── Fetch from GEE ────────────────────────────────────────────────────
    if use_gee:
        try:
            from gee_gateway import initialize_gee, fetch_district_climate_features
        except ImportError:
            log.warning("gee_gateway.py not found. Skipping GEE enrichment.")
            return _add_empty_gee_columns(seasonal_df)

        if not initialize_gee():
            log.warning("GEE not available. Skipping enrichment.")
            return _add_empty_gee_columns(seasonal_df)

        log.info("Fetching GEE features for all districts and years...")
        gee_rows = []

        for district in seasonal_df["district_name"].unique():
            if district not in district_geometries:
                log.warning("  No geometry for %s — skipping.", district)
                continue

            geom  = district_geometries[district]
            years = sorted(
                seasonal_df[seasonal_df["district_name"] == district]["year"].unique()
            )

            for year in years:
                try:
                    row = fetch_district_climate_features(district, geom, year)
                    gee_rows.append(row)
                    log.info("  ✅  %s %d", district, year)
                except Exception as exc:
                    log.error("  ❌  %s %d: %s", district, year, exc)
                    gee_rows.append({
                        "district_name"        : district,
                        "year"                 : year,
                        "susm_may_mean"        : np.nan,
                        "susm_may_max"         : np.nan,
                        "temp_june_mean"       : np.nan,
                        "temp_june_stress_days": np.nan,
                    })

        if gee_rows:
            gee_df = pd.DataFrame(gee_rows)
            os.makedirs("data/external", exist_ok=True)
            gee_df.to_csv(GEE_CACHE_PATH, index=False)
            log.info("GEE features saved → %s", GEE_CACHE_PATH)
            seasonal_df = seasonal_df.merge(
                gee_df, on=["district_name", "year"], how="left"
            )
    else:
        log.warning(
            "GEE not enabled and no cache found. "
            "Run with use_gee=True after authenticating GEE."
        )
        seasonal_df = _add_empty_gee_columns(seasonal_df)

    return seasonal_df


def _add_empty_gee_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Adds GEE feature columns as NaN when GEE is unavailable."""
    for col in ["susm_may_mean", "susm_may_max",
                "temp_june_mean", "temp_june_stress_days"]:
        if col not in df.columns:
            df[col] = np.nan
    return df


# =============================================================================
# SECTION 7 — MONTHLY AGGREGATION
# =============================================================================

def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapses daily rows into one row per (district, year, month).
    Passes through SPI column if computed.
    Uses SUM — farmers care about total monthly rainfall, not daily averages.
    """
    agg_dict: dict = {
        "total_rainfall_mm" : ("rainfall_mm", "sum"),
        "rainy_days"        : ("rainfall_mm", lambda x: (x > 2.5).sum()),
        "data_days"         : ("rainfall_mm", "count"),
    }
    if "spi_30d" in df.columns:
        agg_dict["mean_spi_30d"] = ("spi_30d", "mean")

    monthly = (
        df.groupby(
            ["state_name", "district_name", "year", "month", "season"],
            as_index=False,
        )
        .agg(**agg_dict)
    )
    monthly = monthly.sort_values(
        ["district_name", "year", "month"]
    ).reset_index(drop=True)

    log.info(
        "Monthly aggregation complete: %s rows\n", f"{len(monthly):,}"
    )
    return monthly


# =============================================================================
# SECTION 8 — SEASONAL AGGREGATION
# =============================================================================

def aggregate_seasonal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapses daily rows into one row per (district, year, season).
    Rabi seasons attributed to start year (October). Passes through SPI.
    """
    df = df.copy()
    df["season_year"] = df.apply(
        lambda row: row["year"] - 1
        if (row["season"] == "Rabi" and row["month"] <= 2)
        else row["year"],
        axis=1,
    )

    agg_dict: dict = {
        "total_rainfall_mm" : ("rainfall_mm", "sum"),
        "rainy_days"        : ("rainfall_mm", lambda x: (x > 2.5).sum()),
        "data_days"         : ("rainfall_mm", "count"),
    }
    if "spi_30d" in df.columns:
        agg_dict["mean_spi_30d"] = ("spi_30d", "mean")

    seasonal = (
        df.groupby(
            ["state_name", "district_name", "season_year", "season"],
            as_index=False,
        )
        .agg(**agg_dict)
        .rename(columns={"season_year": "year"})
    )
    seasonal = seasonal.sort_values(
        ["district_name", "year", "season"]
    ).reset_index(drop=True)

    log.info(
        "Seasonal aggregation complete: %s rows\n", f"{len(seasonal):,}"
    )
    return seasonal


# =============================================================================
# SECTION 9 — LPA AND DEPARTURE FROM MEAN
# =============================================================================

def calculate_lpa_and_departure(
    monthly_df: pd.DataFrame,
    seasonal_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes LPA and Departure% for both monthly and seasonal aggregations.
    Formula: Departure % = ((Actual − LPA) / LPA) × 100
    """
    def _add_departure(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
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
# SECTION 10 — FEATURE ENGINEERING (Hasi)
# =============================================================================

def add_lag_features(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds temporal lag features per district time series:
      lag_1m / lag_2m / lag_3m      : rainfall 1–3 months prior
      lag_12m / lag_24m             : same month in previous 1–2 years
      roll_3m_mean / roll_6m_mean   : rolling averages (shift to avoid leakage)
      cum_seasonal                  : cumulative total within current season
    """
    df = monthly_df.sort_values(["district_name", "year", "month"]).copy()
    grp = df.groupby("district_name")["total_rainfall_mm"]

    df["lag_1m"]  = grp.shift(1)
    df["lag_2m"]  = grp.shift(2)
    df["lag_3m"]  = grp.shift(3)
    df["lag_12m"] = grp.shift(12)
    df["lag_24m"] = grp.shift(24)

    df["roll_3m_mean"] = grp.transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    df["roll_6m_mean"] = grp.transform(
        lambda x: x.shift(1).rolling(6, min_periods=2).mean()
    )
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
    Encodes month and season as sine/cosine pairs so the model understands
    that January and December are adjacent, not opposite ends of a scale.
    """
    df = df.copy()
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    season_order = {"Kharif": 0, "Rabi": 1, "Zaid": 2}
    season_num   = df["season"].map(season_order)
    df["season_sin"] = np.sin(2 * np.pi * season_num / 3)
    df["season_cos"] = np.cos(2 * np.pi * season_num / 3)

    log.info("Cyclical features added.\n")
    return df


def add_spatial_features(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds state-level mean and std as a spatial proxy.
    Captures regional climate signals that individual districts may miss.
    """
    state_agg = (
        monthly_df
        .groupby(["state_name", "year", "month"])["total_rainfall_mm"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={
            "mean": "state_mean_rainfall",
            "std" : "state_std_rainfall",
        })
    )

    df = monthly_df.merge(
        state_agg, on=["state_name", "year", "month"], how="left"
    )
    df["state_std_rainfall"] = df["state_std_rainfall"].fillna(0)

    log.info("Spatial (state-level) features added.\n")
    return df


def engineer_all_features(
    monthly_df: pd.DataFrame,
    enso_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Orchestrates all feature engineering in the correct order.

    Order: lags → cyclical → spatial → ENSO context (from LOOKUP table).
    The enso_path arg is kept for backward compatibility but ENSO_LOOKUP
    is now used by default; pass a CSV path to override with Niño 3.4 SST.
    """
    df = add_lag_features(monthly_df)
    df = add_cyclical_features(df)
    df = add_spatial_features(df)
    df = add_enso_context(df)          # uses ENSO_LOOKUP — no file needed

    # If a Niño 3.4 SST CSV is provided, also merge the continuous SST value
    # (gives the model a numeric ENSO intensity, not just the categorical code)
    if enso_path is not None and Path(enso_path).exists():
        enso_sst = pd.read_csv(enso_path)[["year", "month", "nino34_sst"]]
        df = df.merge(enso_sst, on=["year", "month"], how="left")
        df = df.sort_values(["district_name", "year", "month"])
        df["enso_lag2"] = df.groupby("district_name")["nino34_sst"].shift(2)
        df["enso_lag3"] = df.groupby("district_name")["nino34_sst"].shift(3)
        log.info("Niño 3.4 SST features added from %s\n", enso_path)

    return df


# =============================================================================
# SECTION 11 — MODEL TRAINING (Hasi)
# =============================================================================

# Feature list — add GEE columns here once gee_features.csv is available.
# prepare_ml_dataset filters to only columns present in the DataFrame,
# so adding a column name here is safe before the data is ready.
FEATURE_COLS: list[str] = [
    # Temporal cyclical
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
    # ENSO (from lookup table — always available)
    "enso_code",
    # ENSO continuous SST (optional — only if enso_path provided)
    "nino34_sst",
    "enso_lag2",
    "enso_lag3",
    # SPI (only if calculate_spi was run)
    "mean_spi_30d",
    # GEE soil/temperature (only if gee_features.csv is cached)
    "susm_may_mean",
    "susm_may_max",
    "temp_june_mean",
    "temp_june_stress_days",
]

TARGET_COL = "departure_pct"   # Regression → classify_anomaly() at inference


def prepare_ml_dataset(
    feature_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Builds a clean ML-ready dataset from the feature-engineered DataFrame.

    Steps:
      1. Label-encode district_name as a numeric feature.
      2. Filter FEATURE_COLS to only those present in the DataFrame.
         This makes the feature list forward-compatible — GEE / ENSO SST
         columns can be listed but are silently skipped if not yet fetched.
      3. Drop NaN rows — lag features produce NaN for earliest rows.
      4. Sort by (year, month) so TimeSeriesSplit stays time-ordered.

    Returns:
        (X, y, meta) where meta carries district_name, year, month for
        joining predictions back to the source DataFrame.
    """
    df = feature_df.copy()

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
    Time-series cross-validation using TimeSeriesSplit.

    CRITICAL: Do NOT use random train_test_split on time-series data.
    Random splitting leaks future data into training, inflating accuracy.
    TimeSeriesSplit always trains on the past and tests on the future.
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

    return {
        k: round(float(np.mean([m[k] for m in fold_metrics])), 3)
        for k in ("MAE", "RMSE", "R2")
    }


def tune_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = OPTUNA_TRIALS,
) -> dict:
    """
    Optuna hyperparameter search for XGBoost via time-series CV.
    Falls back to sensible defaults if Optuna is not installed or n_trials=0.
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
        log.info("Skipping Optuna — using default XGBoost params.")
        return defaults

    log.info("Running Optuna search (%d trials)…", n_trials)

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
        metrics = _cv_score(model, X, y, n_splits=3)
        return metrics["MAE"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = {**defaults, **study.best_params}
    log.info("Best params: %s", best)
    return best


def train_models(
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, object]:
    """
    Trains XGBoost and Random Forest regressors with TimeSeriesSplit CV.

    Regression workflow:
      model predicts departure_pct (continuous)
      → classify_anomaly() converts to IMD 5-tier label at inference time
    This gives far more signal than direct classification on 7 years of data.

    Returns:
        Dict mapping model name → fitted model (trained on full dataset).
    """
    gee_cols    = {"susm_may_mean", "susm_may_max",
                   "temp_june_mean", "temp_june_stress_days"}
    spi_cols    = {"mean_spi_30d"}
    enso_cols   = {"enso_code", "nino34_sst", "enso_lag2", "enso_lag3"}
    present     = set(X.columns)

    log.info("=" * 60)
    log.info("MODEL TRAINING — target: '%s'", TARGET_COL)
    log.info("Features: %d total", X.shape[1])
    log.info("  GEE features present : %s", bool(gee_cols & present))
    log.info("  SPI features present : %s", bool(spi_cols & present))
    log.info("  ENSO features present: %s", bool(enso_cols & present))
    log.info("=" * 60)

    trained: dict[str, object] = {}

    # ── Random Forest ─────────────────────────────────────────────────
    log.info("\n── Random Forest ──────────────────────────────────────")
    rf = RandomForestRegressor(
        n_estimators     = 300,
        max_depth        = 8,
        min_samples_leaf = 3,
        max_features     = "sqrt",
        random_state     = RANDOM_STATE,
        n_jobs           = -1,
    )
    rf_cv = _cv_score(rf, X, y)
    log.info("RF CV — MAE: %.2f  RMSE: %.2f  R²: %.3f",
             rf_cv["MAE"], rf_cv["RMSE"], rf_cv["R2"])
    rf.fit(X, y)
    trained["random_forest"] = rf

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
        log.info("XGB CV — MAE: %.2f  RMSE: %.2f  R²: %.3f",
                 xgb_cv["MAE"], xgb_cv["RMSE"], xgb_cv["R2"])
        xgb_model.fit(X, y)
        trained["xgboost"] = xgb_model
    else:
        log.warning("XGBoost not available — skipping.")

    return trained


def explain_model(
    model,
    X: pd.DataFrame,
    model_name: str = "model",
    max_display: int = 15,
) -> Optional[pd.DataFrame]:
    """
    SHAP feature importance. Returns a DataFrame sorted by mean |SHAP|.
    Returns None if shap is not installed.
    """
    if not HAS_SHAP:
        log.warning("shap not installed — skipping feature importance.")
        return None

    sample      = X.sample(min(200, len(X)), random_state=RANDOM_STATE)
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    importance = (
        pd.DataFrame({
            "feature"      : X.columns,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        })
        .sort_values("mean_abs_shap", ascending=False)
        .head(max_display)
    )
    log.info("\nTop features (%s):\n%s", model_name, importance.to_string(index=False))
    return importance


def predict_with_category(
    model,
    X: pd.DataFrame,
    meta: pd.DataFrame,
) -> pd.DataFrame:
    """
    Runs inference and converts departure_pct predictions to IMD categories.
    Enforces the exact feature columns the model was trained on.
    """
    X_aligned = X[list(model.feature_names_in_)] \
        if hasattr(model, "feature_names_in_") else X

    preds  = model.predict(X_aligned)
    result = meta.copy()
    result["predicted_departure_pct"] = preds.round(2)
    result["predicted_category"]      = [classify_anomaly(p) for p in preds]
    return result


# =============================================================================
# SECTION 12 — MODEL PERSISTENCE
# =============================================================================

def save_model_artifacts(
    model,
    feature_cols: list[str],
    model_name: str,
    output_folder: Path = MODELS_FOLDER,
) -> None:
    """
    Saves the fitted model and the exact feature list used at training time.
    Loading via load_and_predict() then guarantees feature alignment,
    which prevents the '9 features vs 10' mismatch error at inference.
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(model,        output_folder / f"{model_name}.pkl")
    joblib.dump(feature_cols, output_folder / f"{model_name}_features.pkl")
    log.info("Saved model + feature list → %s/", output_folder)


def load_and_predict(
    X_new: pd.DataFrame,
    model_name: str,
    output_folder: Path = MODELS_FOLDER,
) -> np.ndarray:
    """
    Loads a saved model and aligns X_new to the training feature list.

    Any column present at training but missing in X_new is filled with NaN
    (logged as a warning). Extra columns in X_new are silently dropped.
    This is the safe inference path — use this instead of model.predict(X)
    directly.
    """
    model        = joblib.load(output_folder / f"{model_name}.pkl")
    feature_cols = joblib.load(output_folder / f"{model_name}_features.pkl")

    for col in feature_cols:
        if col not in X_new.columns:
            log.warning(
                "Feature '%s' missing at inference — filling with NaN. "
                "This may degrade accuracy.", col
            )
            X_new[col] = np.nan

    return model.predict(X_new[feature_cols])


# =============================================================================
# SECTION 13 — ABOVE-NORMAL PROBABILITY (EMPIRICAL BASELINE)
# =============================================================================

def calculate_above_normal_probability(
    seasonal_with_lpa: pd.DataFrame,
) -> pd.DataFrame:
    """
    Empirical P(Above-Normal) per district-season.
    Formula: count(years where departure > +5%) / total_years
    Useful as a model-free baseline to compare against ML outputs.
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
# SECTION 14 — SAVE OUTPUTS
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
    Saves all processed DataFrames to CSV. Feeds the Streamlit dashboard.

    Files:
        01_daily_clean.csv
        02_monthly_with_departure.csv   ← upload to Drive for dashboard
        03_seasonal_with_departure.csv  ← upload to Drive for dashboard
        04_above_normal_probability.csv ← upload to Drive for dashboard
        05_features_for_ml.csv
        06_predictions.csv              (if predictions_df provided)
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

    log.info("\nAll outputs saved. Upload 02/03/04 to Drive to refresh dashboard.")


# =============================================================================
# SECTION 15 — MAIN PIPELINE ORCHESTRATOR
# =============================================================================

def run_pipeline(
    enso_path          : Optional[str | Path] = None,
    use_gee            : bool = False,
    district_geometries: Optional[dict] = None,
) -> dict:
    """
    Runs the full pipeline end-to-end.

    Args:
        enso_path           : Optional path to Niño 3.4 SST CSV
                              (columns: year, month, nino34_sst).
                              Download: https://psl.noaa.gov/data/correlation/nina34.data
                              ENSO_LOOKUP is always used regardless; this
                              adds the continuous SST value on top.
        use_gee             : Set True to re-fetch GEE features.
                              Requires gee_gateway.py + GEE authentication.
        district_geometries : Dict {district_name: geojson_dict}.
                              Required only when use_gee=True.
                              Load from run_once_build_geometries.py output.

    Returns:
        Dict with keys: clean_df, monthly, seasonal, features,
                        models, predictions.

    Usage:
        python rainfall_pipeline.py
        from rainfall_pipeline import run_pipeline
        results = run_pipeline(use_gee=True, district_geometries=geoms)
    """
    log.info("=" * 65)
    log.info("  RAINFALL PIPELINE v5 — START")
    log.info("=" * 65)

    # 1. Load
    raw_df   = load_all_csvs(RAW_FILE_IDS)

    # 2. Clean
    clean_df = clean_and_standardise(raw_df)
    log.info(
        "Date range: %s → %s  |  Districts: %d  |  States: %d",
        clean_df["date"].min().date(), clean_df["date"].max().date(),
        clean_df["district_name"].nunique(), clean_df["state_name"].nunique(),
    )

    # 3. SPI on daily data
    clean_df = calculate_spi(clean_df, window_days=30)

    # 4. ENSO on daily data
    clean_df = add_enso_context(clean_df)

    # 5. Aggregate
    monthly_df  = aggregate_monthly(clean_df)
    seasonal_df = aggregate_seasonal(clean_df)

    # 6. LPA & Departure
    monthly_final, seasonal_final = calculate_lpa_and_departure(
        monthly_df, seasonal_df
    )

    # 7. Add ENSO to seasonal (needed by train_model.py)
    seasonal_final = add_enso_context(seasonal_final)

    # 8. GEE enrichment (soil moisture + temperature)
    seasonal_final = enrich_with_gee_features(
        seasonal_final, district_geometries or {}, use_gee=use_gee
    )

    # 9. Empirical above-normal probability
    probability_df = calculate_above_normal_probability(seasonal_final)

    # 10. Feature engineering (lags, cyclical, spatial, ENSO, optional SST)
    feature_df = engineer_all_features(monthly_final, enso_path=enso_path)

    # 11. Train models
    X, y, meta = prepare_ml_dataset(feature_df)
    models     = train_models(X, y)

    # 12. SHAP importance for best model
    best_name = "xgboost" if "xgboost" in models else "random_forest"
    explain_model(models[best_name], X, model_name=best_name)

    # 13. Save model artifacts (fixes the 9-vs-10 feature mismatch at inference)
    save_model_artifacts(models[best_name], list(X.columns), best_name)

    # 14. Predictions
    predictions_df = predict_with_category(models[best_name], X, meta)

    # 15. Save CSVs
    log.info("\nSaving processed files…")
    save_outputs(
        clean_df, monthly_final, seasonal_final,
        probability_df, feature_df, predictions_df,
    )

    # 16. Preview
    log.info("\n── Seasonal output (first 3 rows) ─────────────────────")
    preview_cols = ["district_name", "year", "season",
                    "total_rainfall_mm", "lpa_mm",
                    "departure_pct", "anomaly_category", "enso_state"]
    if "mean_spi_30d" in seasonal_final.columns:
        preview_cols.append("mean_spi_30d")
    log.info("\n%s", seasonal_final[preview_cols].head(3).to_string(index=False))

    log.info("\n── Predictions sample (first 3 rows) ───────────────────")
    log.info("\n%s", predictions_df.head(3).to_string(index=False))

    log.info("\n" + "=" * 65)
    log.info("  PIPELINE COMPLETE — outputs in %s/", OUTPUT_FOLDER)
    log.info("  Upload 02/03/04 CSVs to Drive to refresh dashboard.")
    log.info("=" * 65)

    return {
        "clean_df"   : clean_df,
        "monthly"    : monthly_final,
        "seasonal"   : seasonal_final,
        "features"   : feature_df,
        "models"     : models,
        "predictions": predictions_df,
    }


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Basic run (no GEE, no ENSO SST file):
    run_pipeline()

    # With Niño 3.4 SST (download from NOAA, parse to year/month/nino34_sst CSV):
    #   run_pipeline(enso_path="data/raw/enso_nino34.csv")

    # With GEE (requires gee_gateway.py + authenticated GEE session):
    #   import json
    #   with open("data/external/district_geometries.json") as f:
    #       geoms = json.load(f)
    #   run_pipeline(use_gee=True, district_geometries=geoms)