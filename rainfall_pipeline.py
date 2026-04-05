"""
=============================================================================
RAINFALL ANALYSIS & PREDICTION FRAMEWORK
Step 2: Data Processing Pipeline
=============================================================================
Purpose : Load IMD/OGD daily CSVs, clean them, aggregate to monthly &
          seasonal totals per district, and compute Departure from Mean (%).
Author  : (your name)
Data    : IMD daily district-wise rainfall CSVs — one file per year
=============================================================================
"""

import pandas as pd
import numpy as np
import os


# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

# Dictionary of yearly Google Drive file IDs.
# Get the file ID from the sharing link:
#   https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing
RAW_FILE_IDS = {
    "2018": "1GJtKaG1Ht82cDrYUSyLi63lUdx_fONrT",
    "2019": "1OS_JAicP0iE-ZiMWye8m_ynfJ5Ypf2eO",
    "2020": "1nB6qe_6SqVPDx5yyCVGJcX-ydeqtwlrE",
    "2021": "1QwtMNFi-TxS3sn2SM9BteuDGSQS3L5mW",
    "2022": "1179FbAiLT1KZJAvQiBcE8T2NQPHAC7zE",
    "2023": "1OgHoFuSwd_JUdadvuxPV1Pj0QFBE4sjf",
    "2024": "1q_yHt0UeqOzo1Kvzz8MTjaP-KEKBU3hr",
}

# ── Column name constants ──────────────────────────────────────────────────
# These must match the EXACT header names in your CSV files.
# Adjust these values if your CSV uses different column headers.
COL_DATE     = "Date"           # e.g. "Date", "DATE", "date"
COL_STATE    = "State"          # e.g. "State", "STATE", "state_name"
COL_DISTRICT = "District"       # e.g. "District", "DISTRICT", "district_name"
COL_RAINFALL = "Avg_rainfall"   # e.g. "Rainfall(mm)", "RAINFALL", "rainfall_mm", "Avg_rainfall"

# ── Season month definitions ───────────────────────────────────────────────
KHARIF_MONTHS = {6, 7, 8, 9}          # June – September
RABI_MONTHS   = {10, 11, 12, 1, 2}    # October – February
# Months 3, 4, 5 (March–May) are automatically classified as "Zaid"


def drive_url(file_id: str) -> str:
    """Converts a Google Drive file ID into a direct-download URL."""
    return f"https://drive.google.com/uc?export=download&id={file_id}"


# =============================================================================
# SECTION 2 — LOAD ALL YEARLY CSVs FROM GOOGLE DRIVE
# =============================================================================

def load_all_csvs(file_ids_dict: dict) -> pd.DataFrame:
    """
    Loads yearly CSVs directly from Google Drive links and concatenates them.

    Each CSV must be shared as "Anyone with the link → Viewer" in Google Drive.
    The function adds a 'source_file' column so you can trace which year
    each row came from during debugging.

    Args:
        file_ids_dict: Dict mapping year strings to Google Drive file IDs.

    Returns:
        A single combined DataFrame with all years stacked.
    """
    yearly_frames = []

    print(f"Starting Drive download for years: {list(file_ids_dict.keys())}")

    for year, f_id in file_ids_dict.items():
        url = drive_url(f_id)
        try:
            df_year = pd.read_csv(url)
            df_year["source_file"] = f"{year}.csv"
            yearly_frames.append(df_year)
            print(f"  ✅ Successfully loaded {year}  ({len(df_year):,} rows)")
        except Exception as e:
            print(f"  ❌ Error loading {year}: {e}")
            print("     → Check that the file is shared as 'Anyone with the link' in Drive.")

    if not yearly_frames:
        raise ValueError(
            "No data was loaded. Check your file IDs and Drive sharing settings."
        )

    combined_df = pd.concat(yearly_frames, ignore_index=True)
    print(f"\nTotal rows loaded: {len(combined_df):,}\n")
    return combined_df


# =============================================================================
# SECTION 3 — CLEAN & STANDARDISE THE RAW DATA
# =============================================================================

def clean_and_standardise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs four cleaning steps:
      A. Rename columns to our standard internal names.
      B. Parse the date column into a proper datetime object.
      C. Strip whitespace from text columns and apply district name corrections.
      D. Handle missing rainfall values via time-based interpolation.

    Args:
        df: Raw combined DataFrame from load_all_csvs().

    Returns:
        Cleaned DataFrame with standardised column names and no missing values.
    """

    # ── A. Rename columns ──────────────────────────────────────────────────
    # Maps whatever your CSV uses → our standard internal names.
    # If a column name already matches the target, this is a harmless no-op.
    rename_map = {}
    if COL_DATE     in df.columns: rename_map[COL_DATE]     = "date"
    if COL_STATE    in df.columns: rename_map[COL_STATE]    = "state_name"
    if COL_DISTRICT in df.columns: rename_map[COL_DISTRICT] = "district_name"
    if COL_RAINFALL in df.columns: rename_map[COL_RAINFALL] = "rainfall_mm"

    df = df.rename(columns=rename_map)

    print(f"Columns after rename: {df.columns.tolist()}")

    # Safety check — confirm required columns exist after rename
    for required_col in ["date", "state_name", "district_name", "rainfall_mm"]:
        assert required_col in df.columns, (
            f"Required column '{required_col}' not found after renaming. "
            f"Available columns: {df.columns.tolist()}\n"
            f"Update the COL_* constants at the top of this file to match "
            f"your CSV headers."
        )

    # ── B. Parse dates ─────────────────────────────────────────────────────
    # dayfirst=True handles Indian DD-MM-YYYY convention.
    # errors="coerce" turns unparseable dates into NaT instead of crashing.
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

    # Drop rows where the date couldn't be parsed at all
    bad_dates = df["date"].isna().sum()
    if bad_dates > 0:
        print(f"  ⚠️  Dropped {bad_dates} rows with unparseable dates.")
        df = df.dropna(subset=["date"])

    # Extract useful time components as separate columns
    df["year"]        = df["date"].dt.year
    df["month"]       = df["date"].dt.month
    df["week_number"] = df["date"].dt.isocalendar().week.astype(int)

    # Assign the agricultural season label to each row
    df["season"] = df["month"].apply(assign_season)

    # ── C. Strip whitespace & standardise district names ───────────────────
    # IMD CSVs often have trailing spaces like "Warangal " — .str.strip() fixes this.
    df["state_name"]    = df["state_name"].str.strip().str.title()
    df["district_name"] = df["district_name"].str.strip().str.title()

    # Known name variations in IMD data → canonical names used in this pipeline
    DISTRICT_NAME_CORRECTIONS = {
        "Jagtial"            : "Jagitial",
        "Jangoan"            : "Jangaon",
        "Kumuram Bheem"      : "Kumuram Bheem Asifabad",
        "Medchal-Malkajgiri" : "Medchal Malkajgiri",
        "Rangareddy"         : "Ranga Reddy",
        "Ranjanna Sircilla"  : "Rajanna Sircilla",
        "Warangal Rural"     : "Warangal (Rural)",
        "Warangal Urban"     : "Warangal (Urban)",
    }
    df["district_name"] = df["district_name"].replace(DISTRICT_NAME_CORRECTIONS)

    # Ensure rainfall_mm is numeric (coerce any stray strings to NaN)
    df["rainfall_mm"] = pd.to_numeric(df["rainfall_mm"], errors="coerce")

    # ── D. Handle missing rainfall values ──────────────────────────────────
    missing_count = df["rainfall_mm"].isna().sum()
    print(f"Missing rainfall values found: {missing_count} "
          f"({missing_count / len(df) * 100:.2f}% of rows)")

    # WHY TIME-BASED INTERPOLATION (not zero-fill, not mean-fill)?
    # ─────────────────────────────────────────────────────────────
    # Option A — Fill with 0  : WRONG. A missing reading ≠ no rainfall.
    #            Zeros would artificially depress your LPA and anomaly scores.
    # Option B — Fill with mean: BAD. Rainfall is seasonal — a June mean
    #            inserted into a December gap is meteorologically nonsensical.
    # Option C — Time interpolation: BEST for this data. It assumes the
    #            missing day's rainfall sits between the day before and after —
    #            physically reasonable for a continuous weather variable.
    #            Works well when gaps are 1–3 days (typical for IMD data).
    #
    # We sort by district + date first so interpolation stays within each
    # district's own time series and doesn't bleed across districts.

    df = df.sort_values(["district_name", "date"]).reset_index(drop=True)

    df["rainfall_mm"] = (
        df.groupby("district_name")["rainfall_mm"]
          .transform(lambda s: s.interpolate(method="linear"))
    )

    # For values at the very start/end of a district's series (interpolation
    # cannot fill edges), back-fill then forward-fill as a last resort.
    df["rainfall_mm"] = (
        df.groupby("district_name")["rainfall_mm"]
          .transform(lambda s: s.bfill().ffill())
    )

    # Rainfall can never be negative — clip any tiny interpolation artefacts
    df["rainfall_mm"] = df["rainfall_mm"].clip(lower=0)

    remaining_missing = df["rainfall_mm"].isna().sum()
    print(f"Missing values remaining after cleaning: {remaining_missing}\n")

    return df


def assign_season(month: int) -> str:
    """
    Maps a month number to its Indian agricultural season name.

    Kharif : June–September  (main rainy/sowing season)
    Rabi   : October–February (winter crop season)
    Zaid   : March–May        (minor summer crop season)
    """
    if month in KHARIF_MONTHS:
        return "Kharif"
    elif month in RABI_MONTHS:
        return "Rabi"
    else:
        return "Zaid"


# =============================================================================
# SECTION 4 — AGGREGATE TO MONTHLY TOTALS PER DISTRICT
# =============================================================================

def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapses daily rows into one row per (district, year, month).

    Why SUM and not mean?
    Farmers and meteorologists care about TOTAL monthly rainfall, not the
    average daily drizzle. "July got 180 mm" is meaningful;
    "July averaged 5.8 mm/day" is not.

    Returns:
        DataFrame with columns: state_name, district_name, year, month,
        total_rainfall_mm, rainy_days, data_days.
    """
    monthly = (
        df.groupby(
            ["state_name", "district_name", "year", "month"],
            as_index=False
        )
        .agg(
            total_rainfall_mm=("rainfall_mm", "sum"),
            rainy_days=(
                "rainfall_mm",
                lambda x: (x > 2.5).sum()   # IMD threshold: >2.5 mm = rainy day
            ),
            data_days=("rainfall_mm", "count"),
        )
    )

    monthly = monthly.sort_values(
        ["district_name", "year", "month"]
    ).reset_index(drop=True)

    print(f"Monthly aggregation complete: {len(monthly):,} rows "
          f"(one per district-year-month)\n")
    return monthly


# =============================================================================
# SECTION 5 — AGGREGATE TO SEASONAL TOTALS PER DISTRICT
# =============================================================================

def aggregate_seasonal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapses daily rows into one row per (district, year, season).

    NOTE on Rabi year attribution:
    Rabi spans October of year Y through February of year Y+1.
    We attribute the ENTIRE Rabi season to the year it STARTED (October).
    Example: Oct 2020 – Feb 2021 is labelled as Rabi 2020.
    This is consistent with how IMD publishes seasonal totals.

    Returns:
        DataFrame with columns: state_name, district_name, year, season,
        total_rainfall_mm, rainy_days, data_days.
    """
    df = df.copy()

    # For Rabi rows in Jan/Feb, attribute them to the previous year (season start)
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
            total_rainfall_mm=("rainfall_mm", "sum"),
            rainy_days=("rainfall_mm", lambda x: (x > 2.5).sum()),
            data_days=("rainfall_mm", "count"),
        )
    )

    seasonal = seasonal.rename(columns={"season_year": "year"})
    seasonal = seasonal.sort_values(
        ["district_name", "year", "season"]
    ).reset_index(drop=True)

    print(f"Seasonal aggregation complete: {len(seasonal):,} rows "
          f"(one per district-year-season)\n")
    return seasonal


# =============================================================================
# SECTION 6 — CALCULATE LONG PERIOD AVERAGE (LPA) AND DEPARTURE FROM MEAN
# =============================================================================

def calculate_lpa_and_departure(
    monthly_df: pd.DataFrame, seasonal_df: pd.DataFrame
):
    """
    Computes:
      1. LPA — the multi-year average rainfall for each district+month
               and each district+season combination.
      2. Departure from Mean (%) — how far each year's actual rainfall
         deviates from that LPA, expressed as a percentage.

    Formula:
        Departure % = ((Actual − LPA) / LPA) × 100

    Positive = above-normal rainfall.
    Negative = below-normal / drought signal.

    Returns:
        (monthly_with_lpa, seasonal_with_lpa) — both DataFrames enriched
        with lpa_mm, departure_pct, and anomaly_category columns.
    """

    # ── Monthly LPA & Departure ────────────────────────────────────────────

    monthly_lpa = (
        monthly_df
        .groupby(["district_name", "month"], as_index=False)["total_rainfall_mm"]
        .mean()
        .rename(columns={"total_rainfall_mm": "lpa_mm"})
    )

    monthly_with_lpa = monthly_df.merge(
        monthly_lpa, on=["district_name", "month"], how="left"
    )

    # np.where guards against division by zero for historically dry months
    monthly_with_lpa["departure_pct"] = np.where(
        monthly_with_lpa["lpa_mm"] > 0,
        (
            (monthly_with_lpa["total_rainfall_mm"] - monthly_with_lpa["lpa_mm"])
            / monthly_with_lpa["lpa_mm"]
        ) * 100,
        0,  # departure = 0 when LPA is 0 (completely dry month historically)
    )

    monthly_with_lpa["departure_pct"] = monthly_with_lpa["departure_pct"].round(2)
    monthly_with_lpa["lpa_mm"]        = monthly_with_lpa["lpa_mm"].round(2)
    monthly_with_lpa["anomaly_category"] = monthly_with_lpa["departure_pct"].apply(
        classify_anomaly
    )

    # ── Seasonal LPA & Departure ───────────────────────────────────────────

    seasonal_lpa = (
        seasonal_df
        .groupby(["district_name", "season"], as_index=False)["total_rainfall_mm"]
        .mean()
        .rename(columns={"total_rainfall_mm": "lpa_mm"})
    )

    seasonal_with_lpa = seasonal_df.merge(
        seasonal_lpa, on=["district_name", "season"], how="left"
    )

    seasonal_with_lpa["departure_pct"] = np.where(
        seasonal_with_lpa["lpa_mm"] > 0,
        (
            (seasonal_with_lpa["total_rainfall_mm"] - seasonal_with_lpa["lpa_mm"])
            / seasonal_with_lpa["lpa_mm"]
        ) * 100,
        0,
    )

    seasonal_with_lpa["departure_pct"] = seasonal_with_lpa["departure_pct"].round(2)
    seasonal_with_lpa["lpa_mm"]        = seasonal_with_lpa["lpa_mm"].round(2)
    seasonal_with_lpa["anomaly_category"] = seasonal_with_lpa["departure_pct"].apply(
        classify_anomaly
    )

    print("LPA and Departure calculations complete.\n")
    return monthly_with_lpa, seasonal_with_lpa


def classify_anomaly(departure_pct: float) -> str:
    """
    Applies IMD's official 5-tier anomaly classification.

    Thresholds source: India Meteorological Department seasonal outlook docs.
        > +20%  : Large Excess
        +5 to +20% : Above Normal
        -5 to +5%  : Normal
        -20 to -5% : Below Normal
        < -20%  : Large Deficit
    """
    if departure_pct > 20:
        return "Large Excess"
    elif departure_pct > 5:
        return "Above Normal"
    elif departure_pct >= -5:
        return "Normal"
    elif departure_pct >= -20:
        return "Below Normal"
    else:
        return "Large Deficit"


# =============================================================================
# SECTION 7 — PROBABILITY OF ABOVE-NORMAL (PREDICTION METRIC)
# =============================================================================

def calculate_above_normal_probability(
    seasonal_with_lpa: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each district+season, computes the empirical probability of receiving
    Above Normal or Large Excess rainfall — i.e., how many of the available
    years cleared the +5% departure threshold.

    Formula:
        P(Above Normal) = count(years where departure > +5%) / total_years

    This is the headline farmer-facing prediction number. It is explainable,
    data-grounded, and requires no ML model.

    Returns:
        DataFrame with columns: district_name, season,
        years_above_normal, total_years, prob_above_normal_pct.
    """
    seasonal_with_lpa = seasonal_with_lpa.copy()

    # Flag each row: 1 if above-normal, 0 otherwise
    seasonal_with_lpa["is_above_normal"] = (
        seasonal_with_lpa["departure_pct"] > 5
    ).astype(int)

    prob_df = (
        seasonal_with_lpa
        .groupby(["district_name", "season"], as_index=False)
        .agg(
            years_above_normal=("is_above_normal", "sum"),
            total_years=("is_above_normal", "count"),
        )
    )

    prob_df["prob_above_normal_pct"] = (
        (prob_df["years_above_normal"] / prob_df["total_years"]) * 100
    ).round(1)

    print("Above-Normal probability calculation complete.\n")
    return prob_df


# =============================================================================
# SECTION 8 — SAVE ALL OUTPUT FILES
# =============================================================================

def save_outputs(
    daily_clean: pd.DataFrame,
    monthly_final: pd.DataFrame,
    seasonal_final: pd.DataFrame,
    probability_df: pd.DataFrame,
    output_folder: str = "data/processed/",
) -> None:
    """
    Saves all four processed DataFrames to CSV files in the output folder.
    These CSVs feed directly into the Streamlit dashboard (app.py).

    Output files:
        01_daily_clean.csv            — cleaned daily data
        02_monthly_with_departure.csv — monthly totals + LPA + departure
        03_seasonal_with_departure.csv— seasonal totals + LPA + departure
        04_above_normal_probability.csv — empirical above-normal probability
    """
    os.makedirs(output_folder, exist_ok=True)

    paths = {
        "01_daily_clean.csv"            : daily_clean,
        "02_monthly_with_departure.csv" : monthly_final,
        "03_seasonal_with_departure.csv": seasonal_final,
        "04_above_normal_probability.csv": probability_df,
    }

    for filename, df in paths.items():
        full_path = os.path.join(output_folder, filename)
        df.to_csv(full_path, index=False)
        print(f"  Saved → {full_path}  ({len(df):,} rows)")

    print("\nAll outputs saved successfully.")


# =============================================================================
# SECTION 9 — MAIN PIPELINE ORCHESTRATOR
# =============================================================================

def run_pipeline():
    """
    Master function — runs the full pipeline end to end.

    Usage:
        python rainfall_pipeline.py          # run from terminal
        from rainfall_pipeline import run_pipeline; run_pipeline()  # from notebook
    """

    print("=" * 65)
    print("  RAINFALL PIPELINE — START")
    print("=" * 65 + "\n")

    # Step 1: Load and merge all yearly CSVs from Google Drive
    raw_df = load_all_csvs(RAW_FILE_IDS)

    # Step 2: Clean, standardise, and fill missing values
    clean_df = clean_and_standardise(raw_df)

    # Sanity check — confirm date range and coverage
    print(f"Date range in data : {clean_df['date'].min().date()} → "
          f"{clean_df['date'].max().date()}")
    print(f"Unique districts   : {clean_df['district_name'].nunique()}")
    print(f"Unique states      : {clean_df['state_name'].nunique()}\n")

    # Step 3: Aggregate to monthly and seasonal totals
    monthly_df  = aggregate_monthly(clean_df)
    seasonal_df = aggregate_seasonal(clean_df)

    # Step 4: Calculate LPA and Departure from Mean
    monthly_final, seasonal_final = calculate_lpa_and_departure(
        monthly_df, seasonal_df
    )

    # Step 5: Calculate Above-Normal probability (primary prediction metric)
    probability_df = calculate_above_normal_probability(seasonal_final)

    # Step 6: Save all outputs to data/processed/
    print("\nSaving processed files …")
    save_outputs(clean_df, monthly_final, seasonal_final, probability_df)

    # Step 7: Preview the key output tables
    print("\n── Monthly Output (first 5 rows) ──────────────────────────")
    print(
        monthly_final[
            ["district_name", "year", "month",
             "total_rainfall_mm", "lpa_mm",
             "departure_pct", "anomaly_category"]
        ].head().to_string(index=False)
    )

    print("\n── Seasonal Output (first 5 rows) ─────────────────────────")
    print(
        seasonal_final[
            ["district_name", "year", "season",
             "total_rainfall_mm", "lpa_mm",
             "departure_pct", "anomaly_category"]
        ].head().to_string(index=False)
    )

    print("\n── Above-Normal Probability (first 5 rows) ─────────────────")
    print(probability_df.head().to_string(index=False))

    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETE — outputs saved to data/processed/")
    print("=" * 65)

    return clean_df


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_pipeline()