# Rainfall Analysis and Prediction Framework
> District-wise monsoon intelligence for Kharif crop planning in Telangana

A Streamlit-based data science application that analyzes IMD historical rainfall data, detects anomalies, and predicts monsoon season outcomes using a soft-voting ensemble of LightGBM, XGBoost, and Random Forest models.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Datasets](#datasets)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [Running the Application](#running-the-application)
- [ML Pipeline](#ml-pipeline)
- [Model Performance](#model-performance)
- [Dashboard Screenshots](#dashboard-screenshots)
- [Project Structure](#project-structure)
- [Future Enhancements](#future-enhancements)

---

## Project Overview

Telangana's farmers and agricultural planners face increasing uncertainty due to erratic monsoon patterns driven by climate change. Conventional forecasting tools either lack granularity at the district level or are too technical for practical on-ground use.

This project addresses that gap by building an end-to-end rainfall analytics and prediction system that:
- Ingests and preprocesses daily district-wise IMD rainfall data (2018–2024)
- Supplements it with NASA SMAP soil moisture and ERA5 temperature data via Google Earth Engine
- Classifies each district-season as **Deficit**, **Normal**, or **Above Normal** relative to the Long Period Average (LPA)
- Presents all outputs through an interactive Streamlit dashboard accessible to farmers and agricultural officers

---

## Features

- **District-wise Rainfall Outlook** — historical above-normal probability, ML prediction for current season, and last season's LPA departure
- **Kharif Season Trend Charts** — year-wise rainfall vs LPA, color-coded for surplus/deficit years
- **Monthly Departure Heatmap** — month × year grid showing % departure from LPA (2018–2024)
- **District Comparison** — horizontal bar chart ranking all 33 Telangana districts by above-normal probability
- **Farmer's Advice** — plain-language guidance (5 bullet points) based on predicted rainfall category and risk level
- **Raw Data View** — underlying processed dataset for transparency

---

## System Architecture

The system is a modular ML pipeline with six stages:

```
IMD Rainfall Data (CSV)
        │
        ▼
┌─────────────────────┐
│   DATA COLLECTION   │  ← IMD Open Gov Portal + Google Earth Engine
│  (Python, Pandas,   │    (NASA SMAP soil moisture, ERA5 temperature)
│   GEE API)          │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   PREPROCESSING     │  ← District name correction, 3-step imputation,
│  (NumPy, Pandas,    │    seasonal segmentation (Kharif/Rabi/Zaid)
│   Datetime)         │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ STATISTICAL ANALYSIS│  ← LPA computation, Departure %, lag features,
│  Feature Engineering│    ENSO state, 7-day moving average, dry streaks
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  ANOMALY DETECTION  │  ← IMD classification: Deficit / Normal / Above Normal
│  (Label Generation) │    Empirical above-normal probability per district-season
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  ML TRAINING &      │  ← Soft-voting ensemble: LightGBM + XGBoost + Random Forest
│  PREDICTION         │    LOOCV evaluation, SMOTE for class balance
│  (sklearn Pipeline) │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  VISUALIZATION &    │  ← Streamlit dashboard, Matplotlib/Plotly charts
│  INSIGHT DELIVERY   │    District-level forecasts + Farmer's Advice
└─────────────────────┘
```

---

## Datasets

| Source | Data | Access |
|--------|------|--------|
| **IMD (India Meteorological Department)** | Daily district-wise rainfall (mm), 2018–2024 | [data.gov.in](https://data.gov.in) |
| **NASA SMAP** (`NASA/USDA/HSL/SMAP10KM_soil_moisture`) | 10 km grid sub-surface soil moisture | Google Earth Engine |
| **ECMWF ERA5-Land** (`ECMWF/ERA5_LAND/HOURLY`) | 2-meter air temperature (hourly reanalysis) | Google Earth Engine |
| **NOAA** | Historical ENSO state codes (La Niña=0, Neutral=1, El Niño=2) | Incorporated as static feature |

> **Note:** GEE satellite features are averaged across each district's boundary using `reduceRegion()` and cached locally to avoid repeated API calls.

---

## Technologies Used

| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| Data Processing | Pandas, NumPy |
| ML Models | Scikit-learn, LightGBM, XGBoost |
| Geospatial | Google Earth Engine (GEE) Python API |
| Visualization | Streamlit, Matplotlib, Plotly |
| Dev Environment | VS Code, Jupyter Notebook |

---

## Setup Instructions

### Prerequisites

- Python 3.10 or higher
- A Google Earth Engine account (for satellite feature extraction)
- Git

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/rainfall-analysis-telangana.git
cd rainfall-analysis-telangana
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Authenticate with Google Earth Engine

```bash
earthengine authenticate
```

Follow the browser prompt to log in with your GEE-registered Google account. This is required only once per machine.

### 5. Download IMD Data

Place the IMD district-wise daily rainfall CSV files inside the `data/raw/` folder. The expected format is:

```
State, District, Date, Year, Month, Avg Rainfall, Agency Name
```

Publicly available from: [https://data.gov.in](https://data.gov.in)

---

## Running the Application

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501` in your browser.

**Usage:**
1. Select a **State** (Telangana) and **District** from the sidebar
2. Enter the **June Rainfall (mm)** recorded so far for the current season
3. View the rainfall outlook, trend charts, heatmap, and district comparison across the tabs

---

## ML Pipeline

### Feature Engineering

Each district-year observation is described by 9 features:

| Feature | Description |
|---------|-------------|
| `june_rainfall` | Total June rainfall (mm) — primary seasonal signal |
| `lpa_departure_pct` | % departure from Long Period Average |
| `rolling_7day_avg` | 7-day moving average of daily rainfall |
| `cumulative_june_rain` | Cumulative June rainfall up to date |
| `lag_1day` | Previous day's rainfall |
| `lag_7day` | Rainfall 7 days ago |
| `dry_streak_days` | Days since last rainfall ≥ 2.5 mm |
| `enso_state` | ENSO phase (0=La Niña, 1=Neutral, 2=El Niño) |
| `district_encoded` | Label-encoded district identifier |

### Anomaly Classification (Labels)

Rainfall is categorized using IMD-standard departure thresholds:

```
departure_pct = ((actual - LPA) / LPA) * 100

Above Normal  →  departure_pct >= +20%
Normal        →  -20% <= departure_pct <= +20%
Deficit       →  departure_pct < -20%
```

### Ensemble Model

A `VotingClassifier` with `voting='soft'` combines three models:

```
Final Class = argmax( (P_LightGBM + P_XGBoost + P_RandomForest) / 3 )
```

- **Random Forest** — 500 trees, `max_features='sqrt'`, `max_depth=5`; reduces variance, robust to outliers
- **XGBoost** — sequential boosting, `learning_rate=0.05`, `subsample=0.8`; corrects misclassifications
- **LightGBM** — leaf-wise growth strategy; fast training, strong on structured tabular data

The full preprocessing workflow (StandardScaler → PCA → VotingClassifier) runs inside a single `sklearn.Pipeline` to prevent data leakage during cross-validation.

### Validation Strategy

Leave-One-Out Cross Validation (LOOCV) is used because the dataset has only 198 Kharif district-year observations. SMOTE is applied inside each training fold to handle class imbalance without leaking test information.

---

## Model Performance

| Model | Accuracy | Performance | Stability |
|-------|----------|-------------|-----------|
| Random Forest | 67.67% | Moderate | Medium |
| XGBoost | 56.62% | Low | Low |
| LightGBM | 56.62% | Moderate | Medium |
| **Voting Ensemble** | **94–95%** | **High** | **High** |

**Ensemble Classification Report (LOOCV):**

```
              precision  recall  f1-score  support
Above Normal     0.94    0.95    0.94      154
Normal           0.98    0.98    0.98      254
Deficit          0.84    0.82    0.83       74

accuracy                          0.95      482
macro avg        0.92    0.92    0.92      482
weighted avg     0.95    0.95    0.95      482
```

---

## Dashboard Screenshots

| View | Description |
|------|-------------|
| **Rainfall Outlook** | Historical probability, ML prediction, last season departure |
| **Rainfall Charts** | Kharif season totals vs LPA (bar chart, color-coded) |
| **Monthly Heatmap** | Month × year % departure grid (red=deficit, green=surplus) |
| **District Comparison** | All 33 Telangana districts ranked by above-normal probability |

*(Screenshots available in `/docs/screenshots/`)*

---

## Project Structure

```
rainfall-analysis-telangana/
│
├── app.py                   # Main Streamlit dashboard
├── requirements.txt         # Python dependencies
├── README.md
│
├── data/
│   ├── raw/                 # IMD CSV files (not tracked in Git)
│   └── processed/           # Preprocessed & feature-engineered data
│
├── src/
│   ├── preprocessing.py     # Data cleaning, imputation, seasonal segmentation
│   ├── feature_engineering.py  # LPA, departure %, lag features, ENSO encoding
│   ├── anomaly_detection.py    # IMD classification, empirical probability
│   ├── model.py             # Ensemble training, LOOCV, SMOTE, evaluation
│   └── gee_extraction.py    # Google Earth Engine soil moisture & temperature fetch
│
├── models/
│   └── ensemble_pipeline.pkl   # Saved trained model (joblib)
│
└── docs/
    └── screenshots/         # Dashboard screenshots
```

---

## Future Enhancements

- Integrate live weather feed APIs for real-time prediction updates
- Extend the framework to other Indian states with region-specific climate parameters
- Add LSTM/GRU-based time series models for multi-step seasonal forecasting
- Include additional variables: humidity, wind speed, sea surface temperature
- Build a mobile-friendly version for direct farmer access

---

## References

Key references from the literature survey (see project report Chapter 2 for full citations):

- Trend analysis of Telangana rainfall using Mann-Kendall and Sen's slope estimators
- Ensemble ML approaches for Indian Summer Monsoon Rainfall prediction
- Rainfall prediction using ANN, SVM, Random Forest — comparative studies
- Spatio-temporal analysis using GIS and Kriging for Indian meteorological divisions

---

## Acknowledgements

- **Indian Meteorological Department (IMD)** — primary rainfall dataset
- **NASA / USDA** — SMAP soil moisture data
- **ECMWF** — ERA5-Land reanalysis temperature data
- **Google Earth Engine** — cloud-based geospatial processing platform
