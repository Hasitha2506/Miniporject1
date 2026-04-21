"""
=============================================================================
RAINFALL ANALYSIS & PREDICTION FRAMEWORK
Step 4: Streamlit Dashboard — The Farmer's Interface  (TABBED LAYOUT)
=============================================================================
Purpose : Interactive dashboard for farmers and agricultural officers.
          Loads pre-processed CSVs and the trained ML model to display:
          - District rainfall trends vs LPA
          - Probability of Above-Normal monsoon (empirical + ML)
          - Plain-English farmer advice
Run with: streamlit run app.py
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import os
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# SECTION 1 — PAGE CONFIGURATION
# Must be the FIRST streamlit command in the script.
# =============================================================================

st.set_page_config(
    page_title="Rainfall Analysis Dashboard",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# SECTION 2 — CUSTOM CSS STYLING
# =============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #2c7be5;
        margin-bottom: 0;
        letter-spacing: -0.5px;
    }
    .main-subtitle {
        font-size: 1.25rem;
        color: #999;
        margin-top: 4px;
        font-weight: 300;
    }

    /* ── Metric cards ──────────────────────────────── */
    .metric-card {
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        background: white;
        border-radius: 12px;
        padding: 24px;
        min-height: 140px;
        border: 1px solid #e8e8e8;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        text-align: center;
        margin-bottom: 16px;
    }
    .metric-label {
        font-size: 0.78rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #888;
        margin-bottom: 8px;
    }
    .metric-value-green  { font-size: 3rem; font-weight: 600; color: #2d7a4f; line-height: 1; }
    .metric-value-blue   { font-size: 3rem; font-weight: 600; color: #1a5fa8; line-height: 1; }
    .metric-value-orange { font-size: 3rem; font-weight: 600; color: #c07a00; line-height: 1; }
    .metric-value-red    { font-size: 3rem; font-weight: 600; color: #b03030; line-height: 1; }
    .metric-value-gray   { font-size: 3rem; font-weight: 600; color: #555;    line-height: 1; }
    .metric-sublabel {
        font-size: 0.85rem;
        color: #999;
        margin-top: 6px;
    }

    /* ── Advice cards ──────────────────────────────── */
    .advice-card-green  { background:#f0faf4; border-left:4px solid #2d7a4f; border-radius:8px; padding:20px; }
    .advice-card-blue   { background:#f0f6ff; border-left:4px solid #1a5fa8; border-radius:8px; padding:20px; }
    .advice-card-orange { background:#fff8e6; border-left:4px solid #c07a00; border-radius:8px; padding:20px; }
    .advice-card-red    { background:#fff0f0; border-left:4px solid #b03030; border-radius:8px; padding:20px; }
    .advice-title { font-size:1rem; font-weight:600; margin-bottom:8px; }
    .advice-body  { font-size:0.92rem; line-height:1.7; color:#333; }

    /* ── Section headers ───────────────────────────── */
    .section-header {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #aaa;
        border-bottom: 1px solid #eee;
        padding-bottom: 8px;
        margin-bottom: 16px;
        margin-top: 8px;
    }

    /* ── Banners ───────────────────────────────────── */
    .info-banner {
        background: #f8f9ff;
        border: 1px solid #dde3ff;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.85rem;
        color: #445;
        margin-bottom: 16px;
    }
    .warn-banner {
        background: #fffbf0;
        border: 1px solid #ffe082;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.85rem;
        color: #664d00;
        margin-bottom: 16px;
    }

    /* ── Tab styling ───────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid #e8e8e8;
        padding-bottom: 0;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        padding: 10px 20px;
        border-radius: 6px 6px 0 0;
        color: #888;
    }
    .stTabs [aria-selected="true"] {
        color: #2c7be5 !important;
        background: #f0f6ff !important;
        border-bottom: 2px solid #2c7be5 !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 24px;
    }

    /* ── Chart container card ──────────────────────── */
    .chart-card {
        background: white;
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #e8e8e8;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        margin-bottom: 24px;
    }
    .chart-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #aaa;
        margin-bottom: 12px;
    }

    #MainMenu {visibility: hidden;}
    footer    {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SECTION 3 — DATA & MODEL LOADERS
# =============================================================================

@st.cache_data
def load_processed_data():
    FILE_IDS = {
        "monthly"     : "1AY2n7HBfu0BsrLlDL80iflWqlqLSYiMH",
        "seasonal"    : "1rnbhP44S_gah-v7L6BRJKBZInwLSLjDG",
        "probability" : "1wHgLiXOuvqLmpzaPSoj73rWpTPmHYgp2",
    }

    def drive_url(file_id: str) -> str:
        return f"https://drive.google.com/uc?export=download&id={file_id}"

    data = {}
    for key, file_id in FILE_IDS.items():
        try:
            data[key] = pd.read_csv(drive_url(file_id))
        except Exception as e:
            st.error(
                f"Could not load '{key}' from Google Drive.\n\n"
                f"Check that the file ID is correct and the file is shared "
                f"publicly (Anyone with the link → Viewer).\n\nError: {e}"
            )
            st.stop()

    return data


@st.cache_resource
def load_model():
    model_path = "rainfall_ensemble_model.pkl"
    if not os.path.exists(model_path):
        return None
    import joblib
    return joblib.load(model_path)


# =============================================================================
# SECTION 4 — HELPER FUNCTIONS
# =============================================================================

def get_colour_for_probability(probability: float) -> str:
    if probability >= 65:   return "green"
    elif probability >= 50: return "blue"
    elif probability >= 35: return "orange"
    else:                   return "red"


def get_advice(probability: float, district: str) -> dict:
    if probability >= 65:
        return {
            "title"  : "✅ Good monsoon likely — plan for full sowing",
            "points" : [
                f"Strong signal of above-normal rainfall for {district} this Kharif season.",
                "Proceed with your full sowing plan for water-intensive crops (paddy, sugarcane).",
                "Ensure drainage channels and bunds are cleared before June to handle surplus water.",
                "Stock up on seeds and fertilisers early — demand will be high.",
                "Consider crop insurance nonetheless; even good monsoons can have dry spells.",
            ]
        }
    elif probability >= 50:
        return {
            "title"  : "🔵 Moderate outlook — proceed with caution",
            "points" : [
                f"Moderate confidence of above-normal rainfall for {district}.",
                "Sow primary crops as planned but reserve 20% of your area as a buffer.",
                "Diversify: mix a high-value water-intensive crop with one drought-tolerant variety.",
                "Monitor weekly district rainfall updates — adjust irrigation schedules accordingly.",
                "Consult your local Krishi Vigyan Kendra (KVK) for variety recommendations.",
            ]
        }
    elif probability >= 35:
        return {
            "title"  : "🟡 Near-normal expected — standard precautions advised",
            "points" : [
                f"Season likely to be near the historical average for {district}.",
                "Follow your standard sowing schedule for the district.",
                "Keep drought-tolerant variety seeds (e.g. millets, sorghum) as backup.",
                "Check soil moisture before each irrigation cycle to avoid over-watering.",
                "Review your crop insurance policy before June 30th.",
            ]
        }
    else:
        return {
            "title"  : "🔴 Below-normal risk — take protective action",
            "points" : [
                f"Higher than usual risk of below-normal monsoon for {district}.",
                "Strongly consider drought-resistant crop varieties this season.",
                "Prioritise water conservation: repair farm ponds and check-dams now.",
                "Avoid over-investment in water-intensive crops like paddy this cycle.",
                "Contact your district agriculture officer and register for drought relief schemes.",
                "Visit the nearest Krishi Vigyan Kendra before finalising sowing plans.",
            ]
        }


def make_ml_prediction(model_dict, district, state, june_mm, seasonal_df, monthly_df):
    if model_dict is None:
        return None, None

    try:
        from ensemble_model import predict as ensemble_predict
        from sklearn.preprocessing import LabelEncoder

        june_rows = monthly_df[
            (monthly_df["district_name"] == district) &
            (monthly_df["month"] == 6)
        ].sort_values("year")

        if june_rows.empty:
            june_rows = monthly_df[monthly_df["district_name"] == district].sort_values(["year", "month"])
        if june_rows.empty:
            return None, None

        base_row = june_rows.tail(1).copy()

        input_data = {
            'year': int(base_row['year'].iloc[0]) if 'year' in base_row.columns else 2024,
            'month': 6,
            'total_rainfall_mm': june_mm,
            'rainy_days': int(base_row['rainy_days'].iloc[0]) if 'rainy_days' in base_row.columns else 15,
            'data_days': int(base_row['data_days'].iloc[0]) if 'data_days' in base_row.columns else 30,
            'mean_spi_30d': float(base_row['mean_spi_30d'].iloc[0]) if 'mean_spi_30d' in base_row.columns else 0.0,
            'lpa_mm': float(base_row['lpa_mm'].iloc[0]) if 'lpa_mm' in base_row.columns else june_mm,
            'month_sin': np.sin(2 * np.pi * 6 / 12),
            'month_cos': np.cos(2 * np.pi * 6 / 12),
        }

        le_d = LabelEncoder()
        le_s = LabelEncoder()
        le_d.fit(monthly_df["district_name"].astype(str).unique())
        le_s.fit(monthly_df["state_name"].astype(str).unique())

        input_data['district_name_enc'] = le_d.transform([district])[0]
        input_data['state_name_enc'] = le_s.transform([state])[0] if state in le_s.classes_ else 0

        june_lpa = input_data['lpa_mm']
        departure_pct = ((june_mm - june_lpa) / june_lpa * 100) if june_lpa > 0 else 0.0

        input_df = pd.DataFrame([input_data])
        input_df['departure_pct'] = departure_pct
        input_df['state_name'] = state
        input_df['district_name'] = district

        labels, probs = ensemble_predict(model_dict, input_df)

        pred_label = labels[0]
        confidence = float(np.max(probs[0]) * 100)
        return confidence, pred_label

    except Exception as e:
        st.warning(f"ML prediction failed: {e}")
        return None, None


def plot_rainfall_trend(seasonal_df: pd.DataFrame, district: str) -> plt.Figure:
    dist_data = seasonal_df[
        (seasonal_df["district_name"] == district) &
        (seasonal_df["season"] == "Kharif")
    ].sort_values("year").copy()

    if dist_data.empty:
        return None

    lpa      = dist_data["lpa_mm"].iloc[0]
    years    = dist_data["year"].astype(str).tolist()
    rainfall = dist_data["total_rainfall_mm"].tolist()
    bar_colours = ["#2d7a4f" if r >= lpa else "#c0392b" for r in rainfall]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")

    bars = ax.bar(years, rainfall, color=bar_colours, width=0.55,
                  zorder=3, edgecolor="white", linewidth=0.8)

    ax.axhline(y=lpa, color="#1a1a2e", linewidth=1.5,
               linestyle="--", zorder=4, label=f"LPA: {lpa:.0f} mm")

    for bar, val in zip(bars, rainfall):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 8,
            f"{val:.0f}",
            ha="center", va="bottom",
            fontsize=8.5, color="#333", fontfamily="monospace"
        )

    ax.set_xlabel("Year", fontsize=10, color="#555", labelpad=8)
    ax.set_ylabel("Total Rainfall (mm)", fontsize=10, color="#555", labelpad=8)
    ax.set_title(
        f"Kharif Season Rainfall — {district}",
        fontsize=13, fontweight="600", color="#1a1a2e", pad=14
    )
    ax.tick_params(colors="#666", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#ddd")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, color="#ddd", zorder=0)
    ax.set_axisbelow(True)

    above_patch = mpatches.Patch(color="#2d7a4f", label="Above LPA")
    below_patch = mpatches.Patch(color="#c0392b", label="Below LPA")
    lpa_line    = plt.Line2D([0], [0], color="#1a1a2e", linewidth=1.5,
                              linestyle="--", label=f"LPA ({lpa:.0f} mm)")
    ax.legend(handles=[above_patch, below_patch, lpa_line],
              fontsize=8.5, framealpha=0.9, loc="upper right")

    plt.tight_layout()
    return fig


def plot_departure_heatmap(monthly_df: pd.DataFrame, district: str) -> plt.Figure:
    dist_monthly = monthly_df[monthly_df["district_name"] == district].copy()

    if dist_monthly.empty:
        return None

    pivot = dist_monthly.pivot_table(
        index="month", columns="year",
        values="departure_pct", aggfunc="mean"
    )

    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot.index = [month_labels[m-1] for m in pivot.index]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")

    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                   vmin=-100, vmax=100)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.astype(str), fontsize=9, color="#444")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9, color="#444")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                text_colour = "white" if abs(val) > 55 else "#333"
                ax.text(j, i, f"{val:.0f}%",
                        ha="center", va="center",
                        fontsize=7.5, color=text_colour, fontfamily="monospace")

    plt.colorbar(im, ax=ax, label="Departure from LPA (%)",
                 fraction=0.03, pad=0.04)

    ax.set_title(
        f"Monthly Departure from LPA — {district}",
        fontsize=13, fontweight="600", color="#1a1a2e", pad=14
    )
    ax.spines[:].set_visible(False)
    plt.tight_layout()
    return fig


# =============================================================================
# SECTION 5 — SHAP EXPLAINABILITY
# =============================================================================

import shap

@st.cache_data
def compute_shap_values(_model_dict, input_df_json: str):
    import json
    input_df = pd.read_json(input_df_json)

    model       = _model_dict["model"]
    le          = _model_dict["label_encoder"]
    class_names = list(le.classes_)

    from ensemble_model import prepare_features
    prepped_df, _ = prepare_features(input_df.copy(), is_predict=True)
    prepped_df = prepped_df.drop(columns=["target"], errors="ignore")

    feature_names = list(prepped_df.columns)
    X = prepped_df.values

    sub_models = {name: est for name, est in model.named_estimators_.items()}
    all_shap   = []

    for name, estimator in sub_models.items():
        explainer = shap.TreeExplainer(estimator)
        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            sv_arr = np.array([sv[c][0] for c in range(len(class_names))])
        else:
            sv_arr = sv[0].T

        all_shap.append(sv_arr)

    shap_vals_avg = np.mean(all_shap, axis=0)
    return shap_vals_avg, feature_names, class_names, X[0]


def plot_shap_waterfall(shap_vals_avg, feature_names, class_names,
                        input_values, predicted_label: str) -> plt.Figure:
    if predicted_label in class_names:
        cls_idx = class_names.index(predicted_label)
    else:
        cls_idx = 0

    vals  = shap_vals_avg[cls_idx]
    order = np.argsort(np.abs(vals))

    sorted_features = [feature_names[i] for i in order]
    sorted_vals     = [vals[i]           for i in order]
    sorted_inputs   = [input_values[i]   for i in order]

    colours = ["#2d7a4f" if v > 0 else "#b03030" for v in sorted_vals]

    fig, ax = plt.subplots(figsize=(8, max(3.5, len(feature_names) * 0.38)))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")

    bars = ax.barh(sorted_features, sorted_vals,
                   color=colours, height=0.55,
                   edgecolor="white", linewidth=0.6)

    for bar, inp, val in zip(bars, sorted_inputs, sorted_vals):
        x_pos = val + (0.003 if val >= 0 else -0.003)
        ha    = "left" if val >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"  {inp:.2f}",
                va="center", ha=ha,
                fontsize=8, color="#444", fontfamily="monospace")

    ax.axvline(x=0, color="#333", linewidth=0.8)
    ax.set_xlabel("SHAP value  (impact on model output)", fontsize=9, color="#555")
    ax.set_title(
        f"Why the model predicted: {predicted_label}",
        fontsize=12, fontweight="600", color="#1a1a2e", pad=12
    )
    ax.tick_params(labelsize=9, colors="#555")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#ddd")
    ax.xaxis.grid(True, linestyle="--", alpha=0.4, color="#ddd")
    ax.set_axisbelow(True)

    pos_patch = mpatches.Patch(color="#2d7a4f", label="Pushes toward prediction")
    neg_patch = mpatches.Patch(color="#b03030", label="Pulls away from prediction")
    ax.legend(handles=[pos_patch, neg_patch], fontsize=8, framealpha=0.9, loc="lower right")

    plt.tight_layout()
    return fig


# =============================================================================
# SECTION 6 — SIDEBAR
# =============================================================================

def render_sidebar(data: dict) -> tuple:
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 📍 Select Location")

    seasonal_df = data["seasonal"]

    states = sorted(seasonal_df["state_name"].dropna().unique().tolist())
    selected_state = st.sidebar.selectbox("State", states)

    districts = sorted(
        seasonal_df[seasonal_df["state_name"] == selected_state]
        ["district_name"].dropna().unique().tolist()
    )
    selected_district = st.sidebar.selectbox("District", districts)

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### June Rainfall Input")
    st.sidebar.markdown(
        "<small>Enter the total June rainfall recorded so far "
        "to generate an ML prediction for the rest of the season.</small>",
        unsafe_allow_html=True
    )

    monthly_df = data["monthly"]
    june_lpa_rows = monthly_df[
        (monthly_df["district_name"] == selected_district) &
        (monthly_df["month"] == 6)
    ]["lpa_mm"]
    june_lpa_default = float(june_lpa_rows.mean()) if len(june_lpa_rows) > 0 else 100.0

    june_rainfall_input = st.sidebar.number_input(
        label     = "June Rainfall (mm)",
        min_value = 0.0,
        max_value = 1000.0,
        value     = round(june_lpa_default, 1),
        step      = 5.0,
        help      = f"Historical June LPA for {selected_district}: {june_lpa_default:.1f} mm"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<small style='color:#aaa'>Data: IMD Grid Model | "
        "Years: 2018–2024 | Districts: Telangana</small>",
        unsafe_allow_html=True
    )

    return selected_state, selected_district, june_rainfall_input


# =============================================================================
# SECTION 7 — MAIN DASHBOARD (TABBED LAYOUT)
# =============================================================================

def main():

    # ── Load data & model ─────────────────────────────────────────────────
    data           = load_processed_data()
    model          = load_model()
    seasonal_df    = data["seasonal"]
    monthly_df     = data["monthly"]
    probability_df = data["probability"]

    # ── Sidebar ───────────────────────────────────────────────────────────
    selected_state, selected_district, june_rainfall_input = render_sidebar(data)

    # ── Page header ───────────────────────────────────────────────────────
    st.markdown(
        f'<div class="main-title">🌧️ Rainfall Analysis Dashboard</div>'
        f'<div class="main-subtitle">District-wise monsoon intelligence for Kharif crop planning</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    # ── Compute probabilities ─────────────────────────────────────────────
    emp_row = probability_df[
        (probability_df["district_name"] == selected_district) &
        (probability_df["season"] == "Kharif")
    ]
    empirical_prob = float(emp_row["prob_above_normal_pct"].iloc[0]) \
                     if not emp_row.empty else None

    ml_confidence, ml_label = make_ml_prediction(
        model, selected_district, selected_state,
        june_rainfall_input, seasonal_df, monthly_df
    )

    primary_prob   = empirical_prob if empirical_prob is not None else 50.0
    primary_colour = get_colour_for_probability(primary_prob)

    # =====================================================================
    # TABS
    # =====================================================================
    tab_overview, tab_charts, tab_data, tab_compare, tab_shap = st.tabs([
        "📊  Overview",
        "📈  Rainfall Charts",
        "📋  Raw Data",
        "🗺️  Compare Districts",
        "🔍  ML Explanation",
    ])

    # ─────────────────────────────────────────────────────────────────────
    # TAB 1 — OVERVIEW
    # ─────────────────────────────────────────────────────────────────────
    with tab_overview:

        st.markdown(
            f'<div style="font-size:1.6rem;font-weight:700;color:#2c7be5;margin-bottom:2px;">'
            f'{selected_district} Rainfall Outlook</div>'
            f'<div style="color:#999;font-size:1rem;margin-bottom:24px;">'
            f'{selected_state} • Kharif Season Intelligence</div>',
            unsafe_allow_html=True
        )

        # Three metric cards
        col1, col2, col3 = st.columns(3)

        with col1:
            colour = get_colour_for_probability(primary_prob)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Historical Probability</div>
                <div class="metric-value-{colour}">{primary_prob:.0f}%</div>
                <div class="metric-sublabel">Above-Normal Kharif<br>
                ({emp_row['years_above_normal'].iloc[0] if not emp_row.empty else '—'} of
                 {emp_row['total_years'].iloc[0] if not emp_row.empty else '—'} years)</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            if ml_label is not None:
                label_colour = {"Above Normal": "green", "Normal": "orange", "Deficit": "red"}.get(ml_label, "blue")
                ml_display   = ml_label
                ml_note      = f"June: {june_rainfall_input:.0f} mm"
            else:
                label_colour = "gray"
                ml_display   = "N/A"
                ml_note      = "Model not loaded"

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">ML Prediction</div>
                <div class="metric-value-{label_colour}">{ml_display}</div>
                <div class="metric-sublabel">{ml_note}<br></div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            last_year_row = seasonal_df[
                (seasonal_df["district_name"] == selected_district) &
                (seasonal_df["season"] == "Kharif")
            ].sort_values("year", ascending=False).head(1)

            if not last_year_row.empty:
                last_year       = int(last_year_row["year"].iloc[0])
                last_departure  = last_year_row["departure_pct"].iloc[0]
                last_total      = last_year_row["total_rainfall_mm"].iloc[0]
                last_lpa        = last_year_row["lpa_mm"].iloc[0]
                dep_colour      = "green" if last_departure > 5 else ("red" if last_departure < -5 else "orange")
                dep_sign        = "+" if last_departure > 0 else ""
            else:
                last_year = last_departure = last_total = last_lpa = "—"
                dep_colour = "gray"
                dep_sign   = ""

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Last Season ({last_year})</div>
                <div class="metric-value-{dep_colour}">{dep_sign}{last_departure:.0f}%</div>
                <div class="metric-sublabel">vs LPA<br>{last_total:.0f} mm actual • {last_lpa:.0f} mm LPA</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Farmer advice card
        st.markdown('<div class="section-header">Farmer Advisory</div>', unsafe_allow_html=True)
        advice  = get_advice(primary_prob, selected_district)
        colour  = primary_colour

        bullet_html = "".join(f"<li style='margin-bottom:6px'>{p}</li>" for p in advice["points"])
        st.markdown(f"""
        <div class="advice-card-{colour}">
            <div class="advice-title">{advice['title']}</div>
            <div class="advice-body"><ul style='margin:0;padding-left:18px'>{bullet_html}</ul></div>
        </div>
        """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────
    # TAB 2 — RAINFALL CHARTS  (stacked: trend on top, heatmap below)
    # ─────────────────────────────────────────────────────────────────────
    with tab_charts:

        st.markdown(
            f'<div style="font-size:1.1rem;font-weight:600;color:#333;margin-bottom:20px;">'
            f'Rainfall Trends for <span style="color:#2c7be5">{selected_district}</span></div>',
            unsafe_allow_html=True
        )

        # ── Chart 1: Kharif season bar chart ──
        st.markdown('<div class="section-header">Kharif Season Totals vs LPA</div>',
                    unsafe_allow_html=True)

        fig_trend = plot_rainfall_trend(seasonal_df, selected_district)
        if fig_trend:
            st.pyplot(fig_trend, width='stretch')
        else:
            st.info("No Kharif data available for this district.")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Chart 2: Monthly departure heatmap ──
        st.markdown('<div class="section-header">Monthly Departure from LPA (Heatmap)</div>',
                    unsafe_allow_html=True)

        fig_heat = plot_departure_heatmap(monthly_df, selected_district)
        if fig_heat:
            st.pyplot(fig_heat, width='stretch')
        else:
            st.info("No monthly data available for this district.")

    # ─────────────────────────────────────────────────────────────────────
    # TAB 3 — RAW DATA TABLE
    # ─────────────────────────────────────────────────────────────────────
    with tab_data:

        st.markdown(
            f'<div style="font-size:1.1rem;font-weight:600;color:#333;margin-bottom:8px;">'
            f'Raw Seasonal Data — <span style="color:#2c7be5">{selected_district}</span></div>'
            f'<div style="color:#999;font-size:0.85rem;margin-bottom:20px;">'
            f'All years and seasons on record</div>',
            unsafe_allow_html=True
        )

        dist_seasonal = seasonal_df[
            seasonal_df["district_name"] == selected_district
        ][["year","season","total_rainfall_mm","lpa_mm",
           "departure_pct","anomaly_category"]].sort_values(
            ["year","season"]
        ).reset_index(drop=True)

        def colour_anomaly(val):
            colours = {
                "Large Excess"  : "background-color: #c8f7c5",
                "Above Normal"  : "background-color: #d4edda",
                "Normal"        : "background-color: #fff9c4",
                "Below Normal"  : "background-color: #fde8d8",
                "Large Deficit" : "background-color: #f9c0c0",
            }
            return colours.get(val, "")

        styled = dist_seasonal.style\
            .applymap(colour_anomaly, subset=["anomaly_category"])\
            .format({
                "total_rainfall_mm": "{:.1f}",
                "lpa_mm"           : "{:.1f}",
                "departure_pct"    : "{:+.1f}%",
            })

        st.dataframe(styled, width='stretch', height=500)

    # ─────────────────────────────────────────────────────────────────────
    # TAB 4 — DISTRICT COMPARISON
    # ─────────────────────────────────────────────────────────────────────
    with tab_compare:

        st.markdown(
            '<div style="font-size:1.1rem;font-weight:600;color:#333;margin-bottom:8px;">'
            'All Districts — Kharif Above-Normal Probability</div>'
            '<div style="color:#999;font-size:0.85rem;margin-bottom:20px;">'
            f'Highlighted: <span style="color:#2c7be5;font-weight:600">{selected_district}</span></div>',
            unsafe_allow_html=True
        )

        kharif_probs = probability_df[
            probability_df["season"] == "Kharif"
        ].sort_values("prob_above_normal_pct", ascending=False).reset_index(drop=True)

        fig_comp, ax_comp = plt.subplots(figsize=(9, max(4, len(kharif_probs) * 0.28)))
        fig_comp.patch.set_facecolor("#fafafa")
        ax_comp.set_facecolor("#fafafa")

        colours_comp = [
            "#2d7a4f" if p >= 65 else
            "#1a5fa8" if p >= 50 else
            "#c07a00" if p >= 35 else "#b03030"
            for p in kharif_probs["prob_above_normal_pct"]
        ]

        for i, (district, val, colour) in enumerate(zip(
            kharif_probs["district_name"],
            kharif_probs["prob_above_normal_pct"],
            colours_comp
        )):
            alpha = 1.0 if district == selected_district else 0.55
            ax_comp.barh(district, val, color=colour, alpha=alpha,
                         height=0.65, edgecolor="white", linewidth=0.5)

        ax_comp.axvline(x=50, color="#aaa", linewidth=1,
                        linestyle="--", label="50% threshold")
        ax_comp.set_xlabel("Probability of Above-Normal Kharif (%)",
                           fontsize=9, color="#555")
        ax_comp.set_title("District Comparison — Above-Normal Probability",
                          fontsize=12, fontweight="600", color="#1a1a2e", pad=12)
        ax_comp.tick_params(labelsize=8, colors="#555")
        ax_comp.spines[["top","right"]].set_visible(False)
        ax_comp.spines[["left","bottom"]].set_color("#ddd")
        ax_comp.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax_comp.set_xlim(0, 105)
        ax_comp.invert_yaxis()

        for i, val in enumerate(kharif_probs["prob_above_normal_pct"]):
            ax_comp.text(val + 1, i,
                         f"{val:.0f}%", va="center", fontsize=7.5,
                         color="#333", fontfamily="monospace")

        plt.tight_layout()
        st.pyplot(fig_comp, width='stretch')

    # ─────────────────────────────────────────────────────────────────────
    # TAB 5 — ML / SHAP EXPLANATION
    # ─────────────────────────────────────────────────────────────────────
    with tab_shap:

        st.markdown(
            '<div style="font-size:1.1rem;font-weight:600;color:#333;margin-bottom:8px;">'
            'Why did the model predict this?</div>'
            '<div style="color:#999;font-size:0.85rem;margin-bottom:20px;">'
            'SHAP feature importance for the current prediction</div>',
            unsafe_allow_html=True
        )

        if model is None:
            st.markdown(
                '<div class="warn-banner">⚠️ Model not loaded — SHAP explanation unavailable. '
                'Ensure <code>rainfall_ensemble_model.pkl</code> is present.</div>',
                unsafe_allow_html=True
            )
        elif ml_label is None:
            st.markdown(
                '<div class="info-banner">Run a prediction first by entering June rainfall in the sidebar.</div>',
                unsafe_allow_html=True
            )
        else:
            try:
                # Rebuild input_df for SHAP (mirrors make_ml_prediction logic)
                from sklearn.preprocessing import LabelEncoder

                june_rows = monthly_df[
                    (monthly_df["district_name"] == selected_district) &
                    (monthly_df["month"] == 6)
                ].sort_values("year")
                if june_rows.empty:
                    june_rows = monthly_df[monthly_df["district_name"] == selected_district]\
                                .sort_values(["year","month"])

                base_row = june_rows.tail(1).copy()

                input_data = {
                    'year'             : int(base_row['year'].iloc[0]) if 'year' in base_row.columns else 2024,
                    'month'            : 6,
                    'total_rainfall_mm': june_rainfall_input,
                    'rainy_days'       : int(base_row['rainy_days'].iloc[0]) if 'rainy_days' in base_row.columns else 15,
                    'data_days'        : int(base_row['data_days'].iloc[0]) if 'data_days' in base_row.columns else 30,
                    'mean_spi_30d'     : float(base_row['mean_spi_30d'].iloc[0]) if 'mean_spi_30d' in base_row.columns else 0.0,
                    'lpa_mm'           : float(base_row['lpa_mm'].iloc[0]) if 'lpa_mm' in base_row.columns else june_rainfall_input,
                    'month_sin'        : np.sin(2 * np.pi * 6 / 12),
                    'month_cos'        : np.cos(2 * np.pi * 6 / 12),
                }

                le_d = LabelEncoder(); le_s = LabelEncoder()
                le_d.fit(monthly_df["district_name"].astype(str).unique())
                le_s.fit(monthly_df["state_name"].astype(str).unique())
                input_data['district_name_enc'] = le_d.transform([selected_district])[0]
                input_data['state_name_enc']    = le_s.transform([selected_state])[0] \
                                                  if selected_state in le_s.classes_ else 0

                june_lpa     = input_data['lpa_mm']
                departure_pct = ((june_rainfall_input - june_lpa) / june_lpa * 100) if june_lpa > 0 else 0.0
                input_df = pd.DataFrame([input_data])
                input_df['departure_pct'] = departure_pct
                input_df['state_name']    = selected_state
                input_df['district_name'] = selected_district

                input_json = input_df.to_json()

                with st.spinner("Computing SHAP explanations…"):
                    shap_vals, feat_names, class_names, input_vals = compute_shap_values(
                        model, input_json
                    )

                fig_shap = plot_shap_waterfall(shap_vals, feat_names, class_names,
                                               input_vals, ml_label)
                st.pyplot(fig_shap, width='stretch')

                top_idx   = int(np.argmax(np.abs(shap_vals[class_names.index(ml_label)])))
                top_feat  = feat_names[top_idx]
                top_val   = shap_vals[class_names.index(ml_label)][top_idx]
                direction = "toward" if top_val > 0 else "away from"

                st.markdown(
                    f'<div class="info-banner">'
                    f'<b>Most influential feature:</b> <code>{top_feat}</code> '
                    f'— pushed the prediction <b>{direction}</b> <em>{ml_label}</em>.'
                    f'</div>',
                    unsafe_allow_html=True
                )

            except Exception as e:
                st.markdown(
                    f'<div class="warn-banner">SHAP computation failed: {e}</div>',
                    unsafe_allow_html=True
                )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()