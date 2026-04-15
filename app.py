"""
=============================================================================
RAINFALL ANALYSIS & PREDICTION FRAMEWORK
Step 4: Streamlit Dashboard — The Farmer's Interface
=============================================================================
Purpose : Interactive dashboard for farmers and agricultural officers.
          Loads pre-processed CSVs and the trained XGBoost regressor to display:
          - District rainfall trends vs LPA
          - Probability of Above-Normal monsoon (empirical)
          - ML-predicted daily rainfall for selected district/date
          - Plain-English farmer advice
Run with: streamlit run app.py
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# ONLY CHANGE 1 — import
from ensemble_model import load_model, predict as ensemble_predict
import warnings

warnings.filterwarnings("ignore")


# =============================================================================
# SECTION 1 — PAGE CONFIGURATION
# Must be the FIRST Streamlit command in the script.
# =============================================================================

st.set_page_config(
    page_title="Rainfall Analysis Framework",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# SECTION 2 — CUSTOM CSS STYLING
# =============================================================================

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono&display=swap');

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

    .main-title {
        font-size: 2.6rem; font-weight: 800;
        color: #2c7be5; margin-bottom: 0; letter-spacing: -0.5px;
    }
    .main-subtitle {
        font-size: 1.1rem; color: #999; margin-top: 4px; font-weight: 300;
    }
    .metric-card {
        background: white; border-radius: 12px; padding: 24px;
        border: 1px solid #e8e8e8; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        text-align: center; margin-bottom: 16px;
    }
    .metric-label {
        font-size: 0.78rem; font-weight: 500; text-transform: uppercase;
        letter-spacing: 1px; color: #888; margin-bottom: 8px;
    }
    .metric-value-green  { font-size: 2.8rem; font-weight: 600; color: #2d7a4f; line-height: 1; }
    .metric-value-blue   { font-size: 2.8rem; font-weight: 600; color: #1a5fa8; line-height: 1; }
    .metric-value-orange { font-size: 2.8rem; font-weight: 600; color: #c07a00; line-height: 1; }
    .metric-value-red    { font-size: 2.8rem; font-weight: 600; color: #b03030; line-height: 1; }
    .metric-value-gray   { font-size: 2.8rem; font-weight: 600; color: #555;    line-height: 1; }
    .metric-sublabel { font-size: 0.82rem; color: #999; margin-top: 6px; }

    .advice-card-green  { background:#f0faf4; border-left:4px solid #2d7a4f; border-radius:8px; padding:20px; }
    .advice-card-blue   { background:#f0f6ff; border-left:4px solid #1a5fa8; border-radius:8px; padding:20px; }
    .advice-card-orange { background:#fff8e6; border-left:4px solid #c07a00; border-radius:8px; padding:20px; }
    .advice-card-red    { background:#fff0f0; border-left:4px solid #b03030; border-radius:8px; padding:20px; }
    .advice-title { font-size:1rem; font-weight:600; margin-bottom:8px; }
    .advice-body  { font-size:0.92rem; line-height:1.7; color:#333; }

    .section-header {
        font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 1.5px; color: #aaa; border-bottom: 1px solid #eee;
        padding-bottom: 8px; margin-bottom: 16px; margin-top: 8px;
    }
    .info-banner {
        background: #f8f9ff; border: 1px solid #dde3ff; border-radius: 8px;
        padding: 12px 16px; font-size: 0.85rem; color: #445; margin-bottom: 16px;
    }
    #MainMenu {visibility: hidden;}
    footer    {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# SECTION 3 — DATA & MODEL LOADERS
# =============================================================================

@st.cache_data
def load_processed_data() -> dict:
    """
    Loads the processed CSVs from Google Drive.
    Replace FILE_IDS values with your own shared Drive file IDs.
    """
    FILE_IDS = {
        "monthly"    : "1AY2n7HBfu0BsrLlDL80iflWqlqLSYiMH",
        "seasonal"   : "1rnbhP44S_gah-v7L6BRJKBZInwLSLjDG",
        "probability": "1wHgLiXOuvqLmpzaPSoj73rWpTPmHYgp2",
    }

    def drive_url(fid: str) -> str:
        return f"https://drive.google.com/uc?export=download&id={fid}"

    data = {}
    for key, fid in FILE_IDS.items():
        try:
            data[key] = pd.read_csv(drive_url(fid))
        except Exception as e:
            st.error(
                f"Could not load '{key}' data from Google Drive.\n\n"
                f"Make sure the file is shared as 'Anyone with the link → Viewer'.\n\n"
                f"Error: {e}"
            )
            st.stop()
    return data


@st.cache_resource
def get_model():
    """Loads the trained XGBoost regressor from disk. Returns None if not found."""
    try:
        return load_model()
    except Exception:
        return None


# =============================================================================
# SECTION 4 — HELPER FUNCTIONS
# =============================================================================

def get_colour_for_probability(prob: float) -> str:
    if prob >= 65:   return "green"
    elif prob >= 50: return "blue"
    elif prob >= 35: return "orange"
    else:            return "red"


def get_colour_for_rainfall(predicted_mm: float, lpa_daily_mm: float) -> str:
    """Colour the ML card relative to the month's daily LPA."""
    if lpa_daily_mm <= 0:
        return "gray"
    ratio = predicted_mm / lpa_daily_mm
    if ratio >= 1.20:   return "green"
    elif ratio >= 1.00: return "blue"
    elif ratio >= 0.80: return "orange"
    else:               return "red"


def get_advice(prob: float, district: str) -> dict:
    if prob >= 65:
        return {
            "title": "✅ Good monsoon likely — plan for full sowing",
            "points": [
                f"Strong signal of above-normal rainfall for {district} this Kharif season.",
                "Proceed with your full sowing plan for water-intensive crops (paddy, sugarcane).",
                "Ensure drainage channels and bunds are cleared before June to handle surplus water.",
                "Stock up on seeds and fertilisers early — demand will be high.",
                "Consider crop insurance nonetheless; even good monsoons can have dry spells.",
            ],
        }
    elif prob >= 50:
        return {
            "title": "🔵 Moderate outlook — proceed with caution",
            "points": [
                f"Moderate confidence of above-normal rainfall for {district}.",
                "Sow primary crops as planned but reserve ~20% of your area as a buffer.",
                "Diversify: mix a water-intensive crop with one drought-tolerant variety.",
                "Monitor weekly district rainfall updates and adjust irrigation schedules.",
                "Consult your local Krishi Vigyan Kendra (KVK) for variety recommendations.",
            ],
        }
    elif prob >= 35:
        return {
            "title": "🟡 Near-normal expected — standard precautions advised",
            "points": [
                f"Season likely to be near the historical average for {district}.",
                "Follow your standard sowing schedule for the district.",
                "Keep drought-tolerant variety seeds (e.g. millets, sorghum) as backup.",
                "Check soil moisture before each irrigation cycle to avoid over-watering.",
                "Review your crop insurance policy before 30th June.",
            ],
        }
    else:
        return {
            "title": "🔴 Below-normal risk — take protective action",
            "points": [
                f"Higher than usual risk of below-normal monsoon for {district}.",
                "Strongly consider drought-resistant crop varieties this season.",
                "Prioritise water conservation: repair farm ponds and check-dams now.",
                "Avoid over-investment in water-intensive crops like paddy this cycle.",
                "Contact your district agriculture officer and register for drought relief schemes.",
                "Visit the nearest Krishi Vigyan Kendra before finalising your sowing plans.",
            ],
        }


def make_ml_prediction(
    model,
    district: str,
    state: str,
    selected_date: pd.Timestamp,
):

    month = selected_date.month

    if month in {6, 7, 8, 9}:     season = "Kharif"
    elif month in {10, 11, 12, 1, 2}: season = "Rabi"
    else:                          season = "Zaid"

    input_df = pd.DataFrame({
        "date": [selected_date],
        "state_name": [state],
        "district_name": [district],
        "rainfall_mm": [np.nan],
        "season": [season],
        "year": [selected_date.year],
        "month": [month],
        "week_number": [selected_date.isocalendar().week],
        "departure_pct": [0]  # required for classifier
    })

    try:
        labels, probs = ensemble_predict(model, input_df)

        pred_label = labels[0]
        confidence = float(np.max(probs[0]) * 100)

        return pred_label, confidence

    except Exception as e:
        st.warning(f"ML prediction failed: {e}")
        return None
    
def plot_rainfall_trend(seasonal_df: pd.DataFrame, district: str):
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

    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")

    bars = ax.bar(years, rainfall, color=bar_colours,
                  width=0.55, zorder=3, edgecolor="white", linewidth=0.8)
    ax.axhline(y=lpa, color="#1a1a2e", linewidth=1.5, linestyle="--", zorder=4)

    for bar, val in zip(bars, rainfall):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                f"{val:.0f}", ha="center", va="bottom",
                fontsize=8.5, color="#333", fontfamily="monospace")

    ax.set_xlabel("Year", fontsize=10, color="#555", labelpad=8)
    ax.set_ylabel("Total Rainfall (mm)", fontsize=10, color="#555", labelpad=8)
    ax.set_title(f"Kharif Season Rainfall — {district}",
                 fontsize=13, fontweight="600", color="#1a1a2e", pad=14)
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


def plot_departure_heatmap(monthly_df: pd.DataFrame, district: str):
    dist_monthly = monthly_df[monthly_df["district_name"] == district].copy()
    if dist_monthly.empty:
        return None

    pivot = dist_monthly.pivot_table(
        index="month", columns="year", values="departure_pct", aggfunc="mean"
    )
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot.index = [month_labels[m - 1] for m in pivot.index]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")

    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=-100, vmax=100)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.astype(str), fontsize=9, color="#444")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9, color="#444")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                tc = "white" if abs(val) > 55 else "#333"
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                        fontsize=7.5, color=tc, fontfamily="monospace")

    plt.colorbar(im, ax=ax, label="Departure from LPA (%)", fraction=0.03, pad=0.04)
    ax.set_title(f"Monthly Departure from LPA — {district}",
                 fontsize=13, fontweight="600", color="#1a1a2e", pad=14)
    ax.spines[:].set_visible(False)
    plt.tight_layout()
    return fig


# =============================================================================
# SECTION 6 — SIDEBAR
# =============================================================================

def render_sidebar(data: dict) -> tuple:
    st.sidebar.markdown("## 🌧️ Rainfall Framework")
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
    st.sidebar.markdown("#### 📅 ML Prediction Date")
    st.sidebar.markdown(
        "<small>Pick a date to get the XGBoost model's predicted "
        "daily rainfall for the selected district.</small>",
        unsafe_allow_html=True,
    )
    selected_date = st.sidebar.date_input(
        "Date",
        value=pd.Timestamp("2024-06-15"),
        min_value=pd.Timestamp("2018-01-01"),
        max_value=pd.Timestamp("2030-12-31"),
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<small style='color:#aaa'>Data: IMD Grid Model | "
        "Years: 2018–2024 | Districts: Telangana</small>",
        unsafe_allow_html=True,
    )
    return selected_state, selected_district, pd.Timestamp(selected_date)


# =============================================================================
# SECTION 7 — MAIN DASHBOARD
# =============================================================================

def main():

    data           = load_processed_data()
    model          = get_model()
    seasonal_df    = data["seasonal"]
    monthly_df     = data["monthly"]
    probability_df = data["probability"]

    selected_state, selected_district, selected_date = render_sidebar(data)

    # ── Header ─────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="main-title">🌧️ Rainfall Analysis Dashboard</div>'
        '<div class="main-subtitle">District-wise monsoon intelligence for Kharif crop planning</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Empirical probability ──────────────────────────────────────────────
    emp_row = probability_df[
        (probability_df["district_name"] == selected_district) &
        (probability_df["season"] == "Kharif")
    ]
    empirical_prob = (
        float(emp_row["prob_above_normal_pct"].iloc[0]) if not emp_row.empty else 50.0
    )
    primary_colour = get_colour_for_probability(empirical_prob)

    # ── XGBoost daily rainfall prediction ─────────────────────────────────
    ml_result = make_ml_prediction(
        model, selected_district, selected_state, selected_date
    )

    if ml_result:
        pred_label, confidence = ml_result
    else:
        pred_label, confidence = None, None

    # Monthly daily LPA for comparison
    month_lpa_rows = monthly_df[
        (monthly_df["district_name"] == selected_district) &
        (monthly_df["month"] == selected_date.month)
    ]["lpa_mm"]
    month_daily_lpa = (
        float(month_lpa_rows.mean()) / 30.0 if len(month_lpa_rows) > 0 else None
    )

    # ── ROW 1: Metric cards ────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        years_above = emp_row["years_above_normal"].iloc[0] if not emp_row.empty else "—"
        total_years = emp_row["total_years"].iloc[0]        if not emp_row.empty else "—"
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Empirical Probability</div>
                <div class="metric-value-{primary_colour}">{empirical_prob:.0f}%</div>
                <div class="metric-sublabel">Above-Normal Kharif Season<br>
                ({years_above} of {total_years} historical years)</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col2:
        if pred_label is not None:
            ml_colour  = "blue"  # classification → no numeric comparison
            ml_display = pred_label
            ml_note    = f"Confidence: {confidence:.1f}%"
            lpa_note   = "Ensemble classification model"
        else:
            ml_colour  = "gray"
            ml_display = "N/A"
            ml_note    = "Model not loaded"
            lpa_note   = "Run ensemble_model.py first"

        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Ensemble Prediction</div>
                <div class="metric-value-{ml_colour}">{ml_display}</div>
                <div class="metric-sublabel">{ml_note}<br>{lpa_note}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col3:
        last_row = (
            seasonal_df[
                (seasonal_df["district_name"] == selected_district) &
                (seasonal_df["season"] == "Kharif")
            ].sort_values("year", ascending=False).head(1)
        )
        if not last_row.empty:
            last_year  = int(last_row["year"].iloc[0])
            last_total = last_row["total_rainfall_mm"].iloc[0]
            last_dep   = last_row["departure_pct"].iloc[0]
            last_lpa   = last_row["lpa_mm"].iloc[0]
            dep_colour = "green" if last_dep > 5 else "red" if last_dep < -5 else "orange"
            dep_sign   = "+" if last_dep > 0 else ""
        else:
            last_year = last_total = last_dep = last_lpa = "—"
            dep_colour = "gray"; dep_sign = ""

        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Last Full Season ({last_year})</div>
                <div class="metric-value-{dep_colour}">{dep_sign}{last_dep:.0f}%</div>
                <div class="metric-sublabel">Departure from LPA<br>
                {last_total:.0f} mm actual vs {last_lpa:.0f} mm LPA</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # ── ROW 2: Advice ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Farmer\'s Advice</div>',
                unsafe_allow_html=True)
    advice      = get_advice(empirical_prob, selected_district)
    advice_html = "".join(
        [f"<li style='margin-bottom:6px'>{pt}</li>" for pt in advice["points"]]
    )
    st.markdown(
        f"""<div class="advice-card-{primary_colour}">
            <div class="advice-title">{advice['title']}</div>
            <div class="advice-body">
                <ul style="margin:0; padding-left:20px">{advice_html}</ul>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """<div class="info-banner">
        ℹ️ <strong>Note on LPA:</strong> This dashboard uses 2018–2024 (7 years) as the
        baseline. IMD's official LPA uses 30 years — probabilities will improve as more data is added.
        </div>
        <div class="info-banner">
        🤖 <strong>About the ML card:</strong> The XGBoost model predicts <em>daily rainfall (mm)</em>
        for the chosen district and date. It is a regression model — not a probability classifier.
        Use it alongside the empirical probability for planning decisions.
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── ROW 3: Charts ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Rainfall Charts</div>',
                unsafe_allow_html=True)
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig = plot_rainfall_trend(seasonal_df, selected_district)
        st.pyplot(fig, use_container_width=True) if fig else st.info("No Kharif data available.")

    with chart_col2:
        fig = plot_departure_heatmap(monthly_df, selected_district)
        st.pyplot(fig, use_container_width=True) if fig else st.info("No monthly data available.")

    # ── ROW 4: Raw data ────────────────────────────────────────────────────
    with st.expander("📊 View raw seasonal data for this district"):
        dist_seasonal = (
            seasonal_df[seasonal_df["district_name"] == selected_district][
                ["year", "season", "total_rainfall_mm", "lpa_mm",
                 "departure_pct", "anomaly_category"]
            ].sort_values(["year", "season"]).reset_index(drop=True)
        )

        def colour_anomaly(val: str) -> str:
            return {
                "Large Excess" : "background-color: #c8f7c5",
                "Above Normal" : "background-color: #d4edda",
                "Normal"       : "background-color: #fff9c4",
                "Below Normal" : "background-color: #fde8d8",
                "Large Deficit": "background-color: #f9c0c0",
            }.get(val, "")

        styled = (
            dist_seasonal.style
            .map(colour_anomaly, subset=["anomaly_category"])
            .format({"total_rainfall_mm": "{:.1f}", "lpa_mm": "{:.1f}",
                     "departure_pct": "{:+.1f}%"})
        )
        st.dataframe(styled, use_container_width=True, height=300)

    # ── ROW 5: District comparison ─────────────────────────────────────────
    with st.expander("🗺️ Compare all districts — Kharif above-normal probability"):
        kharif_probs = (
            probability_df[probability_df["season"] == "Kharif"]
            .sort_values("prob_above_normal_pct", ascending=False)
            .reset_index(drop=True)
        )
        fig_comp, ax_comp = plt.subplots(
            figsize=(6, max(3, len(kharif_probs) * 0.22))
        )
        fig_comp.patch.set_facecolor("#fafafa")
        ax_comp.set_facecolor("#fafafa")

        colours_comp = [
            "#2d7a4f" if p >= 65 else "#1a5fa8" if p >= 50
            else "#c07a00" if p >= 35 else "#b03030"
            for p in kharif_probs["prob_above_normal_pct"]
        ]
        for i, (dist, val, colour) in enumerate(zip(
            kharif_probs["district_name"],
            kharif_probs["prob_above_normal_pct"],
            colours_comp,
        )):
            alpha = 1.0 if dist == selected_district else 0.55
            ax_comp.barh(dist, val, color=colour, alpha=alpha,
                         height=0.65, edgecolor="white", linewidth=0.5)

        ax_comp.axvline(x=50, color="#aaa", linewidth=1, linestyle="--")
        ax_comp.set_xlabel("Probability of Above-Normal Kharif (%)", fontsize=9, color="#555")
        ax_comp.set_title("District Comparison — Above-Normal Probability",
                          fontsize=11, fontweight="600", color="#1a1a2e", pad=10)
        ax_comp.tick_params(labelsize=8, colors="#555")
        ax_comp.spines[["top", "right"]].set_visible(False)
        ax_comp.spines[["left", "bottom"]].set_color("#ddd")
        ax_comp.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax_comp.set_xlim(0, 105)
        ax_comp.invert_yaxis()

        for i, val in enumerate(kharif_probs["prob_above_normal_pct"]):
            ax_comp.text(val + 1, i, f"{val:.0f}%", va="center",
                         fontsize=7.5, color="#333", fontfamily="monospace")
        plt.tight_layout()
        st.pyplot(fig_comp, use_container_width=True)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()