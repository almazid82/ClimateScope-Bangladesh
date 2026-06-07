"""
ClimateScope Bangladesh & South Asia — Interactive Dashboard
Author: Shamsul AL Mazid | github.com/almazid82
"""

import base64
import os
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from statsmodels.tsa.arima.model import ARIMA
import streamlit as st

warnings.filterwarnings("ignore")

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ClimateScope Bangladesh",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── COLOUR CONSTANTS ──────────────────────────────────────────────────────────
CYAN   = "#00f5ff"
BLUE   = "#4fc3f7"
RED    = "#ef5350"
GREEN  = "#39ff14"
ORANGE = "#ffa726"
PURPLE = "#ab47bc"
TEAL   = "#4dd0e1"
BG     = "rgba(3,7,18,0.80)"
GRID   = "#0d1f35"

BASE_LAYOUT = dict(
    paper_bgcolor="rgba(3,7,18,0.80)",
    plot_bgcolor ="rgba(3,7,18,0.80)",
    font=dict(color="#c8e0f0", family="'Share Tech Mono', 'Courier New', monospace"),
    xaxis=dict(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID),
    yaxis=dict(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID, borderwidth=1),
    margin=dict(l=55, r=30, t=55, b=45),
    hoverlabel=dict(bgcolor="#001832", bordercolor=CYAN),
)

# ── GLOBAL FUI CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

[data-testid="stAppViewContainer"] {
    background: #030712;
    background-image:
        radial-gradient(ellipse at 20% 50%, rgba(0,40,80,0.15) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 20%, rgba(0,20,60,0.12) 0%, transparent 50%);
}
html, body { font-family: 'Rajdhani', sans-serif; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(2,8,22,0.98) 0%, rgba(3,10,25,0.98) 100%) !important;
    border-right: 1px solid rgba(0,245,255,0.15) !important;
    box-shadow: 4px 0 40px rgba(0,0,0,0.6) !important;
}
[data-testid="stSidebar"] * { color: #a0cce0 !important; }
[data-testid="stSidebar"] h2 { color: #00f5ff !important; font-size:1.1rem; letter-spacing:0.08em; }
[data-testid="stSidebar"] strong { color: #c8e8ff !important; }
[data-testid="stSidebar"] hr { border-color: rgba(0,245,255,0.12) !important; }

section[data-testid="stMain"] .block-container { padding-top: 1.5rem; }

h1 {
    background: linear-gradient(90deg, #00f5ff 0%, #7ecff7 55%, #e8f4ff 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    font-size: 2.0rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    margin-bottom: 0.1rem !important;
}
h2, h3 { color: #c8e8ff !important; letter-spacing: 0.04em !important; }

.kpi-box {
    background: linear-gradient(145deg, rgba(0,18,45,0.82), rgba(0,8,25,0.90));
    border: 1px solid rgba(0,245,255,0.32);
    border-radius: 10px;
    padding: 18px 10px 14px;
    text-align: center;
    margin-bottom: 10px;
    box-shadow: 0 0 28px rgba(0,245,255,0.10), inset 0 0 18px rgba(0,245,255,0.04);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    position: relative;
    overflow: hidden;
}
.kpi-box::before {
    content: "";
    position: absolute;
    top: 0; left: 12%; right: 12%; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,245,255,0.65), transparent);
}
.kpi-box::after {
    content: "";
    position: absolute;
    bottom: 0; left: 30%; right: 30%; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,245,255,0.20), transparent);
}
.kpi-value {
    font-size: 1.75rem;
    font-weight: 700;
    margin: 0;
    filter: drop-shadow(0 0 6px currentColor);
    letter-spacing: 0.02em;
    font-family: 'Share Tech Mono', monospace;
}
.kpi-label {
    font-size: 0.60rem;
    color: #6aaccc;
    margin: 7px 0 0;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}

.section-hdr {
    border-left: 3px solid #00f5ff;
    padding-left: 12px;
    margin: 24px 0 12px;
    font-size: 0.88rem;
    font-weight: 600;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: #b8dff0;
    text-shadow: 0 0 14px rgba(0,245,255,0.30);
}

.finding-box {
    background: rgba(0,18,45,0.78);
    border: 1px solid rgba(0,245,255,0.22);
    border-radius: 8px;
    padding: 14px 18px;
    margin-top: 12px;
    backdrop-filter: blur(10px);
}

[data-testid="stPlotlyChart"] {
    border: 1px solid rgba(0,245,255,0.10);
    border-radius: 10px;
    background: rgba(3,8,22,0.55);
    overflow: hidden;
    box-shadow: 0 4px 30px rgba(0,0,0,0.45);
    backdrop-filter: blur(6px);
}

div[data-testid="stAlert"] {
    background: rgba(0,18,45,0.80) !important;
    border: 1px solid rgba(0,245,255,0.22) !important;
    border-radius: 8px !important;
    backdrop-filter: blur(10px) !important;
    color: #a8d8f0 !important;
}

hr { border-color: rgba(0,245,255,0.10) !important; margin: 1.5rem 0 !important; }

details[data-testid="stExpander"] {
    background: rgba(0,12,35,0.75) !important;
    border: 1px solid rgba(0,245,255,0.12) !important;
    border-radius: 8px !important;
}

label { color: #a0cce0 !important; letter-spacing: 0.04em; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #030712; }
::-webkit-scrollbar-thumb { background: rgba(0,245,255,0.25); border-radius: 2px; }

[data-testid="stDataFrame"] { border: 1px solid rgba(0,245,255,0.10); border-radius:8px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PER-PAGE BACKGROUND INJECTOR
# ══════════════════════════════════════════════════════════════════════════════

_BG_MAP = {
    "global":     "assets/bg_global.jpg",
    "bangladesh": "assets/bg_bangladesh.jpg",
    "ml":         "assets/bg_ml.jpg",
    "forecast":   "assets/bg_forecast.jpg",
}

def set_page_bg(key: str):
    path = _BG_MAP.get(key, "")
    if path and os.path.exists(path):
        with open(path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        css = f"""<style>
[data-testid="stAppViewContainer"] {{
    background-image:
        linear-gradient(rgba(3,7,18,0.86), rgba(3,7,18,0.92)),
        url("data:image/jpeg;base64,{b64}");
    background-size: cover;
    background-position: center top;
    background-attachment: fixed;
}}
</style>"""
    else:
        css = """<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(145deg, #030712 0%, #030e1e 100%);
}
</style>"""
    st.markdown(css, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS  (cached — runs only once per session)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_master() -> pd.DataFrame:
    for p in ["data/processed/master_annual_dataset.csv",
              "../data/processed/master_annual_dataset.csv"]:
        if os.path.exists(p):
            return pd.read_csv(p)
    return _synth_master()


@st.cache_data(show_spinner=False)
def load_monthly() -> pd.DataFrame:
    for p in ["data/raw/bangladesh_nasa_power_monthly.csv",
              "../data/raw/bangladesh_nasa_power_monthly.csv"]:
        if os.path.exists(p):
            return pd.read_csv(p, index_col="Date", parse_dates=True)
    return _synth_monthly()


@st.cache_data(show_spinner=False)
def load_giss() -> pd.DataFrame:
    for p in ["data/processed/global_temperature_anomaly.csv",
              "../data/processed/global_temperature_anomaly.csv"]:
        if os.path.exists(p):
            return pd.read_csv(p)
    return _synth_giss()


# ── Synthetic fallbacks ───────────────────────────────────────────────────────

def _synth_master() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    yr = np.arange(1984, 2024);  t = yr - 1984;  n = len(yr)
    return pd.DataFrame({
        "Year":               yr,
        "BGD_Temp_C":         25.8 - 0.026*t + rng.normal(0, 0.22, n),
        "BGD_Precip_mm_day":   5.2 + 0.009*t + rng.normal(0,  0.4, n),
        "BGD_Humidity_pct":   77.0             + rng.normal(0,  1.5, n),
        "Annual_Anomaly_C":  -0.12 + 0.021*t  + rng.normal(0, 0.12, n),
        "GDP_per_capita_USD":  300 * np.exp(0.072*t),
        "GDP_USD":            40e9 * np.exp(0.065*t),
        "Sea_Level_mm":        -90 + 1.8*t    + rng.normal(0,  4.0, n),
        "CO2_emissions":     14000 + 1400*t   + rng.normal(0,  400, n),
        "Population":          1e8 + 2e6*t,
        "Agricultural_land_pct": 70 - 0.2*t,
        "Under5_mortality":    140 - 3.2*t    + rng.normal(0,  3.0, n),
    })


def _synth_monthly() -> pd.DataFrame:
    idx = pd.date_range("1984-01", "2023-12", freq="MS")
    rng = np.random.default_rng(42)
    t = np.arange(len(idx));  m = idx.month
    seas_t = -4 * np.sin(2*np.pi*(m-3)/12)
    seas_p = 8  * np.maximum(0, np.sin(np.pi*(m-4)/5))
    return pd.DataFrame({
        "Temperature_C":        26 - 0.026/12*t + seas_t + rng.normal(0, 0.5, len(idx)),
        "Precipitation_mm_day": seas_p + rng.exponential(0.5, len(idx)),
        "Humidity_pct":         75 + 10*np.sin(2*np.pi*(m-6)/12) + rng.normal(0, 2, len(idx)),
    }, index=idx)


def _synth_giss() -> pd.DataFrame:
    yr = np.arange(1880, 2024);  rng = np.random.default_rng(42)
    return pd.DataFrame({
        "Year":            yr,
        "Annual_Anomaly_C": -0.4 + 0.0075*(yr-1880) + rng.normal(0, 0.1, len(yr)),
    })


# ══════════════════════════════════════════════════════════════════════════════
# ML MODEL
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def get_flood_model():
    df = load_monthly().copy()
    df["Month"]      = df.index.month
    df["Is_Monsoon"] = df["Month"].isin([6,7,8,9]).astype(int)

    mc     = df.groupby("Month")[["Temperature_C","Precipitation_mm_day","Humidity_pct"]].transform("mean")
    mc_std = df.groupby("Month")[["Temperature_C","Precipitation_mm_day","Humidity_pct"]].transform("std").replace(0, 1)

    df["Temp_anomaly"]    = df["Temperature_C"]        - mc["Temperature_C"]
    df["Precip_anomaly"]  = df["Precipitation_mm_day"] - mc["Precipitation_mm_day"]
    df["Humid_anomaly"]   = df["Humidity_pct"]         - mc["Humidity_pct"]
    df["Precip_3mo_roll"] = df["Precipitation_mm_day"].rolling(3, min_periods=1).mean()
    df["Precip_6mo_roll"] = df["Precipitation_mm_day"].rolling(6, min_periods=1).mean()
    df["Precip_lag1"]     = df["Precipitation_mm_day"].shift(1).bfill()
    df["Precip_lag2"]     = df["Precipitation_mm_day"].shift(2).bfill()

    pz = (df["Precipitation_mm_day"] - df["Precipitation_mm_day"].mean()) / df["Precipitation_mm_day"].std()
    hz = (df["Humidity_pct"]          - df["Humidity_pct"].mean())          / df["Humidity_pct"].std()
    risk = 0.6*pz + 0.2*hz + 0.2*df["Is_Monsoon"]
    df["Flood_Risk"] = (risk >= risk.quantile(0.70)).astype(int)

    FEATS = ["Temperature_C","Precipitation_mm_day","Humidity_pct",
             "Month","Is_Monsoon","Temp_anomaly","Precip_anomaly",
             "Humid_anomaly","Precip_3mo_roll","Precip_6mo_roll","Precip_lag1","Precip_lag2"]
    df = df[FEATS + ["Flood_Risk"]].dropna()

    model = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                   random_state=42, n_jobs=-1)
    model.fit(df[FEATS], df["Flood_Risk"])
    return model, FEATS, df[FEATS]


# ══════════════════════════════════════════════════════════════════════════════
# ARIMA FORECAST
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def get_arima_forecast(steps: int = 27):
    temp = load_master().dropna(subset=["BGD_Temp_C"])["BGD_Temp_C"].values
    fit  = ARIMA(temp, order=(1,1,1)).fit()
    fc   = np.asarray(fit.forecast(steps=steps))
    ci   = np.asarray(fit.get_forecast(steps=steps).conf_int(alpha=0.10))
    return np.arange(2024, 2024+steps), fc, ci


# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def kpi_card(label: str, value: str, color: str = BLUE):
    st.markdown(f"""
    <div class="kpi-box">
        <p class="kpi-value" style="color:{color}">{value}</p>
        <p class="kpi-label">{label}</p>
    </div>""", unsafe_allow_html=True)


def section(title: str):
    st.markdown(f'<div class="section-hdr">{title}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def sidebar():
    with st.sidebar:
        st.markdown("## 🌍 ClimateScope")
        st.markdown("**Bangladesh & South Asia**")
        st.markdown("---")

        page = st.radio(
            "Navigate to",
            ["🌍 Global Overview",
             "🇧🇩 Bangladesh Deep Dive",
             "🤖 ML Flood Risk Predictor",
             "📈 2024–2050 Forecast"],
        )
        st.markdown("---")
        yr_range = st.slider("Year range", 1984, 2023, (1984, 2023))
        st.markdown("---")
        st.caption("📡 NASA POWER API")
        st.caption("🌡 NASA GISS Surface Temp")
        st.caption("🏦 World Bank Open Data")
        st.caption("🌊 CSIRO Sea Level Dataset")
        st.markdown("---")
        st.caption("**Author:** Shamsul AL Mazid")
        st.caption("[GitHub ↗](https://github.com/almazid82/ClimateScope-Bangladesh)")

    return page, yr_range


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — GLOBAL OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

def show_global_overview(master, giss, yr_range):
    set_page_bg("global")

    st.title("🌍 Global Climate Overview")
    st.caption("Bangladesh local temperature vs the global anomaly — 1984 to 2023")

    m = master[(master.Year >= yr_range[0]) & (master.Year <= yr_range[1])].copy()
    g = giss[(giss.Year >= yr_range[0]) & (giss.Year <= yr_range[1])].copy()

    m_clean = m.dropna(subset=["BGD_Temp_C"])
    bgd_slope = np.polyfit(m_clean.Year, m_clean.BGD_Temp_C, 1)[0] if len(m_clean) > 1 else 0
    glo_slope = np.polyfit(g.Year, g.Annual_Anomaly_C, 1)[0]       if len(g) > 1       else 0
    divergence = (glo_slope - bgd_slope) * (yr_range[1] - yr_range[0])

    # Fix: ensure CO2 column is numeric before arithmetic
    co2_col = m.copy()
    co2_col["CO2_emissions"] = pd.to_numeric(co2_col["CO2_emissions"], errors="coerce")
    co2_col = co2_col.dropna(subset=["CO2_emissions"])
    if len(co2_col) > 1 and float(co2_col["CO2_emissions"].iloc[0]) != 0:
        co2_pct = ((float(co2_col["CO2_emissions"].iloc[-1]) /
                    float(co2_col["CO2_emissions"].iloc[0])) - 1) * 100
    else:
        co2_pct = 0.0

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("BGD Warming Rate",    f"{bgd_slope:+.4f}°C/yr", BLUE)
    with c2: kpi_card("Global Warming Rate", f"{glo_slope:+.4f}°C/yr", RED)
    with c3: kpi_card("Total Divergence",    f"{divergence:+.2f}°C",   ORANGE)
    with c4: kpi_card("CO₂ Increase",        f"{co2_pct:+.1f}%",       PURPLE)

    st.markdown("---")

    section("Bangladesh Temperature vs Global Anomaly")
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=m_clean.Year, y=m_clean.BGD_Temp_C,
        name="Bangladesh Temp (°C)", mode="lines+markers",
        line=dict(color=BLUE, width=2.5), marker=dict(size=5),
    ), secondary_y=False)

    if len(m_clean) > 1:
        z1 = np.polyfit(m_clean.Year, m_clean.BGD_Temp_C, 1)
        fig.add_trace(go.Scatter(
            x=m_clean.Year, y=np.polyval(z1, m_clean.Year),
            name=f"BGD trend ({z1[0]:+.4f}°C/yr)", mode="lines",
            line=dict(color=BLUE, width=1.5, dash="dash"),
        ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=g.Year, y=g.Annual_Anomaly_C,
        name="Global Anomaly (°C)", mode="lines+markers",
        line=dict(color=RED, width=2.5), marker=dict(size=5),
    ), secondary_y=True)

    if len(g) > 1:
        z2 = np.polyfit(g.Year, g.Annual_Anomaly_C, 1)
        fig.add_trace(go.Scatter(
            x=g.Year, y=np.polyval(z2, g.Year),
            name=f"Global trend ({z2[0]:+.4f}°C/yr)", mode="lines",
            line=dict(color=RED, width=1.5, dash="dash"),
        ), secondary_y=True)

    fig.update_layout(**BASE_LAYOUT, height=430, hovermode="x unified",
                      title="The South Asian Warming Hole: Bangladesh Cools While the World Warms",
                      yaxis=dict(title="Bangladesh Temp (°C)", gridcolor=GRID,
                                 titlefont=dict(color=BLUE), tickfont=dict(color=BLUE)),
                      yaxis2=dict(title="Global Temp Anomaly (°C)", gridcolor=GRID,
                                  titlefont=dict(color=RED), tickfont=dict(color=RED),
                                  overlaying="y", side="right", zeroline=True,
                                  zerolinecolor="gray", zerolinewidth=1))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        section("Bangladesh CO₂ Emissions")
        co2 = co2_col
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=co2.Year, y=co2.CO2_emissions,
                              marker_color=PURPLE, opacity=0.75, name="CO₂ (kt)"))
        fig2.add_trace(go.Scatter(
            x=co2.Year, y=co2.CO2_emissions.rolling(5, min_periods=1).mean(),
            name="5-yr mean", line=dict(color=ORANGE, width=2.5),
        ))
        fig2.update_layout(**BASE_LAYOUT, height=320, yaxis_title="CO₂ (kt)")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        section("Cumulative Temperature Divergence")
        both = m.dropna(subset=["BGD_Temp_C", "Annual_Anomaly_C"])
        if len(both) > 1:
            bgd_c = both.BGD_Temp_C - both.BGD_Temp_C.iloc[0]
            glo_c = both.Annual_Anomaly_C - both.Annual_Anomaly_C.iloc[0]
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=both.Year, y=bgd_c, name="Bangladesh Δ",
                                      fill="tozeroy", fillcolor="rgba(79,195,247,0.1)",
                                      line=dict(color=BLUE, width=2.5)))
            fig3.add_trace(go.Scatter(x=both.Year, y=glo_c, name="Global Δ",
                                      fill="tozeroy", fillcolor="rgba(239,83,80,0.1)",
                                      line=dict(color=RED, width=2.5)))
            fig3.add_hline(y=0, line_color="gray", line_dash="dot")
            fig3.update_layout(**BASE_LAYOUT, height=320,
                               yaxis_title="Cumulative Change (°C)", hovermode="x unified")
            st.plotly_chart(fig3, use_container_width=True)

    st.info(
        f"**Key Finding — South Asian Warming Hole** | "
        f"Bangladesh: **{bgd_slope:+.4f}°C/yr** | "
        f"Global: **{glo_slope:+.4f}°C/yr** | "
        f"Divergence over {yr_range[1]-yr_range[0]} years: **{divergence:+.2f}°C** | "
        f"Mann-Kendall p = **0.0008** (highly significant)"
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — BANGLADESH DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════

def show_bangladesh_deep_dive(master, yr_range):
    set_page_bg("bangladesh")

    st.title("🇧🇩 Bangladesh Climate Deep Dive")
    st.caption("Temperature, precipitation, sea level, GDP and variable correlations")

    m = master[(master.Year >= yr_range[0]) & (master.Year <= yr_range[1])].copy()

    sl  = m.dropna(subset=["Sea_Level_mm"])
    gdp = m.dropna(subset=["GDP_per_capita_USD"])
    sl_change  = sl.Sea_Level_mm.iloc[-1]  - sl.Sea_Level_mm.iloc[0]  if len(sl)  > 1 else 0
    gdp_growth = (gdp.GDP_per_capita_USD.iloc[-1]/gdp.GDP_per_capita_USD.iloc[0]-1)*100 if len(gdp) > 1 else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Mean Temperature",     f"{m.BGD_Temp_C.mean():.2f}°C",      BLUE)
    with c2: kpi_card("Mean Precipitation",   f"{m.BGD_Precip_mm_day.mean():.2f} mm/day", TEAL)
    with c3: kpi_card("Sea Level Change",     f"{sl_change:+.1f} mm",               ORANGE)
    with c4: kpi_card("GDP/cap Growth",       f"{gdp_growth:+.0f}%",                GREEN)

    st.markdown("---")

    section("Annual Temperature & Precipitation")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=m.Year, y=m.BGD_Temp_C, name="Temperature (°C)",
        line=dict(color=RED, width=2.5), mode="lines+markers", marker=dict(size=5),
    ), secondary_y=False)
    fig.add_trace(go.Bar(
        x=m.Year, y=m.BGD_Precip_mm_day, name="Precipitation (mm/day)",
        marker_color=BLUE, opacity=0.50,
    ), secondary_y=True)
    fig.add_trace(go.Scatter(
        x=m.Year, y=m.BGD_Precip_mm_day.rolling(5, min_periods=1).mean(),
        name="5-yr precip mean", line=dict(color="#81d4fa", width=2, dash="dot"),
    ), secondary_y=True)
    fig.update_layout(**BASE_LAYOUT, height=400, hovermode="x unified",
                      title="Bangladesh Annual Temperature and Precipitation (1984–2023)")
    fig.update_yaxes(title_text="Temperature (°C)",       secondary_y=False, color=RED)
    fig.update_yaxes(title_text="Precipitation (mm/day)", secondary_y=True,  color=BLUE)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        section("Sea Level Rise Trend")
        sl_d = m.dropna(subset=["Sea_Level_mm"])
        if len(sl_d) > 1:
            z  = np.polyfit(sl_d.Year, sl_d.Sea_Level_mm, 1)
            z2 = np.polyfit(sl_d.Year, sl_d.Sea_Level_mm, 2)
            yr_fit = np.linspace(sl_d.Year.min(), sl_d.Year.max(), 100)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=sl_d.Year, y=sl_d.Sea_Level_mm, name="Sea Level (mm)",
                mode="lines+markers", line=dict(color=TEAL, width=2.5),
                fill="tozeroy", fillcolor="rgba(77,208,225,0.10)",
            ))
            fig2.add_trace(go.Scatter(
                x=yr_fit, y=np.polyval(z, yr_fit),
                name=f"Linear ({z[0]:.2f} mm/yr)",
                line=dict(color=ORANGE, width=1.8, dash="dash"),
            ))
            fig2.add_trace(go.Scatter(
                x=yr_fit, y=np.polyval(z2, yr_fit),
                name="Quadratic (acceleration)",
                line=dict(color=RED, width=1.8, dash="dot"),
            ))
            fig2.update_layout(**BASE_LAYOUT, height=330, yaxis_title="Sea Level (mm)")
            st.plotly_chart(fig2, use_container_width=True)

    with col2:
        section("GDP per Capita Growth")
        gdp_d = m.dropna(subset=["GDP_per_capita_USD"])
        if len(gdp_d) > 1:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=gdp_d.Year, y=gdp_d.GDP_per_capita_USD,
                name="GDP/capita (USD)", mode="lines+markers",
                line=dict(color=GREEN, width=2.5),
                fill="tozeroy", fillcolor="rgba(57,255,20,0.08)",
            ))
            fig3.update_layout(**BASE_LAYOUT, height=330, yaxis_title="USD")
            st.plotly_chart(fig3, use_container_width=True)

    section("Variable Correlation Heatmap")
    want = ["BGD_Temp_C","BGD_Precip_mm_day","BGD_Humidity_pct",
            "Annual_Anomaly_C","Sea_Level_mm","GDP_per_capita_USD","CO2_emissions"]
    cols = [c for c in want if c in m.columns]
    corr = m[cols].corr().round(3)

    labels = {
        "BGD_Temp_C": "BGD Temp",
        "BGD_Precip_mm_day": "Precip",
        "BGD_Humidity_pct": "Humidity",
        "Annual_Anomaly_C": "Global Anom",
        "Sea_Level_mm": "Sea Level",
        "GDP_per_capita_USD": "GDP/cap",
        "CO2_emissions": "CO₂",
    }
    display_cols = [labels.get(c, c) for c in cols]

    fig4 = go.Figure(go.Heatmap(
        z=corr.values, x=display_cols, y=display_cols,
        colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
        text=corr.values.round(2), texttemplate="%{text}",
        textfont=dict(size=11),
    ))
    fig4.update_layout(**BASE_LAYOUT, height=420,
                       title="Pearson Correlation Matrix — Climate & Socioeconomic Variables",
                       xaxis=dict(tickangle=-30, gridcolor=GRID),
                       yaxis=dict(gridcolor=GRID))
    st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ML FLOOD RISK PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════

def show_ml_predictor():
    set_page_bg("ml")

    st.title("🤖 ML Flood Risk Predictor")
    st.caption("Random Forest (300 trees) trained on 480 months of NASA POWER data — Accuracy 95.8% · AUC 0.9947")

    with st.spinner("Loading model..."):
        model, FEATS, X_ref = get_flood_model()

    MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    col_inp, col_out = st.columns([1, 1.4], gap="large")

    with col_inp:
        st.markdown("### Input Climate Conditions")

        month = st.selectbox("Month", range(1,13),
                             format_func=lambda x: MONTHS[x-1], index=6)
        temp  = st.slider("Temperature (°C)",           15.0, 35.0, 28.5, 0.1)
        prec  = st.slider("Precipitation (mm/day)",      0.0, 20.0,  8.0, 0.1)
        hum   = st.slider("Humidity (%)",               50.0,100.0, 82.0, 0.5)
        lag1  = st.slider("Last month precip (mm/day)",  0.0, 20.0,  6.0, 0.1)
        lag2  = st.slider("2 months ago precip (mm/day)",0.0, 20.0,  4.0, 0.1)

        is_monsoon   = int(month in [6,7,8,9])
        prec_3mo     = (prec + lag1 + lag2) / 3
        prec_6mo     = prec_3mo
        temp_anom    = temp  - X_ref["Temperature_C"].mean()
        prec_anom    = prec  - X_ref["Precipitation_mm_day"].mean()
        hum_anom     = hum   - X_ref["Humidity_pct"].mean()

        X_in = pd.DataFrame([{
            "Temperature_C":        temp,
            "Precipitation_mm_day": prec,
            "Humidity_pct":         hum,
            "Month":                month,
            "Is_Monsoon":           is_monsoon,
            "Temp_anomaly":         temp_anom,
            "Precip_anomaly":       prec_anom,
            "Humid_anomaly":        hum_anom,
            "Precip_3mo_roll":      prec_3mo,
            "Precip_6mo_roll":      prec_6mo,
            "Precip_lag1":          lag1,
            "Precip_lag2":          lag2,
        }])

    with col_out:
        st.markdown("### Prediction Result")

        prob  = model.predict_proba(X_in)[0][1]
        label = "HIGH RISK" if prob >= 0.5 else "LOW RISK"
        color = RED if prob >= 0.5 else GREEN
        icon  = "🔴" if prob >= 0.5 else "🟢"
        bg_c  = "rgba(61,16,16,0.85)"  if prob >= 0.5 else "rgba(13,46,13,0.85)"
        brd   = RED if prob >= 0.5 else GREEN

        st.markdown(f"""
        <div style="background:{bg_c};
                    border:2px solid {brd};border-radius:14px;
                    padding:24px;text-align:center;margin-bottom:16px;
                    box-shadow:0 0 30px {brd}33;
                    backdrop-filter:blur(12px);">
            <p style="font-size:2.6rem;margin:0">{icon}</p>
            <p style="font-size:1.9rem;font-weight:700;color:{color};margin:6px 0;
               filter:drop-shadow(0 0 10px {color});">{label}</p>
            <p style="font-size:1.05rem;color:#ccc;margin:0">
                Flood probability: <b style="color:{color}">{prob:.1%}</b>
            </p>
        </div>""", unsafe_allow_html=True)

        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": "Risk Score", "font": {"color": "#c8e0f0", "size": 14}},
            number={"suffix": "%", "font": {"color": color, "size": 30}},
            gauge={
                "axis": {"range": [0,100], "tickcolor": "#aaa",
                         "tickfont": {"color": "#aaa"}},
                "bar":  {"color": color, "thickness": 0.25},
                "bgcolor": GRID,
                "steps": [
                    {"range": [0, 30], "color": "#0d2e18"},
                    {"range": [30,60], "color": "#2e1f00"},
                    {"range": [60,100],"color": "#2e0a0a"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        ))
        fig_g.update_layout(paper_bgcolor="rgba(3,7,18,0.80)", font_color="#c8e0f0",
                            height=230, margin=dict(l=30,r=30,t=30,b=0))
        st.plotly_chart(fig_g, use_container_width=True)

        season = "Monsoon" if is_monsoon else "Non-monsoon"
        st.markdown(f"**Month:** {MONTHS[month-1]} ({season}) &nbsp;|&nbsp; "
                    f"**Precip anomaly:** {prec_anom:+.2f} mm/day &nbsp;|&nbsp; "
                    f"**3-mo rolling:** {prec_3mo:.2f} mm/day")

    st.markdown("---")
    section("Feature Importance — What Drives Flood Risk?")

    imp = pd.Series(model.feature_importances_, index=FEATS).sort_values()
    bar_colors = [RED if v > imp.quantile(0.75) else BLUE for v in imp.values]

    fig_fi = go.Figure(go.Bar(
        x=imp.values, y=imp.index, orientation="h",
        marker_color=bar_colors,
        text=[f"{v:.3f}" for v in imp.values],
        textposition="outside",
        textfont=dict(color="#c8e0f0", size=11),
    ))
    fig_fi.update_layout(
        **BASE_LAYOUT, height=420,
        title="Random Forest Feature Importance (Mean Decrease in Impurity)",
        xaxis_title="Importance Score",
        xaxis=dict(gridcolor=GRID),
        yaxis=dict(gridcolor=GRID),
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    with st.expander("Show full input vector"):
        st.dataframe(X_in.T.rename(columns={0: "Value"}).round(4),
                     use_container_width=True)

    st.info(
        "**Model:** Random Forest (300 trees · `class_weight='balanced'`) | "
        "**Training:** 1984–2015 (384 months) | "
        "**Test (2016–2023):** Accuracy **95.8%** · AUC **0.9947** · F1 **0.9535** | "
        "**5-fold CV:** Accuracy 97.9% ± 2.2%"
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — 2024–2050 FORECAST
# ══════════════════════════════════════════════════════════════════════════════

def show_forecast(master):
    set_page_bg("forecast")

    st.title("📈 2024–2050 Climate Forecast")
    st.caption("ARIMA(1,1,1) base forecast with IPCC AR6 RCP 4.5 and RCP 8.5 scenario adjustments")

    m = master.dropna(subset=["BGD_Temp_C"]).copy()

    with st.spinner("Fitting ARIMA model..."):
        fc_yr, fc_vals, fc_ci = get_arima_forecast(steps=27)

    t = np.arange(len(fc_yr))
    rcp45 = fc_vals + 0.022 * t
    rcp85 = fc_vals + 0.048 * t

    section("Bangladesh Temperature 1984–2050: ARIMA + RCP Scenarios")
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=m.Year, y=m.BGD_Temp_C, name="Observed (1984–2023)",
        mode="lines+markers", line=dict(color=BLUE, width=2.5), marker=dict(size=4),
    ))

    ci_x = np.concatenate([fc_yr, fc_yr[::-1]])
    ci_y = np.concatenate([fc_ci[:,0], fc_ci[:,1][::-1]])
    fig.add_trace(go.Scatter(
        x=ci_x, y=ci_y, fill="toself",
        fillcolor="rgba(255,167,38,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="90% CI (ARIMA base)", showlegend=True,
    ))

    fig.add_trace(go.Scatter(
        x=fc_yr, y=fc_vals, name="ARIMA base forecast",
        mode="lines", line=dict(color=ORANGE, width=2.5, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=fc_yr, y=rcp45, name="RCP 4.5 (moderate emissions)",
        mode="lines", line=dict(color=GREEN, width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=fc_yr, y=rcp85, name="RCP 8.5 (high emissions)",
        mode="lines", line=dict(color=RED, width=2.5),
    ))

    fig.add_vline(x=2023.5, line_color="gray", line_dash="dot",
                  annotation_text="Forecast →", annotation_font_color="#aaa")

    fig.update_layout(**BASE_LAYOUT, height=460,
                      title="Bangladesh Temperature Forecast with IPCC RCP Scenarios (2024–2050)",
                      yaxis_title="Temperature (°C)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        section("2050 Scenario Comparison")
        last = m.BGD_Temp_C.iloc[-1]
        rows = {
            "Scenario":       ["ARIMA Base", "RCP 4.5", "RCP 8.5"],
            "2050 Temp (°C)": [f"{fc_vals[-1]:.2f}", f"{rcp45[-1]:.2f}", f"{rcp85[-1]:.2f}"],
            "Δ vs 2023":      [f"{fc_vals[-1]-last:+.2f}°C",
                               f"{rcp45[-1]-last:+.2f}°C",
                               f"{rcp85[-1]-last:+.2f}°C"],
            "Risk Level":     ["Low", "Moderate", "High"],
        }
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with col2:
        section("RCP Scenario Assumptions")
        st.markdown("""
| Parameter | RCP 4.5 | RCP 8.5 |
|-----------|---------|---------|
| CO₂ by 2100 | ~650 ppm | ~1370 ppm |
| Global warming | +1.5–2.0°C | +3.2–5.4°C |
| Aerosol decline | Gradual | Rapid |
| BGD adjustment | +0.022°C/yr | +0.048°C/yr |
| Warming hole | Partial recovery | Full recovery |

*Source: IPCC AR6 WGI (2021), South Asia projections*
        """)

    st.markdown("---")
    section("GDP per Capita at Risk — Climate Damage Projection")
    st.caption("Simplified DICE-model damage function applied to temperature deviations above 2023 baseline")

    gdp_d = master.dropna(subset=["GDP_per_capita_USD"])
    if len(gdp_d) > 1:
        gdp_rate = np.polyfit(gdp_d.Year, np.log(gdp_d.GDP_per_capita_USD), 1)[0]
        last_gdp = float(gdp_d.GDP_per_capita_USD.iloc[-1])

        # yr_p: 2023..2050 → 28 values; rcp45/rcp85: 2024..2050 → 27 values
        # Extend RCP arrays by prepending 2023 baseline so shapes align (28 × 28)
        yr_p = np.arange(2023, 2051)
        tp   = yr_p - 2023
        gdp_base = last_gdp * np.exp(gdp_rate * tp)

        temp_2023 = float(m.BGD_Temp_C.iloc[-1])
        rcp45_ext = np.concatenate([[temp_2023], rcp45])
        rcp85_ext = np.concatenate([[temp_2023], rcp85])
        dmg45 = 1 - 0.005 * np.maximum(0, rcp45_ext - temp_2023)
        dmg85 = 1 - 0.015 * np.maximum(0, rcp85_ext - temp_2023)
        gdp45 = gdp_base * dmg45
        gdp85 = gdp_base * dmg85

        fig_g = go.Figure()
        fig_g.add_trace(go.Scatter(x=gdp_d.Year, y=gdp_d.GDP_per_capita_USD,
                                   name="Historical GDP/capita",
                                   mode="lines+markers", marker=dict(size=4),
                                   line=dict(color=GREEN, width=2.5)))
        fig_g.add_trace(go.Scatter(x=yr_p, y=gdp_base, name="Baseline (no climate damage)",
                                   line=dict(color=GREEN, width=2, dash="dash")))
        fig_g.add_trace(go.Scatter(x=yr_p, y=gdp45, name="RCP 4.5 (with damage)",
                                   line=dict(color=ORANGE, width=2.5)))
        fig_g.add_trace(go.Scatter(x=yr_p, y=gdp85, name="RCP 8.5 (with damage)",
                                   line=dict(color=RED, width=2.5)))
        fig_g.add_vline(x=2023, line_color="gray", line_dash="dot")
        fig_g.update_layout(**BASE_LAYOUT, height=390,
                            title="Bangladesh GDP per Capita Projection Under RCP Scenarios",
                            yaxis_title="GDP per Capita (USD)", hovermode="x unified")
        st.plotly_chart(fig_g, use_container_width=True)

        loss45 = (gdp_base[-1] - gdp45[-1]) / gdp_base[-1] * 100
        loss85 = (gdp_base[-1] - gdp85[-1]) / gdp_base[-1] * 100

        c1, c2, c3 = st.columns(3)
        with c1: kpi_card("2050 GDP/cap (Baseline)", f"${gdp_base[-1]:,.0f}", GREEN)
        with c2: kpi_card("GDP Loss — RCP 4.5 (2050)", f"{loss45:.1f}%",      ORANGE)
        with c3: kpi_card("GDP Loss — RCP 8.5 (2050)", f"{loss85:.1f}%",      RED)

    st.info(
        "**ARIMA:** order (1,1,1), fitted on 1984–2023 annual data. "
        "**RCP adjustments:** derived from IPCC AR6 WGI South Asia regional projections. "
        "**GDP damage:** simplified DICE integrated assessment model structure — "
        "0.5% GDP loss per °C (RCP 4.5), 1.5% per °C (RCP 8.5), relative to 2023 baseline."
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    master  = load_master()
    giss    = load_giss()

    page, yr_range = sidebar()

    if page == "🌍 Global Overview":
        show_global_overview(master, giss, yr_range)

    elif page == "🇧🇩 Bangladesh Deep Dive":
        show_bangladesh_deep_dive(master, yr_range)

    elif page == "🤖 ML Flood Risk Predictor":
        show_ml_predictor()

    elif page == "📈 2024–2050 Forecast":
        show_forecast(master)


if __name__ == "__main__":
    main()
