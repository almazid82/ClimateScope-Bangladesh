# ClimateScope Bangladesh & South Asia
### Data-Driven Climate Risk Analysis | 1984–2023

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://climatescope-bangladesh-tha93d72fvtx5gfchsmqqx.streamlit.app/)

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)
![Notebooks](https://img.shields.io/badge/Notebooks-5-orange?style=flat-square&logo=jupyter)
![Figures](https://img.shields.io/badge/Figures-30-purple?style=flat-square)

A rigorous, end-to-end climate data science project applying statistical modeling, machine learning, and explainable AI to 40 years of Bangladesh climate data. Built to demonstrate research-grade data science skills for graduate-level applications.

---

## Key Finding: The South Asian Warming Hole

> **Bangladesh temperature decreased at −0.026°C/year** (Mann-Kendall p < 0.001) over 1984–2023, while the global temperature anomaly rose at +0.021°C/year — a **1.88°C divergence** over four decades.

This counterintuitive phenomenon, driven by aerosol loading and intensified monsoon cloud cover, is documented and analyzed across all five phases of this project.

![Climate Paradox](outputs/figures/phase5_climate_paradox.png)

---

## Project Overview

| Phase | Topic | Methods | Key Output |
|-------|-------|---------|-----------|
| **1** | Data Collection & Cleaning | NASA POWER API, World Bank API, Sea Level datasets | 40-year master dataset (480 monthly rows) |
| **2** | Exploratory Data Analysis | Mann-Kendall trend test, Folium maps, Correlation analysis | South Asian Warming Hole discovery |
| **3** | Statistical Modeling | ARIMA, SARIMA, ANOVA + Tukey HSD, OLS regression | Temperature forecast 2024–2030 |
| **4** | Machine Learning | Random Forest, SHAP values, Cross-validation | 95.8% flood risk prediction accuracy |
| **5** | Visualization & Storytelling | Dashboard, CVI index, Policy brief | Publication-ready figures |

---

## Results

### Machine Learning — Flood Risk Prediction

| Metric | Value |
|--------|-------|
| Test Accuracy | **95.8%** |
| ROC-AUC | **0.9947** |
| F1 Score | **0.9535** |
| CV Accuracy (5-fold) | **97.9% ± 2.2%** |
| CV ROC-AUC (5-fold) | **99.9% ± 0.2%** |

### Statistical Modeling

| Test | Result | Interpretation |
|------|--------|---------------|
| Mann-Kendall (Temperature) | p = 0.0008, τ = −0.26 | Significant **downward** trend |
| Mann-Kendall (Precipitation) | p = 0.031 | Significant upward trend |
| One-Way ANOVA (Decades) | F = 11.12, p < 0.001 | Significant inter-decade difference |
| OLS Bangladesh Model | R² = 0.814 | GDP, sea level, precip explain 81.4% of temp variance |
| ARIMA Temperature Forecast | AIC-optimized | 2024–2030 projection generated |

### SHAP Explainability — Top Flood Drivers

![SHAP Summary](outputs/figures/phase4_shap_summary.png)

---

## Dashboard

![Master Dashboard](outputs/figures/phase5_master_dashboard.png)

---

## Flood Risk Calendar

Monthly flood risk across all 40 years — red = high risk, blue = low risk. September is consistently the peak risk month.

![Flood Risk Heatmap](outputs/figures/phase5_flood_risk_heatmap.png)

---

## Climate Vulnerability Index

Composite vulnerability score combining sea level exposure, precipitation variability, GDP (adaptive capacity), and under-5 mortality — inspired by IPCC AR6 framework.

![Vulnerability Index](outputs/figures/phase5_vulnerability_index.png)

---

## Policy Recommendations

Five evidence-based interventions derived directly from statistical findings:

![Policy Recommendations](outputs/figures/phase5_policy_recommendations.png)

---

## Data Sources

| Source | Data | Coverage |
|--------|------|----------|
| [NASA POWER API](https://power.larc.nasa.gov/) | Temperature, Precipitation, Humidity | 1984–2023 (monthly) |
| [NASA GISS](https://data.giss.nasa.gov/gistemp/) | Global surface temperature anomaly | 1880–2023 |
| [World Bank API](https://data.worldbank.org/) | GDP, Population, Agricultural land, CO₂ | 1960–2023 |
| [CSIRO Sea Level Dataset](https://github.com/datasets/sea-level-rise) | Global mean sea level | 1880–2013 |
| Global CO₂ Emissions | Fossil fuel CO₂ (kt) | 1960–2020 |

---

## Project Structure

```
ClimateScope-Bangladesh/
│
├── notebooks/
│   ├── 01_data_collection_cleaning.ipynb     # Phase 1: APIs + master dataset
│   ├── 02_exploratory_data_analysis.ipynb    # Phase 2: EDA + Mann-Kendall
│   ├── 03_statistical_modeling.ipynb         # Phase 3: ARIMA, ANOVA, OLS
│   ├── 04_machine_learning.ipynb             # Phase 4: Random Forest + SHAP
│   └── 05_visualization_storytelling.ipynb   # Phase 5: Dashboard + Policy
│
├── data/
│   ├── raw/                                  # Original API downloads
│   └── processed/                            # Cleaned, merged datasets
│       └── master_annual_dataset.csv         # 40 rows × 12 columns
│
├── outputs/
│   └── figures/                              # 30 publication-quality figures
│
└── requirements.txt
```

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/almazid82/ClimateScope-Bangladesh.git
cd ClimateScope-Bangladesh
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run notebooks in order**
```
01 → 02 → 03 → 04 → 05
```

> **Note:** Phase 1 requires internet access to download from NASA POWER and World Bank APIs. Phases 2–5 use the cached data in `data/processed/`.

---

## Requirements

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
statsmodels>=0.14.0
pmdarima>=2.0.0
scipy>=1.10.0
folium>=0.14.0
wbgapi>=1.0.12
pymannkendall>=1.4.3
shap>=0.47.0
requests>=2.28.0
nbformat>=5.9.0
jupyter>=1.0.0
```

---

## Scientific Context

The **South Asian Warming Hole** is a documented regional cooling anomaly in the northern Indian subcontinent. While the global mean surface temperature has risen by approximately +0.85°C since 1984, Bangladesh experienced a −1.03°C change over the same period. Leading explanations include:

- **Aerosol forcing** from rapid industrialisation across the Indo-Gangetic Plain
- **Increased cloud cover** during the intensifying monsoon season
- **Land-use change** — agricultural expansion increasing latent heat flux (evapotranspiration cooling)

Despite local cooling, **climate risk is rising** through sea level acceleration, extreme precipitation events, and increasing economic exposure. This project quantifies both dimensions.

---

## Author

**Shamsul AL Mazid**
BSc Statistics — Bangladesh

[![GitHub](https://img.shields.io/badge/GitHub-almazid82-black?style=flat-square&logo=github)](https://github.com/almazid82)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/shamsul-al-mazid-bb6068298)

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
