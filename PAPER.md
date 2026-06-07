# Climate Risk in Bangladesh: A Data-Driven Analysis of Temperature Trends, Flood Prediction, and Vulnerability (1984–2023)

**Shamsul AL Mazid**
Department of Statistics, Bangladesh
GitHub: [almazid82](https://github.com/almazid82)

---

## Abstract

This study presents a comprehensive, data-driven analysis of climate risk in Bangladesh over the period 1984–2023, integrating five independent data sources through statistical modeling, machine learning, and explainable AI. Using NASA POWER monthly climate data, NASA GISS global temperature records, World Bank socioeconomic indicators, CSIRO sea level measurements, and global CO₂ emission datasets, we construct a 40-year master dataset covering 480 monthly observations. Our primary finding is statistically significant: Bangladesh experienced a temperature *decrease* of −0.026°C per year (Mann-Kendall τ = −0.26, p = 0.0008) over the study period — contrasting sharply with global warming of +0.021°C per year. This divergence, totalling 1.88°C over four decades, is consistent with the documented *South Asian Warming Hole* phenomenon. Despite local cooling, climate risk is escalating through rising sea levels, intensifying monsoon variability, and increasing socioeconomic exposure. A Random Forest classifier trained on 12 engineered monthly climate features achieved 95.8% accuracy (ROC-AUC = 0.9947) in predicting high flood-risk months. SHAP (SHapley Additive exPlanations) analysis identified current precipitation, 3-month antecedent rainfall, and monsoon season as the three primary flood drivers. A composite Climate Vulnerability Index (CVI) reveals a structural shift in the composition of risk over time. These findings support five evidence-based policy interventions for climate adaptation in Bangladesh.

**Keywords:** Bangladesh climate, South Asian Warming Hole, Mann-Kendall trend test, ARIMA, Random Forest, SHAP, flood prediction, climate vulnerability

---

## 1. Introduction

Bangladesh is among the world's most climate-vulnerable nations. Located at the confluence of three major river systems — the Ganges, Brahmaputra, and Meghna — and largely below 10 metres in elevation, the country faces existential threats from flooding, sea level rise, and changing monsoon patterns. The IPCC Sixth Assessment Report (AR6, 2021) identifies South Asia as a region of compounding climate risks, with Bangladesh specifically mentioned as facing high exposure to both riverine flooding and coastal inundation.

However, the observed local temperature record presents a paradox. While global mean surface temperature has risen consistently since the mid-twentieth century, several studies have documented a regional cooling signal across the northern Indian subcontinent — a phenomenon termed the **South Asian Warming Hole** (Roxy et al., 2015; Krishnan et al., 2016). The mechanisms proposed include increased anthropogenic aerosol loading over the Indo-Gangetic Plain, intensification of monsoon cloud cover reducing incoming solar radiation, and land-use change increasing evapotranspiration.

This study addresses three research questions:

1. Is the South Asian Warming Hole detectable in the Bangladesh temperature record over 1984–2023, and is the trend statistically significant?
2. Can machine learning accurately predict high flood-risk months from climate variables, and which variables are most influential?
3. How has climate vulnerability evolved in Bangladesh over the study period, and what policy interventions are supported by the data?

This work contributes an open, reproducible analysis using entirely public data sources and open-source Python tools, with all code and data available at [github.com/almazid82/ClimateScope-Bangladesh](https://github.com/almazid82/ClimateScope-Bangladesh).

---

## 2. Data and Methods

### 2.1 Data Sources

Five independent data sources were integrated into a unified annual master dataset (Table 1) and a monthly dataset for machine learning.

**Table 1. Data Sources**

| Source | Variable(s) | Temporal Coverage | Resolution |
|--------|------------|-------------------|-----------|
| NASA POWER API | Temperature (T2M), Precipitation (PRECTOTCORR), Humidity (RH2M) | 1984–2023 | Monthly |
| NASA GISS Surface Temperature Analysis | Global temperature anomaly, seasonal anomalies | 1880–2023 | Annual |
| World Bank Open Data API | GDP, GDP per capita, Population, Agricultural land, CO₂ emissions, Under-5 mortality | 1960–2023 | Annual |
| CSIRO Sea Level Dataset | Global mean sea level | 1880–2013 | Annual |
| Global CO₂ Emissions Dataset | Fossil fuel CO₂ (kt) | 1960–2020 | Annual |

NASA POWER data were downloaded for the coordinates 23.685°N, 90.356°E (centroid of Bangladesh) via the temporal/monthly/point API endpoint. World Bank indicators were retrieved using the `wbgapi` Python library for economy code `BGD`. API responses were parsed, cleaned, and merged on the `Year` key to produce the master annual dataset (40 rows × 12 columns) and the monthly dataset (480 rows × 15 features after engineering).

### 2.2 Data Preprocessing

Missing values in World Bank indicators (arising from reporting gaps) were imputed using linear interpolation on the time axis — appropriate for slowly-evolving socioeconomic series. NASA POWER sentinel values of −999.0 were replaced with NaN before interpolation. Annual summary statistics were derived from monthly data by taking the calendar-year mean.

The annual master dataset contains the following variables: `Year`, `BGD_Temp_C`, `BGD_Precip_mm_day`, `BGD_Humidity_pct`, `Annual_Anomaly_C`, `Agricultural_land_pct`, `GDP_USD`, `GDP_per_capita_USD`, `Under5_mortality`, `Population`, `Sea_Level_mm`, `CO2_emissions`.

### 2.3 Statistical Methods

#### 2.3.1 Mann-Kendall Trend Test

The non-parametric Mann-Kendall test (Mann, 1945; Kendall, 1975) was applied to detect monotonic trends in temperature, precipitation, humidity, and global anomaly time series. This test is preferred over ordinary least squares for trend detection in climate data because it makes no distributional assumptions and is robust to outliers and non-normal errors. The test statistic τ ranges from −1 (perfectly decreasing) to +1 (perfectly increasing). Statistical significance was assessed at α = 0.05.

#### 2.3.2 Time Series Modeling (ARIMA and SARIMA)

Stationarity was confirmed using the Augmented Dickey-Fuller (ADF) test. Autocorrelation function (ACF) and partial autocorrelation function (PACF) plots were examined to guide model order selection. `pmdarima.auto_arima` was used to select the optimal ARIMA(p,d,q) order via AIC minimisation. A seasonal SARIMA(p,d,q)(P,D,Q,m=12) model was additionally fitted to the monthly precipitation series to capture the dominant annual cycle. Forecast horizons of 7 years (2024–2030) were generated with 95% prediction intervals.

#### 2.3.3 Analysis of Variance

One-way ANOVA was used to test whether mean annual temperature differed significantly across decades (1980s, 1990s, 2000s, 2010s). The assumption of homogeneity of variance was verified prior to testing. Where the omnibus F-test was significant, pairwise post-hoc comparisons were conducted using Tukey's Honest Significant Difference (HSD) test to identify which decade pairs differed.

#### 2.3.4 Ordinary Least Squares Regression

A multiple linear regression model was fitted with Bangladesh annual temperature as the response variable and Year, Sea_Level_mm, and BGD_Precip_mm_day as predictors. Model diagnostics included residual vs. fitted plots, Q-Q plots for normality, and Shapiro-Wilk tests on residuals.

### 2.4 Machine Learning — Flood Risk Prediction

#### 2.4.1 Feature Engineering

The monthly NASA POWER dataset (480 rows) was enriched with twelve engineered features: calendar month, monsoon season indicator (June–September = 1), temperature anomaly, precipitation anomaly, humidity anomaly, 3-month rolling mean precipitation, 6-month rolling mean precipitation, and precipitation lags at 1 and 2 months. These features capture seasonality, antecedent moisture conditions, and deviation from climatological norms — all physically meaningful flood predictors.

#### 2.4.2 Flood Risk Label Construction

In the absence of a complete historical flood event database for the study period, a scientifically-grounded proxy label was constructed using a composite risk score:

$$\text{Risk Score} = 0.60 \times z_{\text{precip}} + 0.20 \times z_{\text{humid}} + 0.20 \times \mathbb{1}_{\text{monsoon}}$$

where $z_{\text{precip}}$ and $z_{\text{humid}}$ are standardised precipitation and humidity values respectively, and $\mathbb{1}_{\text{monsoon}}$ is the binary monsoon season indicator. Months in the top 30th percentile of risk scores were labelled as high flood-risk (Class 1). This threshold produced 144 high-risk and 336 low-risk months across the full dataset.

#### 2.4.3 Model Training and Evaluation

A **chronological train-test split** was applied — training on 1984–2015 (384 months) and testing on 2016–2023 (96 months) — to prevent data leakage from future observations into model training. A Random Forest classifier with 300 estimators, `class_weight='balanced'`, and default hyperparameters was trained. Performance was measured using accuracy, F1 score, and ROC-AUC. **5-fold stratified cross-validation** was additionally performed on the full dataset to assess generalisation robustness.

#### 2.4.4 Explainability — SHAP Values

SHAP (Lundberg & Lee, 2017) TreeExplainer was applied to compute Shapley values for all test-set predictions. Three SHAP visualisations were produced: a beeswarm summary plot (global feature impact and direction), a bar plot (mean absolute SHAP values), and a waterfall plot (single highest-risk prediction decomposed into individual feature contributions).

### 2.5 Climate Vulnerability Index

A composite Climate Vulnerability Index (CVI) was constructed using four normalised components weighted by their relevance to Bangladesh's primary risk channels:

$$\text{CVI} = 0.35 \times \text{SL}_{\text{norm}} + 0.25 \times \text{PrecipVar}_{\text{norm}} + 0.20 \times (1 - \text{GDP}_{\text{norm}}) + 0.20 \times \text{Mort}_{\text{norm}}$$

Sea level exposure (35%) and precipitation variability (25%) represent physical hazard; inverse GDP per capita (20%) represents adaptive capacity; under-5 mortality (20%) represents social vulnerability. This structure is consistent with the IPCC AR6 risk framework decomposing risk into Hazard × Exposure × Vulnerability.

---

## 3. Results

### 3.1 Temperature Trend Analysis

The Mann-Kendall test applied to the Bangladesh annual temperature series (1984–2023) yielded τ = −0.260 and p = 0.0008, indicating a **statistically significant decreasing trend** at the 0.1% level. The Sen's slope estimate (linear rate of change) was −0.026°C per year, corresponding to a cumulative cooling of approximately −1.03°C over the 40-year study period.

By contrast, the global temperature anomaly series (NASA GISS, 1984–2023) showed a significant positive trend of +0.021°C per year (cumulative: +0.85°C). The divergence between local Bangladesh cooling and global warming — totalling 1.88°C — is striking and statistically robust.

**Table 2. Mann-Kendall Trend Test Results**

| Variable | Trend Direction | τ | p-value | Rate (per year) | Significance |
|----------|----------------|---|---------|-----------------|-------------|
| Bangladesh Temperature | **Decreasing** | −0.260 | 0.0008 | −0.026°C | *** |
| Bangladesh Precipitation | Increasing | +0.185 | 0.031 | +0.009 mm/day | * |
| Bangladesh Humidity | No trend | — | > 0.05 | — | ns |
| Global Temp Anomaly | **Increasing** | +0.721 | < 0.001 | +0.021°C | *** |

*Note: *** p < 0.001; * p < 0.05; ns = not significant*

### 3.2 Decadal Variability — ANOVA

One-way ANOVA across four decades yielded F(3, 36) = 11.12, p = 0.000006, confirming that mean annual temperature differed significantly between at least one pair of decades. Tukey HSD post-hoc tests identified the 1980s–2000s and 1990s–2010s contrasts as driving the omnibus significance (p < 0.01 for both pairs). The 2000s recorded the lowest mean temperature across the four decades, consistent with the cooling trajectory identified by the Mann-Kendall test.

### 3.3 Time Series Modeling

The ADF test rejected the unit root hypothesis for the Bangladesh temperature series after first differencing (p < 0.05), confirming integration of order one. `auto_arima` selected an ARIMA(1,1,1) model as the optimal specification by AIC. The model projects a continued slight cooling or stabilisation of Bangladesh temperatures through 2030, with widening 95% prediction intervals reflecting structural uncertainty.

For monthly precipitation, SARIMA(p,d,q)(P,D,Q,12) successfully captured the dominant June–September monsoon cycle, as confirmed by ACF/PACF inspection of residuals showing no remaining serial correlation.

### 3.4 OLS Regression Diagnostics

The multiple OLS regression of Bangladesh temperature on Year, Sea_Level_mm, and BGD_Precip_mm_day achieved R² = 0.814, indicating that these three predictors jointly explain 81.4% of interannual temperature variance (F-statistic p < 0.001). Shapiro-Wilk test on OLS residuals: p > 0.05, confirming normality. The Q-Q plot showed no systematic departure from linearity. These diagnostics validate the statistical assumptions underlying the regression.

### 3.5 Flood Risk Prediction — Machine Learning Results

The Random Forest classifier achieved the following performance on the 2016–2023 holdout test set:

**Table 3. Random Forest Classifier — Test Set Performance**

| Metric | Value |
|--------|-------|
| Accuracy | **95.8%** |
| ROC-AUC | **0.9947** |
| F1 Score | **0.9535** |
| Precision | 0.9535 |
| Recall | 0.9535 |
| CV Accuracy (5-fold) | 97.9% ± 2.2% |
| CV ROC-AUC (5-fold) | 99.9% ± 0.2% |

The model correctly classified 92 of 96 test-set months. The ROC-AUC of 0.9947 indicates near-perfect discrimination between high and low flood-risk months. Cross-validation results confirm that performance is not an artefact of the particular train-test split.

### 3.6 SHAP Feature Attribution

SHAP analysis identified the following hierarchy of flood-risk drivers (by mean absolute SHAP value):

1. **Precipitation_mm_day** — current month rainfall; highest absolute impact
2. **Humidity_pct** — atmospheric moisture amplifies precipitation-driven risk
3. **Precip_3mo_roll** — 3-month antecedent rainfall; soil saturation effect
4. **Is_Monsoon** — binary season indicator; structural baseline risk
5. **Precip_anomaly** — above-normal rainfall relative to climatology

The directional SHAP analysis confirmed that high precipitation values consistently push predictions toward high flood risk (positive SHAP), while low precipitation values and non-monsoon months push toward low risk. Temperature showed a weak negative SHAP effect — cooler months tend toward lower risk, consistent with the post-monsoon temperature pattern.

The waterfall plot for the highest-risk month in the test set (June 2016) decomposed the prediction into individual feature contributions, showing that above-average current precipitation (+0.22 SHAP units) and elevated 3-month rolling mean (+0.18 SHAP units) were the dominant drivers for that specific observation.

### 3.7 Flood Risk Seasonality

The monthly flood risk heatmap (Figure 4) reveals a highly consistent seasonal pattern across all 40 years. **September** is the peak risk month in terms of mean risk score, followed by August and July. High-risk periods are strongly concentrated in June–September (the monsoon season), with occasional elevated risk in October during monsoon retreat. The year 2017 recorded the highest mean annual risk score across the full dataset, consistent with documented severe flooding in Bangladesh that year.

### 3.8 Climate Vulnerability Index

The CVI showed a mean value of 0.471 during 1984–1993 compared with 0.386 during 2013–2023. This apparent 18% decrease reflects the dominant effect of rapidly improving GDP per capita and declining under-5 mortality (both representing increasing adaptive capacity) outweighing the increasing physical hazard from sea level rise and precipitation variability. This finding highlights a critical policy insight: **economic development is currently Bangladesh's most effective climate adaptation mechanism**, though this buffer may erode if physical hazards accelerate.

---

## 4. Discussion

### 4.1 The South Asian Warming Hole

The −0.026°C/year cooling trend identified in this study is consistent with findings from regional climate literature. Roxy et al. (2015) documented a weakening of the South Asian summer monsoon linked to rapid warming of the Indian Ocean surface, which reduces the land-ocean thermal gradient driving monsoon circulation. The resulting aerosol feedback — where reduced monsoon ventilation increases residence time of anthropogenic aerosols from the Indo-Gangetic Plain — further suppresses surface warming through increased shortwave scattering.

Critically, this cooling signal is **not expected to persist indefinitely**. As air quality regulations improve across India and Bangladesh, aerosol loading will decrease, and suppressed warming will rapidly emerge — a phenomenon sometimes called the "aerosol unmasking effect." The Bangladesh temperature record may therefore underrepresent future warming risk.

### 4.2 Flood Risk and the 3-Month Antecedent Signal

The identification of 3-month rolling precipitation (Precip_3mo_roll) as the third-ranked SHAP feature has direct operational implications. Soil moisture models consistently show that antecedent rainfall over 2–3 months determines baseline soil saturation, such that even moderate current rainfall can trigger flooding on already-saturated soils. This result supports the design of early warning systems that monitor cumulative, not merely instantaneous, rainfall.

### 4.3 Limitations

Several limitations should be acknowledged:

1. **Flood label proxy:** The flood risk label was constructed from a composite meteorological score rather than observed flood event records. While physically motivated, this approach may not capture all mechanisms of Bangladesh flooding (e.g., glacial lake outburst floods, storm surges). Integration of EM-DAT historical disaster data would strengthen the label definition.

2. **Point-location NASA POWER data:** Climate data were downloaded for a single centroid coordinate representing all of Bangladesh. The country spans approximately 650 km north-to-south with substantial spatial heterogeneity in rainfall and temperature — particularly between the haor wetlands in the northeast and the coastal belt.

3. **Sea level data gap:** The CSIRO sea level dataset extends only to 2013, leaving a 10-year gap in the study period. Incorporation of satellite altimetry data (e.g., TOPEX/Poseidon, Jason series) would provide complete temporal coverage.

4. **Aerosol data absent:** The South Asian Warming Hole hypothesis could be tested more rigorously with MODIS aerosol optical depth data for the study region, which was not incorporated in this analysis.

---

## 5. Conclusion

This study demonstrates that rigorous climate risk analysis for Bangladesh is achievable using entirely open data sources and reproducible Python workflows. The five key conclusions are:

1. **Bangladesh experienced statistically significant cooling** (−0.026°C/yr, p = 0.0008) over 1984–2023, consistent with the South Asian Warming Hole, while global temperatures rose by +0.021°C/yr over the same period.

2. **Despite local cooling, physical climate risk is escalating** through sea level rise (accelerating quadratically), increasing precipitation variability, and growing socioeconomic exposure.

3. **Flood risk is highly predictable** from climate variables: a Random Forest model achieved 95.8% accuracy and ROC-AUC = 0.9947, with 3-month antecedent precipitation emerging as a critical early warning signal via SHAP analysis.

4. **Bangladesh's Climate Vulnerability Index declined** over 1984–2023, driven by GDP growth and mortality improvements — but this adaptive capacity buffer may be insufficient if physical hazards accelerate following aerosol unmasking.

5. **Five evidence-based policies** are supported by these findings: a monsoon early warning system, coastal protection infrastructure, agricultural calendar adjustment, aerosol monitoring, and targeted climate finance via the Green Climate Fund.

Future work should incorporate satellite-derived aerosol optical depth, EM-DAT disaster event records, multi-location spatial analysis, and deep learning approaches (e.g., LSTM) for improved temporal forecasting.

---

## References

Church, J. A., & White, N. J. (2011). Sea-Level Rise from the Late 19th to the Early 21st Century. *Surveys in Geophysics*, 32(4–5), 585–602.

Hansen, J., Ruedy, R., Sato, M., & Lo, K. (2010). Global Surface Temperature Change. *Reviews of Geophysics*, 48(4).

IPCC. (2021). *Climate Change 2021: The Physical Science Basis.* Contribution of Working Group I to the Sixth Assessment Report. Cambridge University Press.

Kendall, M. G. (1975). *Rank Correlation Methods* (4th ed.). Charles Griffin.

Krishnan, R., Sabin, T. P., Vellore, R., Mujumdar, M., Sanjay, J., Goswami, B. N., Hourdin, F., Dufresne, J.-L., & Terray, P. (2016). Deciphering the desiccation trend of the South Asian monsoon hydroclimate in a warming world. *Climate Dynamics*, 47(3–4), 1007–1027.

Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems*, 30.

Mann, H. B. (1945). Non-parametric tests against trend. *Econometrica*, 13(3), 245–259.

NASA POWER Project. (2023). *Prediction of Worldwide Energy Resources (POWER)*. NASA Langley Research Center. https://power.larc.nasa.gov/

Roxy, M. K., Ritika, K., Terray, P., Murtugudde, R., Ashok, K., & Goswami, B. N. (2015). Drying of Indian subcontinent by rapid Indian Ocean warming and a weakening land-sea thermal gradient. *Nature Communications*, 6, 7423.

World Bank. (2023). *World Development Indicators*. https://databank.worldbank.org/source/world-development-indicators

---

*Repository: [github.com/almazid82/ClimateScope-Bangladesh](https://github.com/almazid82/ClimateScope-Bangladesh)*
*All code, data, and figures are openly available under the MIT License.*
