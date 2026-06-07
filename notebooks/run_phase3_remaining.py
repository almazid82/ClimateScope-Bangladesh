"""Run ANOVA, Polynomial Regression, OLS — Phase 3 remaining sections"""
import pandas as pd
import numpy as np
import warnings
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')

PROC = '../data/processed'
FIG  = '../outputs/figures'

master  = pd.read_csv(f'{PROC}/master_annual_dataset.csv')
df_giss = pd.read_csv(f'{PROC}/global_temperature_anomaly.csv')
master['Decade'] = (master['Year'] // 10) * 10

# ── 1. ANOVA ──────────────────────────────────────────────────────────────────
print('1. Running ANOVA...')
valid = master.dropna(subset=['BGD_Temp_C'])
decade_vals   = sorted(valid['Decade'].unique())
decade_groups = [valid.loc[valid['Decade'] == d, 'BGD_Temp_C'].values for d in decade_vals]
decade_groups = [g for g in decade_groups if len(g) > 1]
decade_labels = [f"{d}s" for d, g in zip(decade_vals, [valid.loc[valid['Decade']==d,'BGD_Temp_C'].values for d in decade_vals]) if len(g) > 1]

f_stat, p_value = stats.f_oneway(*decade_groups)
print(f'   F={f_stat:.4f}, p={p_value:.6f}')

temp_all   = valid['BGD_Temp_C'].values
decade_all = (valid['Decade'].astype(str) + 's').values
tukey = pairwise_tukeyhsd(endog=temp_all, groups=decade_all, alpha=0.05)
print(tukey)

palette = ['#74b9ff', '#0984e3', '#fdcb6e', '#e17055']
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

sns.boxplot(x=decade_all, y=temp_all, order=decade_labels,
            palette=palette[:len(decade_labels)], ax=ax1, width=0.5)
ax1.set_xlabel('Decade')
ax1.set_ylabel('Temperature (°C)')
ax1.set_title(f'ANOVA: Temperature by Decade\nF={f_stat:.3f}, p={p_value:.4f}')
sig_text = '*** Significant' if p_value < 0.05 else 'Not significant'
sig_color = 'darkred' if p_value < 0.05 else 'gray'
ax1.text(0.5, 0.95, sig_text, transform=ax1.transAxes,
         ha='center', fontsize=10, color=sig_color)

means = [g.mean() for g in decade_groups]
stds  = [g.std()  for g in decade_groups]
ax2.bar(range(len(means)), means, yerr=stds, color=palette[:len(means)],
        capsize=4, alpha=0.8, width=0.5, edgecolor='black', linewidth=0.5)
ax2.set_xticks(range(len(decade_labels)))
ax2.set_xticklabels(decade_labels)
ax2.set_ylabel('Mean Temperature +/- SD (°C)')
ax2.set_title('Decade Mean Temperatures (±1 SD)')
ax2.set_ylim(min(means) - 0.5, max(means) + 0.5)

plt.tight_layout()
plt.savefig(f'{FIG}/phase3_anova_decades.png', dpi=150)
plt.close()
print('   ANOVA figure saved.')

# ── 2. POLYNOMIAL REGRESSION ──────────────────────────────────────────────────
print('2. Running Polynomial Regression...')
X = master['Year'].values.reshape(-1, 1)
y = master['BGD_Temp_C'].values

lin_model  = make_pipeline(PolynomialFeatures(1), LinearRegression())
quad_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
lin_model.fit(X, y)
quad_model.fit(X, y)

y_lin  = lin_model.predict(X)
y_quad = quad_model.predict(X)
r2_lin  = r2_score(y, y_lin)
r2_quad = r2_score(y, y_quad)

future = np.arange(1984, 2031).reshape(-1, 1)

fig, ax = plt.subplots(figsize=(13, 5))
ax.scatter(master['Year'], y, color='tomato', s=30, zorder=4, alpha=0.7, label='Observed')
ax.plot(future, lin_model.predict(future), color='steelblue', linewidth=2,
        linestyle='--', label=f'Linear fit (R²={r2_lin:.3f})')
ax.plot(future, quad_model.predict(future), color='darkred', linewidth=2.5,
        label=f'Polynomial degree-2 (R²={r2_quad:.3f})')
ax.axvline(2023, color='gray', linewidth=1, linestyle=':', label='Forecast boundary')
ax.set_xlabel('Year')
ax.set_ylabel('Temperature (°C)')
ax.set_title('Polynomial vs Linear Regression — Bangladesh Temperature Trend')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f'{FIG}/phase3_polynomial_regression.png', dpi=150)
plt.close()

poly_coef = quad_model.named_steps['linearregression'].coef_
direction = 'DECELERATING' if poly_coef[2] < 0 else 'ACCELERATING'
print(f'   Quadratic coef: {poly_coef[2]:.6f} — trend is {direction}')
print(f'   Linear R²={r2_lin:.4f} | Polynomial R²={r2_quad:.4f}')

# ── 3. OLS REGRESSION + DIAGNOSTICS ──────────────────────────────────────────
print('3. Running OLS Regression...')

# Global anomaly ~ Year
X_giss = sm.add_constant(df_giss['Year'].tail(80))
y_giss = df_giss['Annual_Anomaly_C'].tail(80)
ols_global = sm.OLS(y_giss, X_giss).fit()
print(f'   OLS Global: R²={ols_global.rsquared:.4f}, coef Year={ols_global.params["Year"]:.6f}')

# Bangladesh temperature ~ Year + Sea Level + Precipitation
reg_data = master[['Year', 'BGD_Temp_C', 'Sea_Level_mm', 'BGD_Precip_mm_day']].dropna()
X_bgd = sm.add_constant(reg_data[['Year', 'Sea_Level_mm', 'BGD_Precip_mm_day']])
y_bgd = reg_data['BGD_Temp_C']
ols_bgd = sm.OLS(y_bgd, X_bgd).fit()
print(f'   OLS BGD:    R²={ols_bgd.rsquared:.4f}, F-p={ols_bgd.f_pvalue:.6f}')

# Residuals diagnostics plot
residuals = ols_bgd.resid
fitted_v  = ols_bgd.fittedvalues

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('OLS Residuals Diagnostics — Bangladesh Temperature Model', fontsize=13)

axes[0].scatter(fitted_v, residuals, color='steelblue', alpha=0.7, s=25)
axes[0].axhline(0, color='red', linewidth=1)
axes[0].set_xlabel('Fitted values')
axes[0].set_ylabel('Residuals')
axes[0].set_title('Residuals vs Fitted')

stats.probplot(residuals, dist='norm', plot=axes[1])
axes[1].set_title('Q-Q Plot (Normality Check)')

axes[2].hist(residuals, bins=10, color='steelblue', edgecolor='white', alpha=0.8)
axes[2].set_xlabel('Residual')
axes[2].set_ylabel('Count')
axes[2].set_title('Residuals Distribution')

plt.tight_layout()
plt.savefig(f'{FIG}/phase3_ols_diagnostics.png', dpi=150)
plt.close()

_, p_sw = stats.shapiro(residuals)
normality = 'Normal' if p_sw > 0.05 else 'Non-normal'
print(f'   Shapiro-Wilk p={p_sw:.4f} ({normality})')

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
print()
print('=' * 55)
print('  ALL PHASE 3 SECTIONS COMPLETE')
print('=' * 55)
phase3_figs = [f for f in os.listdir(FIG) if 'phase3' in f]
for fname in sorted(phase3_figs):
    kb = os.path.getsize(f'{FIG}/{fname}') // 1024
    print(f'  {fname:<45} {kb} KB')
print('=' * 55)
