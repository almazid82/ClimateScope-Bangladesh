"""Run Phase 5 — Visualization & Storytelling (standalone, no Jupyter needed)"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

RAW  = '../data/raw'
PROC = '../data/processed'
FIG  = '../outputs/figures'
os.makedirs(FIG, exist_ok=True)

master  = pd.read_csv(f'{PROC}/master_annual_dataset.csv')
df_giss = pd.read_csv(f'{PROC}/global_temperature_anomaly.csv')
df_nasa = pd.read_csv(f'{RAW}/bangladesh_nasa_power_monthly.csv',
                      index_col='Date', parse_dates=True)

print(f'Master   : {master.shape[0]} rows x {master.shape[1]} cols')
print(f'GISS     : {df_giss.shape[0]} years')
print(f'Monthly  : {df_nasa.shape[0]} months')

# ── 1. CLIMATE PARADOX ────────────────────────────────────────────────────────
print('\n1. Climate Paradox chart...')
bgd = master.dropna(subset=['BGD_Temp_C', 'Annual_Anomaly_C'])
giss_overlap = df_giss[df_giss.Year.isin(bgd.Year)]
bgd_slope = np.polyfit(bgd.Year, bgd.BGD_Temp_C, 1)[0]
glo_slope = np.polyfit(giss_overlap.Year, giss_overlap.Annual_Anomaly_C, 1)[0]

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('The Bangladesh Climate Paradox: Local Cooling vs Global Warming',
             fontsize=14, fontweight='bold', y=1.02)

ax1 = axes[0]
ax2 = ax1.twinx()
l1, = ax1.plot(bgd.Year, bgd.BGD_Temp_C, color='#0984e3', linewidth=2,
               label='Bangladesh Temp (°C)')
z1 = np.polyfit(bgd.Year, bgd.BGD_Temp_C, 1)
ax1.plot(bgd.Year, np.polyval(z1, bgd.Year), '--', color='#0984e3',
         linewidth=1.5, alpha=0.6)
l2, = ax2.plot(giss_overlap.Year, giss_overlap.Annual_Anomaly_C,
               color='#e17055', linewidth=2, label='Global Temp Anomaly (°C)')
z2 = np.polyfit(giss_overlap.Year, giss_overlap.Annual_Anomaly_C, 1)
ax2.plot(giss_overlap.Year, np.polyval(z2, giss_overlap.Year), '--',
         color='#e17055', linewidth=1.5, alpha=0.6)
ax2.axhline(0, color='gray', linewidth=0.8, linestyle=':')
ax1.set_xlabel('Year')
ax1.set_ylabel('Bangladesh Temp (°C)', color='#0984e3')
ax2.set_ylabel('Global Anomaly (°C)', color='#e17055')
ax1.set_title(f'Temperature Trends 1984-2023\n'
              f'BGD: {bgd_slope:+.4f} C/yr  |  Global: {glo_slope:+.4f} C/yr')
ax1.legend(handles=[l1, l2], loc='upper right', fontsize=9)

axes[1].barh(['Bangladesh\n(local)', 'Global\n(anomaly)'],
             [bgd_slope * 40, glo_slope * 40],
             color=['#0984e3', '#e17055'], height=0.4)
axes[1].axvline(0, color='black', linewidth=0.8)
axes[1].set_xlabel('Total change 1984-2023 (°C)')
axes[1].set_title('40-Year Temperature Change\nBangladesh vs Global')
for i, v in enumerate([bgd_slope * 40, glo_slope * 40]):
    axes[1].text(v + (0.01 if v >= 0 else -0.01), i,
                 f'{v:+.3f} C', va='center',
                 ha='left' if v >= 0 else 'right', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{FIG}/phase5_climate_paradox.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'   BGD trend : {bgd_slope:+.5f} C/yr  ({bgd_slope*40:+.3f} C over 40 yrs)')
print(f'   Global    : {glo_slope:+.5f} C/yr  ({glo_slope*40:+.3f} C over 40 yrs)')

# ── 2. MASTER 6-PANEL DASHBOARD ───────────────────────────────────────────────
print('\n2. Master 6-panel dashboard...')
fig = plt.figure(figsize=(16, 10))
fig.suptitle('ClimateScope Bangladesh 1984-2023 — Master Dashboard',
             fontsize=15, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)
m = master.copy()

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(m.Year, m.BGD_Temp_C, color='#e17055', linewidth=1.5)
ax1.fill_between(m.Year, m.BGD_Temp_C, m.BGD_Temp_C.mean(), alpha=0.15, color='#e17055')
zz = np.polyfit(m.dropna(subset=['BGD_Temp_C']).Year,
                m.dropna(subset=['BGD_Temp_C']).BGD_Temp_C, 1)
ax1.plot(m.dropna(subset=['BGD_Temp_C']).Year,
         np.polyval(zz, m.dropna(subset=['BGD_Temp_C']).Year),
         '--', color='darkred', linewidth=1.5, label=f'Trend: {zz[0]:+.4f} C/yr')
ax1.set_title('Bangladesh Temperature'); ax1.set_ylabel('°C'); ax1.legend(fontsize=7)

ax2 = fig.add_subplot(gs[0, 1])
ax2.bar(m.Year, m.BGD_Precip_mm_day, color='#74b9ff', alpha=0.8, width=0.7)
ax2.plot(m.Year, m.BGD_Precip_mm_day.rolling(5, min_periods=1).mean(),
         color='navy', linewidth=2, label='5-yr rolling mean')
ax2.set_title('Precipitation'); ax2.set_ylabel('mm/day'); ax2.legend(fontsize=7)

ax3 = fig.add_subplot(gs[0, 2])
sl = m.dropna(subset=['Sea_Level_mm'])
ax3.plot(sl.Year, sl.Sea_Level_mm, color='#0984e3', linewidth=2)
ax3.fill_between(sl.Year, sl.Sea_Level_mm, sl.Sea_Level_mm.min(),
                 alpha=0.2, color='#0984e3')
ax3.set_title('Global Sea Level'); ax3.set_ylabel('mm')

ax4 = fig.add_subplot(gs[1, 0])
gdp = m.dropna(subset=['GDP_per_capita_USD'])
ax4.plot(gdp.Year, gdp.GDP_per_capita_USD / 1000, color='#00b894', linewidth=2)
ax4.fill_between(gdp.Year, gdp.GDP_per_capita_USD / 1000, alpha=0.15, color='#00b894')
ax4.set_title('GDP per Capita'); ax4.set_ylabel('USD (thousands)')

ax5 = fig.add_subplot(gs[1, 1])
co2 = m.dropna(subset=['CO2_emissions'])
ax5.bar(co2.Year, co2.CO2_emissions, color='#636e72', alpha=0.7, width=0.7)
ax5.set_title('Bangladesh CO2 Emissions'); ax5.set_ylabel('kt CO2')

ax6 = fig.add_subplot(gs[1, 2])
giss_sub = df_giss[df_giss.Year.between(1984, 2023)]
colors6 = ['#e17055' if v > 0 else '#74b9ff' for v in giss_sub.Annual_Anomaly_C]
ax6.bar(giss_sub.Year, giss_sub.Annual_Anomaly_C, color=colors6, width=0.7, alpha=0.85)
ax6.axhline(0, color='black', linewidth=0.8)
ax6.set_title('Global Temp Anomaly (NASA GISS)'); ax6.set_ylabel('°C anomaly')

plt.savefig(f'{FIG}/phase5_master_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print('   Master dashboard saved.')

# ── 3. FLOOD RISK HEATMAP ─────────────────────────────────────────────────────
print('\n3. Monthly flood risk heatmap...')
df = df_nasa.copy()
df['Month'] = df.index.month
df['Year']  = df.index.year

mc_mean = df.groupby('Month')[['Precipitation_mm_day', 'Humidity_pct']].transform('mean')
mc_std  = df.groupby('Month')[['Precipitation_mm_day', 'Humidity_pct']].transform('std').replace(0, 1)
pz = (df['Precipitation_mm_day'] - mc_mean['Precipitation_mm_day']) / mc_std['Precipitation_mm_day']
hz = (df['Humidity_pct']          - mc_mean['Humidity_pct'])          / mc_std['Humidity_pct']
df['Risk_Score'] = 0.60 * pz + 0.20 * hz + 0.20 * df['Month'].isin([6, 7, 8, 9]).astype(int)

pivot = df.pivot_table(index='Month', columns='Year', values='Risk_Score', aggfunc='mean')
MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

fig, ax = plt.subplots(figsize=(18, 6))
sns.heatmap(pivot, ax=ax, cmap='RdYlBu_r', center=0,
            linewidths=0.3, linecolor='white',
            cbar_kws={'label': 'Flood Risk Score', 'shrink': 0.8},
            yticklabels=MONTH_NAMES)
ax.set_title('Monthly Flood Risk Heatmap — Bangladesh 1984-2023\n'
             '(Red = High Risk, Blue = Low Risk)', fontsize=13, fontweight='bold')
ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Month', fontsize=11)

for m_idx in [5, 6, 7, 8]:
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, m_idx), pivot.shape[1], 1,
        boxstyle='square,pad=0', linewidth=2,
        edgecolor='#2d3436', facecolor='none'
    ))

plt.tight_layout()
plt.savefig(f'{FIG}/phase5_flood_risk_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
peak_month = MONTH_NAMES[int(pivot.mean(axis=1).idxmax()) - 1]
peak_year  = int(pivot.mean(axis=0).idxmax())
print(f'   Peak risk month: {peak_month}')
print(f'   Peak risk year : {peak_year}')

# ── 4. CLIMATE VULNERABILITY INDEX ───────────────────────────────────────────
print('\n4. Climate Vulnerability Index...')
cvi_data = master.dropna(subset=['Sea_Level_mm', 'BGD_Precip_mm_day',
                                  'GDP_per_capita_USD', 'Under5_mortality']).copy()

def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

cvi_data['SL_norm']     = normalize(cvi_data['Sea_Level_mm'])
cvi_data['Precip_var']  = (cvi_data['BGD_Precip_mm_day'] - cvi_data['BGD_Precip_mm_day'].mean()).abs()
cvi_data['Precip_norm'] = normalize(cvi_data['Precip_var'])
cvi_data['GDP_norm']    = 1 - normalize(cvi_data['GDP_per_capita_USD'])
cvi_data['Mort_norm']   = normalize(cvi_data['Under5_mortality'])
cvi_data['CVI'] = (0.35 * cvi_data['SL_norm']   +
                   0.25 * cvi_data['Precip_norm'] +
                   0.20 * cvi_data['GDP_norm']    +
                   0.20 * cvi_data['Mort_norm'])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.fill_between(cvi_data.Year, cvi_data.CVI, alpha=0.3, color='#d63031')
ax1.plot(cvi_data.Year, cvi_data.CVI, color='#d63031', linewidth=2.5)
ax1.plot(cvi_data.Year, cvi_data.CVI.rolling(5, min_periods=1).mean(),
         '--', color='darkred', linewidth=1.5, label='5-yr rolling mean')
ax1.set_xlabel('Year'); ax1.set_ylabel('Climate Vulnerability Index (0-1)')
ax1.set_title('Bangladesh Climate Vulnerability Index 1984-2023')
ax1.legend(fontsize=9)

last20 = cvi_data[cvi_data.Year >= 2004]
comps  = ['SL_norm', 'Precip_norm', 'GDP_norm', 'Mort_norm']
labels_area = ['Sea Level (35%)', 'Precip Variability (25%)',
               'Low GDP (20%)', 'Mortality (20%)']
colors_area = ['#0984e3', '#74b9ff', '#e17055', '#fdcb6e']
weights = [0.35, 0.25, 0.20, 0.20]
scaled = [last20[c] * w for c, w in zip(comps, weights)]
ax2.stackplot(last20.Year, scaled, labels=labels_area, colors=colors_area, alpha=0.85)
ax2.set_xlabel('Year'); ax2.set_ylabel('Weighted Vulnerability Component')
ax2.set_title('CVI Component Breakdown (2004-2023)')
ax2.legend(loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig(f'{FIG}/phase5_vulnerability_index.png', dpi=150, bbox_inches='tight')
plt.close()
cvi_early  = cvi_data[cvi_data.Year <= 1994].CVI.mean()
cvi_recent = cvi_data[cvi_data.Year >= 2013].CVI.mean()
print(f'   CVI 1984-1993  : {cvi_early:.4f}')
print(f'   CVI 2013-2023  : {cvi_recent:.4f}')
print(f'   Vulnerability  : +{(cvi_recent - cvi_early)/cvi_early*100:.1f}% increase')

# ── 5. SEA LEVEL ANALYSIS ─────────────────────────────────────────────────────
print('\n5. Sea level analysis...')
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Sea Level Rise: Relationships with Climate & Socioeconomic Variables',
             fontsize=13, fontweight='bold')

sl_data = master.dropna(subset=['Sea_Level_mm'])
z1_sl = np.polyfit(sl_data.Year, sl_data.Sea_Level_mm, 1)
z2_sl = np.polyfit(sl_data.Year, sl_data.Sea_Level_mm, 2)
yr_range = np.linspace(sl_data.Year.min(), sl_data.Year.max(), 100)

axes[0].scatter(sl_data.Year, sl_data.Sea_Level_mm,
                color='#0984e3', s=20, alpha=0.6, zorder=3)
axes[0].plot(yr_range, np.polyval(z1_sl, yr_range), '--', color='gray',
             linewidth=1.5, label=f'Linear ({z1_sl[0]:.2f}mm/yr)')
axes[0].plot(yr_range, np.polyval(z2_sl, yr_range), '-', color='navy',
             linewidth=2, label='Quadratic (acceleration)')
axes[0].set_xlabel('Year'); axes[0].set_ylabel('Sea Level (mm)')
axes[0].set_title('Sea Level Rise Trend'); axes[0].legend(fontsize=8)

sub = master.dropna(subset=['Sea_Level_mm', 'Annual_Anomaly_C'])
sc = axes[1].scatter(sub.Annual_Anomaly_C, sub.Sea_Level_mm,
                     c=sub.Year, cmap='plasma', s=30, alpha=0.8)
z_sl2 = np.polyfit(sub.Annual_Anomaly_C, sub.Sea_Level_mm, 1)
x_fit = np.linspace(sub.Annual_Anomaly_C.min(), sub.Annual_Anomaly_C.max(), 50)
axes[1].plot(x_fit, np.polyval(z_sl2, x_fit), '--', color='darkred', linewidth=1.5)
plt.colorbar(sc, ax=axes[1], label='Year')
r1 = np.corrcoef(sub.Annual_Anomaly_C, sub.Sea_Level_mm)[0, 1]
axes[1].set_xlabel('Global Temp Anomaly (°C)'); axes[1].set_ylabel('Sea Level (mm)')
axes[1].set_title(f'Sea Level vs Global Temp\nr = {r1:.3f}')

sub2 = master.dropna(subset=['Sea_Level_mm', 'BGD_Precip_mm_day'])
sc2 = axes[2].scatter(sub2.BGD_Precip_mm_day, sub2.Sea_Level_mm,
                      c=sub2.Year, cmap='viridis', s=30, alpha=0.8)
plt.colorbar(sc2, ax=axes[2], label='Year')
r2 = np.corrcoef(sub2.BGD_Precip_mm_day, sub2.Sea_Level_mm)[0, 1]
axes[2].set_xlabel('BGD Precipitation (mm/day)'); axes[2].set_ylabel('Sea Level (mm)')
axes[2].set_title(f'Sea Level vs Precipitation\nr = {r2:.3f}')

plt.tight_layout()
plt.savefig(f'{FIG}/phase5_sea_level_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'   Sea level rise : {z1_sl[0]:.2f} mm/yr')
print(f'   Quadratic coef : {z2_sl[0]:.5f} (positive = accelerating)')

# ── 6. POLICY RECOMMENDATIONS ─────────────────────────────────────────────────
print('\n6. Policy recommendations chart...')
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('#f8f9fa')
ax.axis('off')
ax.text(0.5, 0.97,
        'Evidence-Based Climate Policy Recommendations for Bangladesh',
        transform=ax.transAxes,
        fontsize=14, fontweight='bold', ha='center', va='top', color='#2d3436')

policies = [
    ('1. Flood Early Warning System', '#0984e3',
     'Evidence: 3-month rolling precipitation (SHAP rank top-3) predicts flood risk\n'
     'with 95.8% accuracy. Automated alerts when Precip_3mo_roll exceeds 70th percentile.',
     'Deploy: 2025 monsoon season'),
    ('2. Coastal Protection Infrastructure', '#e17055',
     'Evidence: Sea level rising at ~1.8 mm/yr with quadratic acceleration.\n'
     'Trajectory: +72 mm by 2060. Invest in mangroves + embankments below 1.0m elevation.',
     'Begin feasibility: 2025'),
    ('3. Agricultural Calendar Shift', '#00b894',
     'Evidence: ANOVA shows significant decade-level temp change (F=11.12, p<0.001).\n'
     'June-Sep monsoon months uniformly high-risk. Shift Boro rice planting 2 weeks earlier.',
     'Pilot: Sylhet division 2025'),
    ('4. Aerosol Monitoring Programme', '#6c5ce7',
     'Evidence: South Asian Warming Hole masks true warming. When industrial aerosols\n'
     'reduce (policy-driven), rapid temperature rebound is expected. Baseline now.',
     'Partner: ICIMOD by 2026'),
    ('5. Climate Finance & GDP Resilience', '#fdcb6e',
     'Evidence: OLS model — GDP explains 81.4% of climate adaptation capacity variance.\n'
     'Under-5 mortality and agricultural land are key vulnerability indicators.',
     'GCF proposal: 2025'),
]

y_pos = 0.88
for name, color, evidence, action in policies:
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.02, y_pos - 0.12), 0.96, 0.135,
        boxstyle='round,pad=0.01', linewidth=1.5,
        edgecolor=color, facecolor=color + '18',
        transform=ax.transAxes, clip_on=False
    ))
    ax.text(0.04, y_pos - 0.01, name, transform=ax.transAxes,
            fontsize=10, fontweight='bold', color=color, va='top')
    ax.text(0.04, y_pos - 0.05, evidence, transform=ax.transAxes,
            fontsize=8, color='#2d3436', va='top', family='monospace')
    ax.text(0.97, y_pos - 0.08, action, transform=ax.transAxes,
            fontsize=8, color=color, va='top', fontweight='bold', ha='right')
    y_pos -= 0.175

plt.savefig(f'{FIG}/phase5_policy_recommendations.png', dpi=150, bbox_inches='tight')
plt.close()
print('   Policy chart saved.')

# ── 7. SUMMARY INFOGRAPHIC ────────────────────────────────────────────────────
print('\n7. Summary infographic...')
fig = plt.figure(figsize=(16, 9))
fig.patch.set_facecolor('#1a1a2e')
ax = fig.add_subplot(111)
ax.set_facecolor('#1a1a2e')
ax.axis('off')

ax.text(0.5, 0.95, 'ClimateScope Bangladesh & South Asia',
        transform=ax.transAxes, fontsize=20, fontweight='bold',
        ha='center', va='top', color='white')
ax.text(0.5, 0.88, 'Data-Driven Climate Risk Analysis  |  1984-2023',
        transform=ax.transAxes, fontsize=12, ha='center', va='top', color='#a0a0c0')

stats = [
    ('40 Years',   'of climate data\nanalysed',              '#0984e3'),
    ('5 Sources',  'NASA POWER * GISS\nWorld Bank * Sea Level * CO2', '#00b894'),
    ('-0.026C/yr', 'Bangladesh temperature\ntrend (cooling!)',        '#74b9ff'),
    ('+1.8mm/yr',  'Sea level rise\nrate (accelerating)',            '#e17055'),
    ('95.8%',      'Flood risk prediction\naccuracy (RF model)',      '#00cec9'),
    ('0.9947',     'ROC-AUC score\n(near-perfect classifier)',        '#fdcb6e'),
    ('p < 0.001',  'Mann-Kendall trend\nsignificance',                '#a29bfe'),
    ('SHAP',       'Explainable AI\nfor flood drivers',               '#fd79a8'),
]

x_positions = [0.10, 0.35, 0.60, 0.85]
for i, (val, label, color) in enumerate(stats):
    row = i // 4
    col = i % 4
    x = x_positions[col]
    y = 0.68 - row * 0.28

    ax.add_patch(mpatches.FancyBboxPatch(
        (x - 0.10, y - 0.14), 0.20, 0.20,
        boxstyle='round,pad=0.01', linewidth=2,
        edgecolor=color, facecolor='#16213e',
        transform=ax.transAxes, clip_on=False
    ))
    ax.text(x, y + 0.01, val, transform=ax.transAxes,
            fontsize=13, fontweight='bold', ha='center', va='center', color=color)
    ax.text(x, y - 0.07, label, transform=ax.transAxes,
            fontsize=7.5, ha='center', va='center', color='#a0a0c0')

ax.text(0.5, 0.04,
        'Author: Shamsul AL Mazid  |  github.com/almazid82  |  '
        'Tools: Python, scikit-learn, SHAP, statsmodels, NASA POWER API, World Bank API',
        transform=ax.transAxes, fontsize=8, ha='center', va='bottom', color='#606080')

plt.savefig(f'{FIG}/phase5_summary_infographic.png', dpi=150,
            bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
print('   Summary infographic saved.')

# ── FINAL REPORT ──────────────────────────────────────────────────────────────
print()
print('=' * 65)
print('  PHASE 5 COMPLETE — ALL FIGURES')
print('=' * 65)
phase5_figs = sorted([f for f in os.listdir(FIG) if 'phase5' in f])
for fname in phase5_figs:
    kb = os.path.getsize(f'{FIG}/{fname}') // 1024
    print(f'  {fname:<50} {kb} KB')

print()
all_figs = sorted([f for f in os.listdir(FIG) if f.endswith('.png')])
total_kb = sum(os.path.getsize(f'{FIG}/{f}') // 1024 for f in all_figs)
print(f'  Total project figures : {len(all_figs)} files  |  {total_kb} KB')
print('=' * 65)
print()
print('  PROJECT COMPLETE — ClimateScope Bangladesh & South Asia')
print('  All 5 phases done. Ready for GitHub push.')
print('=' * 65)
