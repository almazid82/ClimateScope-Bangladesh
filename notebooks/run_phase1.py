"""Phase 1 Data Collection — ClimateScope Bangladesh"""
import pandas as pd
import numpy as np
import requests
import warnings
import os
import wbgapi as wb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

RAW  = '../data/raw'
PROC = '../data/processed'
FIG  = '../outputs/figures'
for p in [RAW, PROC, FIG]:
    os.makedirs(p, exist_ok=True)

# ── 1. NASA POWER: Bangladesh climate ─────────────────────────────────────────
print("1. Downloading Bangladesh climate from NASA POWER...")
url = 'https://power.larc.nasa.gov/api/temporal/monthly/point'
params = {
    'parameters': 'T2M,PRECTOTCORR,RH2M',
    'community': 'RE',
    'longitude': 90.356,
    'latitude': 23.685,
    'start': 1984,
    'end': 2023,
    'format': 'JSON'
}
data = requests.get(url, params=params, timeout=120).json()['properties']['parameter']

# Manually build records, skipping any non-YYYYMM keys (e.g. ANN)
records = {}
for param, monthly in data.items():
    for key, val in monthly.items():
        if len(key) == 6 and key.isdigit():
            year  = int(key[:4])
            month = int(key[4:])
            if 1 <= month <= 12:
                records.setdefault(key, {'year': year, 'month': month})[param] = val

rows = []
for key, vals in sorted(records.items()):
    row = {'Date': pd.Timestamp(year=vals['year'], month=vals['month'], day=1)}
    for param in ['T2M', 'PRECTOTCORR', 'RH2M']:
        row[param] = vals.get(param, np.nan)
    rows.append(row)

df_nasa = pd.DataFrame(rows).set_index('Date')
df_nasa.replace(-999.0, np.nan, inplace=True)
df_nasa.columns = ['Temperature_C', 'Precipitation_mm_day', 'Humidity_pct']
df_nasa.to_csv(f'{RAW}/bangladesh_nasa_power_monthly.csv')
print(f"   OK: {df_nasa.shape[0]} rows | {df_nasa.index[0].date()} to {df_nasa.index[-1].date()}")

# ── 2. NASA GISS: Global temperature anomaly ──────────────────────────────────
print("2. Processing NASA GISS global temperature...")
df_giss = pd.read_csv(f'{RAW}/GLB.Ts+dSST.csv', skiprows=1)
df_giss = df_giss[['Year', 'J-D', 'DJF', 'MAM', 'JJA', 'SON']]
df_giss.columns = ['Year', 'Annual_Anomaly_C', 'Winter', 'Spring', 'Summer', 'Autumn']
df_giss.replace('***', np.nan, inplace=True)
df_giss = df_giss.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))
df_giss = df_giss.apply(pd.to_numeric, errors='coerce')
df_giss.fillna(df_giss.median(numeric_only=True), inplace=True)
df_giss.dropna(inplace=True)
df_giss['Year'] = df_giss['Year'].astype(int)
df_giss.to_csv(f'{PROC}/global_temperature_anomaly.csv', index=False)
print(f"   OK: {df_giss.Year.min()}–{df_giss.Year.max()} ({df_giss.shape[0]} years)")

# ── 3. CO2 data ───────────────────────────────────────────────────────────────
print("3. Processing CO2 data...")
df_co2 = pd.read_csv(f'{RAW}/CO2_emissions_global.csv')
df_co2.columns = [c.strip() for c in df_co2.columns]
year_col = next(c for c in df_co2.columns if 'year' in c.lower() or c == 'Year')
df_co2.rename(columns={year_col: 'Year'}, inplace=True)
df_co2['Year'] = pd.to_numeric(df_co2['Year'], errors='coerce')
df_co2.dropna(subset=['Year'], inplace=True)
df_co2['Year'] = df_co2['Year'].astype(int)
num_cols = [c for c in df_co2.columns if c != 'Year']
if num_cols:
    df_co2.rename(columns={num_cols[0]: 'CO2_emissions'}, inplace=True)
df_co2.to_csv(f'{PROC}/co2_emissions_clean.csv', index=False)
print(f"   OK: {df_co2.shape[0]} rows")

# ── 4. World Bank Bangladesh ──────────────────────────────────────────────────
print("4. Downloading World Bank Bangladesh indicators...")
INDICATORS = {
    'NY.GDP.MKTP.CD': 'GDP_USD',
    'NY.GDP.PCAP.CD': 'GDP_per_capita_USD',
    'SP.POP.TOTL':    'Population',
    'AG.LND.AGRI.ZS': 'Agricultural_land_pct',
    'EN.ATM.CO2E.KT': 'CO2_kt_BGD',
    'SH.DYN.MORT':    'Under5_mortality',
}
raw_wb = wb.data.DataFrame(
    list(INDICATORS.keys()), economy='BGD',
    time=range(1960, 2024), skipBlanks=True, columns='series'
)
df_wb = raw_wb.reset_index()
df_wb.rename(columns={'time': 'Year'}, inplace=True)
df_wb['Year'] = df_wb['Year'].astype(str).str.extract(r'(\d{4})').astype(int)
df_wb.rename(columns=INDICATORS, inplace=True)
df_wb = df_wb.sort_values('Year').reset_index(drop=True)
df_wb_clean = df_wb.set_index('Year').interpolate(method='linear').reset_index()
df_wb_clean.to_csv(f'{PROC}/bangladesh_worldbank_clean.csv', index=False)
print(f"   OK: {df_wb_clean.Year.min()}–{df_wb_clean.Year.max()} ({df_wb_clean.shape[0]} years)")

# ── 5. Sea level rise ─────────────────────────────────────────────────────────
print("5. Downloading sea level data...")
df_sea = pd.read_csv(
    'https://raw.githubusercontent.com/datasets/sea-level-rise/master/data/epa-sea-level.csv'
)
df_sea.columns = [c.strip() for c in df_sea.columns]
year_col = next(c for c in df_sea.columns if 'year' in c.lower())
df_sea.rename(columns={year_col: 'Year'}, inplace=True)
df_sea['Year'] = df_sea['Year'].apply(
    lambda x: int(str(x).split('.')[0]) if pd.notnull(x) else np.nan
)
level_col = next(c for c in df_sea.columns if any(k in c.lower() for k in ['csiro', 'adjusted']))
df_sea.rename(columns={level_col: 'Sea_Level_mm'}, inplace=True)
df_sea = df_sea[['Year', 'Sea_Level_mm']].dropna()
df_sea['Year'] = df_sea['Year'].astype(int)
df_sea.to_csv(f'{RAW}/sea_level_rise.csv', index=False)
df_sea.to_csv(f'{PROC}/sea_level_rise_clean.csv', index=False)
print(f"   OK: {df_sea.Year.min()}–{df_sea.Year.max()} ({df_sea.shape[0]} rows)")

# ── 6. Master annual dataset ──────────────────────────────────────────────────
print("6. Building master annual dataset...")
df_nasa_annual = df_nasa.resample('YE').mean()
df_nasa_annual.index = df_nasa_annual.index.year
df_nasa_annual.index.name = 'Year'
df_nasa_annual.columns = ['BGD_Temp_C', 'BGD_Precip_mm_day', 'BGD_Humidity_pct']
df_nasa_annual = df_nasa_annual.reset_index()

master = df_nasa_annual.copy()
master = master.merge(df_giss[['Year', 'Annual_Anomaly_C']], on='Year', how='left')
master = master.merge(df_wb_clean, on='Year', how='left')
master = master.merge(df_sea, on='Year', how='left')
master = master.merge(df_co2[['Year', 'CO2_emissions']], on='Year', how='left')
master = master.sort_values('Year').reset_index(drop=True)
master.to_csv(f'{PROC}/master_annual_dataset.csv', index=False)
print(f"   OK: {master.shape[0]} rows x {master.shape[1]} columns")

# ── 7. Summary chart ──────────────────────────────────────────────────────────
print("7. Generating summary chart...")
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle('ClimateScope Bangladesh — Phase 1 Data Overview', fontsize=14, fontweight='bold')

axes[0,0].plot(df_nasa.index, df_nasa['Temperature_C'], color='tomato', linewidth=0.7)
axes[0,0].set_title('Bangladesh Monthly Temperature'); axes[0,0].set_ylabel('Celsius')

axes[0,1].bar(df_nasa.index, df_nasa['Precipitation_mm_day'],
              color='steelblue', width=20, alpha=0.7)
axes[0,1].set_title('Bangladesh Monthly Precipitation'); axes[0,1].set_ylabel('mm/day')

axes[1,0].plot(df_giss['Year'], df_giss['Annual_Anomaly_C'], color='darkred', linewidth=1.2)
axes[1,0].axhline(0, color='gray', linewidth=0.8, linestyle='--')
axes[1,0].set_title('Global Temperature Anomaly (NASA GISS)'); axes[1,0].set_ylabel('C anomaly')

axes[1,1].plot(df_sea['Year'], df_sea['Sea_Level_mm'], color='navy', linewidth=1.5)
axes[1,1].fill_between(df_sea['Year'], df_sea['Sea_Level_mm'],
                       df_sea['Sea_Level_mm'].min(), alpha=0.1, color='navy')
axes[1,1].set_title('Global Sea Level Rise'); axes[1,1].set_ylabel('mm')

plt.tight_layout()
plt.savefig(f'{FIG}/phase1_data_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Figure saved.")

# ── Final report ──────────────────────────────────────────────────────────────
print()
print("=" * 55)
print("  PHASE 1 COMPLETE")
print("=" * 55)
print(f"  Master dataset : {master.shape[0]} rows x {master.shape[1]} cols")
print(f"  Columns        : {master.columns.tolist()}")
print()
print("  Files in data/processed/:")
for f in sorted(os.listdir(PROC)):
    kb = os.path.getsize(f'{PROC}/{f}') // 1024
    print(f"    {f:<45} {kb} KB")
print("=" * 55)
print("  Next: Open 02_exploratory_data_analysis.ipynb")
