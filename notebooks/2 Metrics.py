import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- 1. CONFIGURATION & SETUP ---
print("PHASE 2: MODELING & METRICS (v3 - Final)")
print("---" * 20)

# Define project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Input file from Phase 1
MASTER_DATA_PATH = PROCESSED_DATA_DIR / "master_nav_data.parquet"
# Output file for this phase
METRICS_OUTPUT_PATH = PROCESSED_DATA_DIR / "fund_metrics.csv"
# We also need to generate the scheme details file for Phase 3
SCHEME_DETAILS_PATH = PROCESSED_DATA_DIR / "scheme_details.csv"


# --- 2. LOAD DATA & VALIDATE TYPES ---
print(f"Step 2.1: Loading processed data from {MASTER_DATA_PATH}...")
if not MASTER_DATA_PATH.exists():
    print(f"FATAL ERROR: Master data file not found. Please run Phase 1 script first.")
    exit()

master_df = pd.read_parquet(MASTER_DATA_PATH)
print(f"Successfully loaded data with shape: {master_df.shape}")

# DEFINITIVE FIX: Enforce numeric data types for ALL columns to prevent TypeErrors.
print("Step 2.2: Enforcing numeric data types for all columns...")
for col in tqdm(master_df.columns, desc="  - Converting columns"):
    master_df[col] = pd.to_numeric(master_df[col], errors='coerce')
print("  - Data type validation complete.")

# --- 3. CALCULATE DAILY RETURNS ---
print("\nStep 3: Calculating daily returns...")
returns_df = master_df.pct_change()
# The first row will be all NaN, so we drop it.
returns_df.dropna(how='all', inplace=True)
print(f"Daily returns calculated. Shape: {returns_df.shape}")


# --- 4. DEFINE METRIC CALCULATION FUNCTIONS ---
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.07 # Placeholder, will be replaced by dynamic rate later

def calculate_cagr(series):
    if series.empty or len(series) < 2: return np.nan
    total_return = (1 + series).prod()
    years = len(series) / TRADING_DAYS_PER_YEAR
    return (total_return ** (1 / years)) - 1 if years > 0 else np.nan

def calculate_annualized_volatility(series):
    if series.empty: return np.nan
    return series.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

def calculate_sharpe_ratio(series, risk_free_rate):
    cagr = calculate_cagr(series)
    volatility = calculate_annualized_volatility(series)
    if pd.isna(cagr) or pd.isna(volatility) or volatility == 0: return np.nan
    return (cagr - risk_free_rate) / volatility

def calculate_sortino_ratio(series, risk_free_rate):
    cagr = calculate_cagr(series)
    if pd.isna(cagr): return np.nan
    downside_returns = series[series < 0]
    if downside_returns.empty: return np.inf
    downside_std = downside_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    if downside_std == 0: return np.nan
    return (cagr - risk_free_rate) / downside_std

def calculate_max_drawdown(series):
    if series.empty: return np.nan
    cumulative_returns = (1 + series).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()

def calculate_calmar_ratio(series):
    cagr = calculate_cagr(series)
    max_dd = calculate_max_drawdown(series)
    if pd.isna(cagr) or pd.isna(max_dd) or max_dd == 0: return np.nan
    return cagr / abs(max_dd)

def calculate_omega_ratio(series, required_return=0.0):
    if len(series) < 2: return np.nan
    daily_req_return = (1 + required_return)**(1/TRADING_DAYS_PER_YEAR) - 1
    returns_less_thresh = series - daily_req_return
    numer = returns_less_thresh[returns_less_thresh > 0].sum()
    denom = -1 * returns_less_thresh[returns_less_thresh < 0].sum()
    return numer / denom if denom != 0 else np.inf

def calculate_information_ratio(fund_returns, benchmark_returns):
    if fund_returns.empty or benchmark_returns.empty: return np.nan
    active_return = fund_returns - benchmark_returns
    tracking_error = active_return.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    if tracking_error == 0: return np.nan
    return active_return.mean() * TRADING_DAYS_PER_YEAR / tracking_error


# --- 5. COMPUTE METRICS FOR ALL FUNDS ---
print("\nStep 5: Computing metrics for all funds over multiple horizons...")

fund_columns = [col for col in master_df.columns if 'NIFTY' not in col]
benchmark_column = 'NIFTY_SMALLCAP_250'
all_metrics = []
time_horizons_years = [1, 3, 5]

for fund_code in tqdm(fund_columns, desc="Calculating Metrics"):
    if len(returns_df[fund_code].dropna()) < TRADING_DAYS_PER_YEAR:
        continue

    fund_metrics = {'scheme_code': fund_code}
    for years in time_horizons_years:
        start_date = returns_df.index.max() - pd.DateOffset(years=years)
        period_returns = returns_df.loc[start_date:, fund_code].dropna()
        
        if len(period_returns) < TRADING_DAYS_PER_YEAR * years * 0.9:
            continue
            
        benchmark_period_returns = returns_df.loc[period_returns.index, benchmark_column].dropna()

        fund_metrics[f'CAGR_{years}Y'] = calculate_cagr(period_returns)
        fund_metrics[f'Volatility_{years}Y'] = calculate_annualized_volatility(period_returns)
        fund_metrics[f'Sharpe_Ratio_{years}Y'] = calculate_sharpe_ratio(period_returns, RISK_FREE_RATE)
        fund_metrics[f'Sortino_Ratio_{years}Y'] = calculate_sortino_ratio(period_returns, RISK_FREE_RATE)
        fund_metrics[f'Max_Drawdown_{years}Y'] = calculate_max_drawdown(period_returns)
        fund_metrics[f'Calmar_Ratio_{years}Y'] = calculate_calmar_ratio(period_returns)
        fund_metrics[f'Omega_Ratio_{years}Y'] = calculate_omega_ratio(period_returns)
        fund_metrics[f'Information_Ratio_{years}Y'] = calculate_information_ratio(period_returns, benchmark_period_returns)

    if len(fund_metrics) > 1:
        all_metrics.append(fund_metrics)

print(f"Successfully calculated metrics for {len(all_metrics)} funds.")

# --- 6. SAVE METRICS TO CSV ---
if not all_metrics:
    print("\nWARNING: No funds had sufficient data to calculate metrics. The output file will be empty.")
    open(METRICS_OUTPUT_PATH, 'w').close()
else:
    print(f"\nStep 6: Saving calculated metrics to {METRICS_OUTPUT_PATH}...")
    metrics_df = pd.DataFrame(all_metrics)
    
    try:
        print("  - Fetching latest scheme names for merging...")
        import requests
        import certifi
        response = requests.get("https://api.mfapi.in/mf", verify=certifi.where())
        all_funds_list = response.json()
        scheme_details_df = pd.DataFrame(all_funds_list)
        scheme_details_df['schemeCode'] = scheme_details_df['schemeCode'].astype(str)
        scheme_details_df[['schemeCode', 'schemeName']].to_csv(SCHEME_DETAILS_PATH, index=False)
        
        metrics_df = pd.merge(scheme_details_df[['schemeCode', 'schemeName']], metrics_df, left_on='schemeCode', right_on='scheme_code')
        cols = ['schemeCode', 'schemeName'] + [c for c in metrics_df.columns if c not in ['schemeCode', 'schemeName', 'scheme_code']]
        metrics_df = metrics_df[cols]
        metrics_df.to_csv(METRICS_OUTPUT_PATH, index=False)
        print(f"SUCCESS! Metrics data saved.")
    except Exception as e:
        print(f"  - WARNING: Could not fetch scheme names. Saving with codes only. Error: {e}")
        metrics_df.to_csv(METRICS_OUTPUT_PATH, index=False)


print("\n---" * 20)
print("PHASE 2 (v3) COMPLETE. You are now ready to run Phase 3.")