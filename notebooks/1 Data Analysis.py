import requests
import pandas as pd
import yfinance as yf
from pathlib import Path
import time
from tqdm import tqdm
import certifi

# --- 1. CONFIGURATION & SETUP ---
print("PHASE 1: DATA INGESTION & VALIDATION (v3 - Final)")
print("---" * 20)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_NAV_DIR = RAW_DATA_DIR / "nav"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_NAV_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

print(f"Project base directory set to: {BASE_DIR}")
print("All necessary directories are present.")

# --- 2. FETCH LIST OF SMALL CAP FUNDS ---

def get_small_cap_funds():
    print("\nStep 2: Fetching and filtering fund list...")
    try:
        funds_list_url = "https://api.mfapi.in/mf"
        response = requests.get(funds_list_url, verify=certifi.where())
        response.raise_for_status()
        all_funds = response.json()
        small_cap_funds = [
            fund for fund in all_funds
            if 'small cap' in fund['schemeName'].lower() and
               'direct' in fund['schemeName'].lower() and
               'growth' in fund['schemeName'].lower() and
               'idcw' not in fund['schemeName'].lower()
        ]
        print(f"  - Found {len(small_cap_funds)} matching Small Cap funds.")
        return small_cap_funds
    except requests.exceptions.RequestException as e:
        print(f"  - ERROR: Could not fetch fund list. {e}")
        return None

# --- 3. DOWNLOAD HISTORICAL NAV DATA ---

def download_fund_nav_history(scheme_code):
    file_path = RAW_NAV_DIR / f"{scheme_code}.csv"
    if file_path.exists(): return True
    try:
        nav_url = f"https://api.mfapi.in/mf/{scheme_code}"
        response = requests.get(nav_url, verify=certifi.where())
        response.raise_for_status()
        data = response.json().get('data')
        if not data: return False
        pd.DataFrame(data).to_csv(file_path, index=False)
        time.sleep(0.1)
        return True
    except (requests.exceptions.RequestException, KeyError, TypeError):
        return False

# --- 4. DOWNLOAD BENCHMARK DATA ---

def download_benchmark_data():
    print("\nStep 4: Downloading benchmark data...")
    benchmarks = { "NIFTY_50": "^NSEI", "NIFTY_SMALLCAP_250": "^NSMIDCP" }
    for name, ticker in benchmarks.items():
        try:
            print(f"  - Downloading {name}...")
            data = yf.download(ticker, period="max", interval="1d", auto_adjust=False)
            if not data.empty:
                data.to_csv(RAW_DATA_DIR / f"{name}.csv")
                print(f"  - Successfully downloaded {name} data.")
            else:
                print(f"  - WARNING: No data returned for {name} ({ticker}).")
        except Exception as e:
            print(f"  - ERROR: Could not download {name}. {e}")

# --- 5. DATA PROCESSING AND MERGING (DEFINITIVE FIX) ---

def process_and_merge_data():
    print("\nStep 5: Processing and Merging Data...")
    
    print("  - [5.1] Loading all downloaded NAV files...")
    all_nav_files = list(RAW_NAV_DIR.glob("*.csv"))
    if not all_nav_files:
        print("    - FATAL: No NAV files found in 'data/raw/nav'. Cannot proceed.")
        return None

    all_funds_df = pd.DataFrame()
    for file in tqdm(all_nav_files, desc="  - Processing NAV files"):
        try:
            df = pd.read_csv(file, usecols=['date', 'nav'])
            df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
            df.rename(columns={'nav': file.stem}, inplace=True)
            if all_funds_df.empty:
                all_funds_df = df
            else:
                all_funds_df = pd.merge(all_funds_df, df, on='date', how='outer')
        except Exception:
            continue

    if all_funds_df.empty:
        print("    - FATAL: Could not process any NAV files. Cannot proceed.")
        return None
        
    all_funds_df.set_index('date', inplace=True)
    all_funds_df.sort_index(inplace=True)
    all_funds_df.index = all_funds_df.index.normalize()
    print(f"    - DIAGNOSTIC: Shape after merging all NAVs: {all_funds_df.shape}")

    print("  - [5.2] Loading and preparing benchmark data...")
    nifty50_path = RAW_DATA_DIR / "NIFTY_50.csv"
    niftysmallcap_path = RAW_DATA_DIR / "NIFTY_SMALLCAP_250.csv"

    if not nifty50_path.exists() or not niftysmallcap_path.exists():
        print("    - FATAL: Benchmark CSV files not found. Cannot proceed.")
        return None

    # DEFINITIVE FIX: Explicitly convert index to datetime, then normalize.
    nifty50_df = pd.read_csv(nifty50_path, index_col=0)[['Adj Close']]
    nifty50_df.index = pd.to_datetime(nifty50_df.index, errors='coerce')
    nifty50_df.dropna(inplace=True) # Drop rows where date conversion failed
    nifty50_df.index = nifty50_df.index.normalize()
    nifty50_df.rename(columns={'Adj Close': 'NIFTY_50'}, inplace=True)

    niftysmallcap_df = pd.read_csv(niftysmallcap_path, index_col=0)[['Adj Close']]
    niftysmallcap_df.index = pd.to_datetime(niftysmallcap_df.index, errors='coerce')
    niftysmallcap_df.dropna(inplace=True) # Drop rows where date conversion failed
    niftysmallcap_df.index = niftysmallcap_df.index.normalize()
    niftysmallcap_df.rename(columns={'Adj Close': 'NIFTY_SMALLCAP_250'}, inplace=True)

    benchmarks_df = nifty50_df.join(niftysmallcap_df, how='outer')
    print(f"    - DIAGNOSTIC: Shape of combined benchmarks: {benchmarks_df.shape}")

    print("  - [5.3] Performing robust inner merge...")
    master_df = pd.merge(all_funds_df, benchmarks_df, left_index=True, right_index=True, how='inner')
    print(f"    - DIAGNOSTIC: Shape after robust inner merge: {master_df.shape}")

    if master_df.empty:
        print("    - FATAL: The DataFrame is empty after merging. This means there are no overlapping dates between NAV and benchmark data.")
        return None

    print("  - [5.4] Final cleaning and validation...")
    master_df.ffill(inplace=True)
    
    min_obs = 252
    original_cols = master_df.shape[1]
    master_df.dropna(axis=1, thresh=min_obs, inplace=True)
    dropped_cols = original_cols - master_df.shape[1]
    print(f"    - Dropped {dropped_cols} funds with less than 1 year of data.")
    print(f"    - DIAGNOSTIC: Shape after final cleaning: {master_df.shape}")

    if master_df.empty:
        print("    - FATAL: The final DataFrame is empty after cleaning. No data will be saved.")
        return None

    print("  - [5.5] Saving processed master data...")
    output_path = PROCESSED_DATA_DIR / "master_nav_data.parquet"
    master_df.to_parquet(output_path)
    print(f"    - SUCCESS! Master data saved to {output_path}")
    print(f"    - Final dataset shape: {master_df.shape}")
    
    return master_df

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    small_cap_funds = get_small_cap_funds()
    if small_cap_funds:
        print("\nStep 3: Downloading NAV history for all funds...")
        for fund in tqdm(small_cap_funds, desc="  - Downloading NAVs"):
            download_fund_nav_history(fund['schemeCode'])
        print("  - NAV download process complete.")
    
    download_benchmark_data()
    process_and_merge_data()

    print("\n---" * 20)
    print("PHASE 1 (v3) COMPLETE.")
