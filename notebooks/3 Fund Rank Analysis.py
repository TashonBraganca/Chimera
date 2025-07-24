
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MASTER_DATA_PATH = PROCESSED_DATA_DIR / "master_nav_data.parquet"
METRICS_PATH = PROCESSED_DATA_DIR / "fund_metrics.csv"
FF_FACTORS_PATH = RAW_DATA_DIR / "India_4_Factors.csv" # Using 4 factors now

FINAL_RANKING_PATH = PROCESSED_DATA_DIR / "final_ranked_funds.csv"

def download_fama_french_factors():
    """
    Downloads Fama-French 4-Factor data using the `indiafactorlibrary`.
    """
    if FF_FACTORS_PATH.exists():
        try:
            df_check = pd.read_csv(FF_FACTORS_PATH)
            if 'Date' in df_check.columns and 'Mkt-RF' in df_check.columns:
                print("Step 2: Valid Fama-French data already exists. Skipping download.")
                return
            else:
                print("  - WARNING: Stale Fama-French data file detected. Deleting and re-downloading.")
                FF_FACTORS_PATH.unlink()
        except Exception:
            print("  - WARNING: Corrupt Fama-French data file detected. Deleting and re-downloading.")
            FF_FACTORS_PATH.unlink()


    print("Downloading Fama-French 4-Factor data via indiafactorlibrary")
    try:
        from indiafactorlibrary import IndiaFactorLibrary
        ifl = IndiaFactorLibrary()
        
        dataset = ifl.read('ff4')
        
        ff_df = dataset[0]
        
        for col in ff_df.columns:
            ff_df[col] = pd.to_numeric(ff_df[col], errors='coerce') / 100
        
        ff_df.reset_index(inplace=True)
        
        date_col_name = ff_df.columns[0]
        ff_df.rename(columns={date_col_name: 'Date'}, inplace=True)

        if 'MKT' in ff_df.columns and 'RF' in ff_df.columns:
            print("  - Calculating 'Mkt-RF' from 'MKT' and 'RF' columns.")
            ff_df['Mkt-RF'] = ff_df['MKT'] - ff_df['RF']
        else:
            raise ValueError("Downloaded data does not contain 'MKT' and 'RF' columns needed to calculate the market factor.")

        ff_df.to_csv(FF_FACTORS_PATH, index=False)
        print("Successfully downloaded and processed Fama-French 4-Factor data.")

    except Exception as e:
        print(f"FATAL ERROR: Could not download Fama-French data using the library. {e}")
        print("Please ensure 'indiafactorlibrary' and 'lxml' are installed (`pip install indiafactorlibrary lxml`).")
        exit()

def calculate_fama_french_alpha(returns_df, ff_factors_df):
    """
    Calculates the Fama-French 4-Factor Alpha for each fund.
    """
    print("\nCalculating Fama-French 4-Factor Alpha for all funds")
    fund_columns = [col for col in returns_df.columns if 'NIFTY' not in col]
    
    
    monthly_returns = returns_df[fund_columns].resample('ME').apply(lambda x: (1 + x).prod() - 1)
    
    
    ff_factors_df['Date'] = pd.to_datetime(ff_factors_df['Date'])
    ff_factors_df.set_index('Date', inplace=True)
    
    merged_df = monthly_returns.join(ff_factors_df, how='inner')
    
    alphas = []
    for fund_code in tqdm(fund_columns, desc="Running Regressions"):
        if fund_code not in merged_df.columns or 'RF' not in merged_df.columns: continue
        
        y = merged_df[fund_code] - merged_df['RF']
        X = merged_df[['Mkt-RF', 'SMB', 'HML', 'WML']]
        X = sm.add_constant(X)
        
        y.dropna(inplace=True)
        X = X.loc[y.index].dropna()
        y = y.loc[X.index]

        if len(y) < 36: continue 

        model = sm.OLS(y, X).fit()
        annual_alpha = model.params['const'] * 12
        p_value = model.pvalues['const']
        alphas.append({'scheme_code': fund_code, 'alpha_ff_4f': annual_alpha, 'alpha_p_value': p_value})
    
    print(f"  - Successfully calculated 4-Factor Alpha for {len(alphas)} funds.")
    return pd.DataFrame(alphas)


def topsis_ranking(metrics_df):
    print("\nRanking funds using TOPSIS algorithm")
    criteria = {
        'CAGR_3Y': 1, 'Volatility_3Y': -1, 'Sortino_Ratio_3Y': 1,
        'Max_Drawdown_3Y': 1, 'Calmar_Ratio_3Y': 1, 'Omega_Ratio_3Y': 1,
        'Information_Ratio_3Y': 1, 'alpha_ff_4f': 1
    }
    rank_df = metrics_df.dropna(subset=list(criteria.keys())).copy()
    if rank_df.empty:
        print("WARNING: No funds had complete data for all TOPSIS criteria. Cannot perform ranking.")
        return pd.DataFrame()

    matrix = rank_df[list(criteria.keys())].values
    scaler = MinMaxScaler()
    normalized_matrix = scaler.fit_transform(matrix)
    weights = np.ones(len(criteria)) / len(criteria)
    weighted_matrix = normalized_matrix * weights
    directions = np.array(list(criteria.values()))
    ideal_best = np.where(directions == 1, weighted_matrix.max(axis=0), weighted_matrix.min(axis=0))
    ideal_worst = np.where(directions == 1, weighted_matrix.min(axis=0), weighted_matrix.max(axis=0))
    dist_best = np.linalg.norm(weighted_matrix - ideal_best, axis=1)
    dist_worst = np.linalg.norm(weighted_matrix - ideal_worst, axis=1)
    topsis_score = dist_worst / (dist_best + dist_worst + 1e-6)
    rank_df['topsis_score'] = topsis_score
    rank_df['rank'] = rank_df['topsis_score'].rank(ascending=False, method='first').astype(int)
    print(f"Successfully ranked {len(rank_df)} funds.")
    return rank_df.sort_values('rank')


if __name__ == "__main__":
    print("\nLoading data from previous phases")
    if not METRICS_PATH.exists() or METRICS_PATH.stat().st_size == 0:
        print(f"FATAL ERROR: The metrics file is missing or empty: {METRICS_PATH}")
        exit()
    if not MASTER_DATA_PATH.exists():
        print(f"FATAL ERROR: Master data file not found: {MASTER_DATA_PATH}")
        exit()

    metrics_df = pd.read_csv(METRICS_PATH)
    master_df = pd.read_parquet(MASTER_DATA_PATH)
    for col in tqdm(master_df.columns, desc="  - Validating data types"):
        master_df[col] = pd.to_numeric(master_df[col], errors='coerce')
    returns_df = master_df.pct_change()

    download_fama_french_factors()
    ff_factors_df = pd.read_csv(FF_FACTORS_PATH)
    
    alpha_df = calculate_fama_french_alpha(returns_df, ff_factors_df)
    
    if not alpha_df.empty:
        metrics_df['schemeCode'] = metrics_df['schemeCode'].astype(str)#mergeing metric along wtithe most imporant aspect : teh alphas
        alpha_df['scheme_code'] = alpha_df['scheme_code'].astype(str)
        
        combined_df = pd.merge(metrics_df, alpha_df, left_on='schemeCode', right_on='scheme_code', how='left')
    else:
        print("WARNING:Alpha calculation did not return any results. Proceeding without Alpha.")
        combined_df = metrics_df
        if 'alpha_ff_4f' not in combined_df.columns:
            combined_df['alpha_ff_4f'] = np.nan
    
    final_ranked_df = topsis_ranking(combined_df)

    if not final_ranked_df.empty:
        print(f"\nStep 5: Saving final ranked list to {FINAL_RANKING_PATH}...")
        final_ranked_df.to_csv(FINAL_RANKING_PATH, index=False)
        
        
        
        #Prints the top 10 funds in the list
        print("\n --- TOP 10 FUNDS ---\n")
        display_cols = ['rank', 'schemeName', 'topsis_score', 'alpha_ff_4f', 'alpha_p_value', 'Sortino_Ratio_3Y', 'Max_Drawdown_3Y']
        display_cols = [col for col in display_cols if col in final_ranked_df.columns]
        print(final_ranked_df[display_cols].head(10).round(4))
    else:
        print("\nNo funds were ranked. Final CSV not created.")

