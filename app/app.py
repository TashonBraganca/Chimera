import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="Small Cap Fund Analyzer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def load_data():
    """
    Loads all the necessary data files from our project structure.
    Returns a dictionary of DataFrames.
    """
    base_dir = Path(__file__).resolve().parent.parent
    processed_data_dir = base_dir / "data" / "processed"
    
    ranked_funds_path = processed_data_dir / "final_ranked_funds.csv"
    nav_data_path = processed_data_dir / "master_nav_data.parquet"

    if not ranked_funds_path.exists() or not nav_data_path.exists():
        st.error("FATAL ERROR: Processed data files not found. Please run all previous phases (01, 02, 03) successfully.")
        return None

    data = {
        'ranked_funds': pd.read_csv(ranked_funds_path),
        'nav_data': pd.read_parquet(nav_data_path)
    }
    return data

data = load_data()

st.sidebar.title("Project Chimera")
st.sidebar.header("Navigation")

page = st.sidebar.radio( 
    "Go to",
    ("🏆 Top Recommendations", "📊 Fund Screener", "🆚 Deep-Dive Comparison")
)# Use a radio button for page navigation


st.sidebar.markdown("---")
st.sidebar.info(
    "This application is the output of a multi-phase quantitative analysis pipeline, "
    "designed to identify superior Small Cap funds in the Indian market."
)

#  Top recommendations page
if page == "🏆 Top Recommendations" and data:
    st.title("🏆 Top Small Cap Fund Recommendations")
    st.markdown("Based on a holistic, multi-factor analysis using the TOPSIS ranking methodology, the following funds have been identified as superior investment choices based on 3-year performance and risk data.")

    top_funds = data['ranked_funds'].head(5)

    st.subheader("Our Top 3 Picks")    #Top 3 funds in metric cards
    cols = st.columns(3)
    for i in range(3):
        with cols[i]:
            fund = top_funds.iloc[i]
            st.metric(
                label=f"Rank #{fund['rank']}: {fund['schemeName']}",
                value=f"{fund['CAGR_3Y']*100:.2f}% CAGR",
                delta=f"{fund['Sortino_Ratio_3Y']:.2f} Sortino Ratio",
                delta_color="off"
            )
    
    st.markdown("---")
    
    st.subheader("Top 10 Ranked Funds")    #The Top 10 funds table
    display_cols = [
        'rank', 'schemeName', 'topsis_score', 'alpha_ff_4f', 
        'alpha_p_value', 'Sortino_Ratio_3Y', 'Max_Drawdown_3Y', 'CAGR_3Y'
    ]
    display_cols = [col for col in display_cols if col in data['ranked_funds'].columns]
    
    st.dataframe(
        data['ranked_funds'][display_cols].head(10).style.format({
            'topsis_score': '{:.4f}',
            'alpha_ff_4f': '{:.4%}',
            'alpha_p_value': '{:.4f}',
            'Sortino_Ratio_3Y': '{:.2f}',
            'Max_Drawdown_3Y': '{:.2%}',
            'CAGR_3Y': '{:.2%}'
        })
    )
    
    st.info("Alpha p-value < 0.05 suggests the fund manager's 'skill' (alpha) is statistically significant.")

elif page == "📊 Fund Screener" and data:
    st.title("📊 Interactive Fund Screener")
    st.markdown("Analyze and compare all funds based on key performance and risk metrics.")

    st.subheader("Risk vs. Return Analysis (3-Year Horizon)")
    
    required_cols = ['CAGR_3Y', 'Volatility_3Y', 'Sortino_Ratio_3Y', 'schemeName']
    plot_df = data['ranked_funds'].dropna(subset=required_cols).copy()


    plot_df['Bubble_Size'] = plot_df['Sortino_Ratio_3Y'].clip(lower=0)

    fig = px.scatter(
        plot_df,
        x="Volatility_3Y",
        y="CAGR_3Y",
        size="Bubble_Size", # Use the new non-negative column for size
        color="Sortino_Ratio_3Y", # Keep original Sortino for color to show negative values
        hover_name="schemeName",
        color_continuous_scale=px.colors.sequential.Viridis,
        labels={
            "Volatility_3Y": "Annualized Volatility (Risk)",
            "CAGR_3Y": "Annualized Return (CAGR)",
            "Sortino_Ratio_3Y": "Sortino Ratio (Color)",
            "Bubble_Size": "Sortino Ratio (Size)"
        },
        title="3-Year Risk-Return Profile"
    )
    fig.update_layout(
        xaxis_title="Volatility (Lower is Better)",
        yaxis_title="Return (Higher is Better)",
        legend_title="Sortino Ratio"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info("The ideal fund is in the **top-left quadrant** (high return, low risk) with a **large, bright bubble** (high Sortino Ratio).")

    st.subheader("Complete Data Table")
    st.dataframe(data['ranked_funds'].style.format(precision=4))

elif page == "🆚 Deep-Dive Comparison" and data:
    st.title("🆚 Deep-Dive Comparison Tool")
    st.markdown("Select up to 4 funds to compare their historical performance and drawdown profiles.")

    all_funds = data['ranked_funds']['schemeName'].tolist()
    selected_funds = st.multiselect(
        "Select funds to compare:",
        options=all_funds,
        default=data['ranked_funds']['schemeName'].head(3).tolist(),
        max_selections=4
    )

    if selected_funds:
        fund_map = data['ranked_funds'].set_index('schemeName')['schemeCode'].astype(str)
        selected_codes = [fund_map[name] for name in selected_funds]
        
        all_codes_to_plot = selected_codes + ['NIFTY_SMALLCAP_250']
        
        plot_df = data['nav_data'][all_codes_to_plot].copy()

        # NUmeric type columns only  plotting.
        for col in plot_df.columns:
            plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
        
        # Normalize each column independently
        normalized_df = pd.DataFrame()
        for col in plot_df.columns:
            # Find the first valid (non-NaN) value for the column
            first_valid_value = plot_df[col].dropna().iloc[0]
            # Normalize the series by this first value
            normalized_df[col] = (plot_df[col] / first_valid_value) * 100
        
        # Rename columns for the legend
        name_map = {code: name for name, code in fund_map.items()}
        name_map['NIFTY_SMALLCAP_250'] = 'NIFTY Smallcap 250'
        normalized_df.rename(columns=name_map, inplace=True)

        # Plot 1: Cumulative Returns
        st.subheader("Cumulative Performance Growth")
        fig1 = px.line(
            normalized_df,
            title="Growth of ₹100 Invested",
            labels={"value": "Portfolio Value (₹)", "index": "Date", "variable": "Asset"}
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Plot 2: Drawdown "Underwater" Plot
        st.subheader("Drawdown Analysis")
        
        # Calculate drawdown robustly on the original (non-normalized) data.
        returns_df = plot_df.pct_change()
        cumulative_returns = (1 + returns_df).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns - peak) / peak
        drawdown.rename(columns=name_map, inplace=True)
        
        fig2 = px.area(
            drawdown * 100, # Convert to percentage
            title="Drawdown Periods (Underwater Chart)",
            labels={"value": "Drawdown (%)", "index": "Date", "variable": "Asset"}
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.info("This chart shows the percentage loss from the previous peak. It helps visualize the magnitude and duration of a fund's worst losing streaks.")

# Handle case where data loading fails
elif not data:
    st.error("Data loading failed. Please check the console for errors and ensure the data pipeline has been run successfully.")
