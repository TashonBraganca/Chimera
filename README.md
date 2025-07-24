#  Indian Small Cap Fund Analyzer

![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)
![Libraries](https://img.shields.io/badge/Libraries-Pandas%20%7C%20Streamlit%20%7C%20Plotly-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

This project is an interactive tool I built to analyze and recommend Small Cap mutual funds in the Indian market. My goal was to go beyond the standard metrics available on most retail platforms and apply a more rigorous, quantitative approach to fund selection, similar to what you might find in an institutional setting.

The result is a complete, end-to-end data pipeline and a live web application that provides objective, data-driven insights.

---

##  Live Application

The final output of this project is a live, interactive dashboard built with Streamlit.

**➡️https://chimera-v9tr844grufuxbdwm9mmlf.streamlit.app/)** 

---

##  Key Features & Analytical Approach

I designed this tool to be more advanced than standard platforms in a few key ways:

| Feature | Standard Approach | **My Approach** |
| :--- | :--- | :--- |
| **Risk Analysis** | Sharpe Ratio (Total Volatility) | **Sortino Ratio:** I chose this metric because it's a smarter way to measure risk. It only penalizes for harmful downside volatility, which better reflects a real investor's experience. |
| **Skill Assessment** | Basic Return Comparison | **Fama-French 4-Factor Regression:** Instead of just looking at returns, I wanted to see if a manager was genuinely skilled. This model isolates a manager's true **Alpha**, separating their performance from simple market movements. |
| **Ranking System** | Subjective Star Ratings | **TOPSIS Algorithm:** To ensure the recommendations are completely objective, I implemented the TOPSIS model. It's a formal, multi-criteria method that provides a transparent and mathematically sound ranking. |
| **Data Sourcing** | Manual Downloads / Static CSVs | **Automated & Resilient Pipeline:** The entire backend is an automated Python system that fetches and validates all data on its own, including academic factors via the `indiafactorlibrary`. |

---

##  How It Works: The Analysis Pipeline

The project is built as a series of scripts that create a repeatable and reliable analysis workflow.

### 1. Data Ingestion & Validation (`1 Data Analysis.py`)

This script automatically finds all relevant Small Cap funds, downloads their complete NAV history along with benchmark data, and then cleans and merges everything into a master dataset, ready for analysis.

### 2. Metric Calculation (`2 Metrics.py`)

This script takes the clean data and calculates a wide range of performance and risk metrics for every fund, including advanced measures like the Sortino Ratio, Max Drawdown, and Omega Ratio over multiple time horizons.

### 3. Ranking & Analysis (`3 Fund Rank Analysis.py`)

This is the core analytical engine. It downloads the Fama-French academic factor data, runs a regression for every fund to determine its true Alpha, and then uses the TOPSIS algorithm to produce the final, objective ranking based on all the data we've generated.

### 4. Web Application (`app/app.py`)

The final piece is the Streamlit app. It's a multi-page dashboard that presents the findings in an intuitive way, with interactive charts and tables that allow users to explore the results for themselves.

---

##  How to Run This Project Locally

### Prerequisites

- Python 3.8+
- A virtual environment (recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/TashonBraganca/Chimera.git
cd Chimera
```
### 2. Setup the Environment
```bash
Create and activate a virtual environment:
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```
### 3. Install Dependencies
```bash
All required libraries are listed in the requirements.txt file.
pip install -r requirements.txt
```
### 4. Run the Data Pipeline

#### Execute the scripts in order. It's a good idea to delete the data folder before the first run to ensure a clean start.
```bash
python notebooks/1 Data Analysis.py
python notebooks/2 Metrics.py
python notebooks/3 Fund Rank Analysis.py
```
### 5. Launch the Streamlit Application
Once the pipeline has successfully generated the analysis files, you can launch the web app.
```bash
streamlit run app/app.py
```

