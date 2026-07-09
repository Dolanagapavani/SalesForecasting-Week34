# Sales Forecasting & Demand Intelligence System

Internship Project — Week 3 & 4

## What this project does

Builds an end-to-end sales forecasting system on 4 years of Superstore retail data.
Predicts future product demand, detects unusual sales patterns, segments products
by demand behavior, and presents everything through an interactive Streamlit dashboard.

## Dataset

`train.csv` — 9,994 transactions across 4 years
(Order Date, Ship Date, Category, Sub-Category, Region, Sales, Quantity, Profit)

## What's in this repo

- `analysis.ipynb` — complete notebook with all 8 tasks
- `app.py` — Streamlit dashboard with 4 interactive pages
- `train.csv` — Superstore sales dataset
- `requirements.txt` — all libraries needed to run the project
- `summary.pdf` — 2-page executive business report
- `charts/` — all saved chart images

## Steps done

1. Data loading, datetime parsing and feature extraction
2. Time series decomposition (Trend, Seasonal, Residual)
3. ADF stationarity test and differencing
4. Built 3 forecasting models — SARIMA, Prophet, XGBoost
5. Compared models using MAE, RMSE, MAPE
6. Segment-level forecasting (Category + Region)
7. Anomaly detection using Isolation Forest and Z-Score
8. Product demand clustering using K-Means + PCA
9. Deployed interactive Streamlit dashboard

## Key Findings

- Technology is the highest revenue category (~$836K over 4 years)
- November/December spike 40-45% above average every year
- West region shows the most consistent sales growth
- Prophet model recommended for production use
- 4 demand clusters identified — each needs a different stocking strategy

## How to Run

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the dashboard:
```bash
streamlit run app.py
```
