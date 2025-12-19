# Finance & Economics Time-Series Analysis: Macroeconomic Forecasting

Forecasting **GDP Growth**, **Inflation**, and **Unemployment** using classical time-series baselines (ARIMA, VAR), deep learning (LSTM), and a feature-engineered machine learning approach (XGBoost).

This project was completed as the capstone for the **NYU Tandon Career Hub Data Science Bootcamp**.

## Project Summary
We worked with a daily dataset spanning **2000–2008** containing **24 economic and financial variables**.  
Our goal was to build forecasting models that meaningfully track three macroeconomic indicators:
- GDP Growth
- Inflation
- Unemployment

We found that the dataset is highly volatile and weakly autocorrelated, which limited the effectiveness of standard sequence-based models. After extensive feature engineering (lags, rolling stats, interactions, temporal features), **XGBoost** performed best and achieved strong predictive performance (R² > 0.95 in our experiments).

## Contents
- `notebooks/` – end-to-end workflow (cleaning, EDA, feature engineering, modeling, evaluation)
- `src/` – reusable scripts for preprocessing, modeling, evaluation (optional if using notebooks only)
- `reports/figures/` – saved charts and visuals
- `reports/slides/` – final project slides
- `results/` – metrics and prediction outputs

## Methods
### 1) Exploratory Data Analysis (EDA)
Key observations:
- GDP and Inflation showed mild cyclical behavior; Unemployment was smoother and slower-moving
- Correlations were weak (GDP vs Unemployment moderately negative; GDP vs Inflation near zero)
- Series appeared volatile and non-stationary
- Lag structure was weak (limited autocorrelation, fast decay)

### 2) Feature Engineering
To better capture economic dynamics and reduce noise, we created:
- Lag features and rolling statistics
- Interaction features (ratios/differences between key indicators)
- Temporal features (year, month, quarter, month-end flags)

### 3) Models Evaluated
- ARIMA (baseline)
- VAR (multivariate baseline)
- LSTM (sequence model)
- XGBoost (primary model with engineered features)

## Results
Model performance summary:
- **XGBoost**: strong fit and stability, **R² > 0.95** for all targets in our experiments
- **ARIMA / VAR / LSTM**: struggled due to volatility + weak sequential structure, **R² ≈ 0**

See `results/metrics/` for detailed metrics and `reports/figures/` for plots.

## How to Run
### Option A: Run notebooks
1. Create and activate an environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   # .venv\Scripts\activate   # Windows
