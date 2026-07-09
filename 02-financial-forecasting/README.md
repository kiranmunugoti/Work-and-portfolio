# Financial Risk Forecasting Pipeline

Time-series forecasting pipeline for financial risk planning, combining
ARIMA and Holt-Winters exponential smoothing into an ensemble that is more
robust to noisy, incomplete operational data than either model alone.

In production this used ARIMA + Prophet (stacked/blended) to support
month-ahead financial risk planning on heterogeneous operational data. This
repo swaps Prophet for statsmodels' Holt-Winters to keep the dependency
footprint minimal, while preserving the same ensembling approach, on
**synthetic data** (real financial series are proprietary).

## Approach

- Generates a monthly series with trend, dual seasonality, noise, and a
  simulated disruption period.
- Fits ARIMA and Holt-Winters independently on a 36-month training window.
- Ensembles both forecasts (simple average, swappable for a learned blend)
  over a 12-month holdout.
- Evaluates each model and the ensemble with MAE and MAPE, and plots
  historical, actual, and forecast series together.

## Run it

```bash
pip install -r requirements.txt
python forecasting_pipeline.py
```

Outputs `forecast_chart.png` and `forecast_output.csv`.
