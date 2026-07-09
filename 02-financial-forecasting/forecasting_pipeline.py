"""
Financial Risk Forecasting Pipeline
--------------------------------------------------------
Forecasts a financial risk-relevant series (e.g. monthly cost variance)
using ARIMA and a Holt-Winters exponential smoothing model, then blends
both forecasts into an ensemble that is more robust to the kind of noisy,
incomplete operational data typical of finance ops reporting.

NOTE: This is a demonstration on synthetic data with realistic trend,
seasonality, and noise. In production this fed month-ahead financial risk
planning; ensembling (stacking ARIMA + Prophet forecasts) reduced variance
versus any single model on live data. Company data is proprietary, so this
repo reproduces the pipeline mechanics on generated data.

Author: Sai Kiran Munugoti
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

RNG = np.random.default_rng(7)


def generate_series(n_months=48):
    """Simulate a monthly financial series with trend + seasonality + noise."""
    t = np.arange(n_months)
    trend = 500 + 4.5 * t
    seasonality = 60 * np.sin(2 * np.pi * t / 12) + 25 * np.sin(2 * np.pi * t / 6)
    noise = RNG.normal(0, 30, n_months)
    shock = np.where((t > 30) & (t < 34), -120, 0)  # simulate a disruption period
    values = trend + seasonality + noise + shock
    dates = pd.date_range("2022-01-01", periods=n_months, freq="MS")
    return pd.Series(values, index=dates, name="risk_cost_index")


def fit_arima(train, order=(2, 1, 2), horizon=12):
    model = ARIMA(train, order=order).fit()
    forecast = model.forecast(steps=horizon)
    return forecast


def fit_holt_winters(train, horizon=12):
    model = ExponentialSmoothing(
        train, trend="add", seasonal="add", seasonal_periods=12
    ).fit()
    forecast = model.forecast(horizon)
    return forecast


def ensemble_forecast(arima_fc, hw_fc, weights=(0.5, 0.5)):
    return weights[0] * arima_fc.values + weights[1] * hw_fc.values


def evaluate(actual, predicted, label):
    mae = mean_absolute_error(actual, predicted)
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    print(f"{label:>18s}  MAE={mae:7.2f}   MAPE={mape:5.2f}%")
    return mae, mape


def plot_forecast(train, test, arima_fc, hw_fc, ens_fc, out_path="forecast_chart.png"):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train.index, train.values, label="Historical", color="#3a3a38", linewidth=1.6)
    ax.plot(test.index, test.values, label="Actual (holdout)", color="#3a3a38",
            linewidth=1.6, linestyle="--")
    ax.plot(test.index, arima_fc.values, label="ARIMA forecast", color="#a3b899", linewidth=1.4)
    ax.plot(test.index, hw_fc.values, label="Holt-Winters forecast", color="#d9a441", linewidth=1.4)
    ax.plot(test.index, ens_fc, label="Ensemble forecast", color="#b3543e", linewidth=2.2)
    ax.axvline(train.index[-1], color="#999", linestyle=":", linewidth=1)
    ax.set_title("Financial Risk Index — Forecast vs. Actual", fontsize=13, fontweight="bold")
    ax.set_ylabel("Risk Cost Index")
    ax.legend(frameon=False, fontsize=9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor="white")
    print(f"Saved chart to {out_path}")


def main():
    series = generate_series(48)
    train, test = series.iloc[:36], series.iloc[36:]

    arima_fc = fit_arima(train, horizon=len(test))
    hw_fc = fit_holt_winters(train, horizon=len(test))
    ens_fc = ensemble_forecast(arima_fc, hw_fc)

    print("Holdout evaluation (12-month horizon):\n")
    evaluate(test.values, arima_fc.values, "ARIMA")
    evaluate(test.values, hw_fc.values, "Holt-Winters")
    evaluate(test.values, ens_fc, "Ensemble")

    plot_forecast(train, test, arima_fc, hw_fc, ens_fc)

    out = pd.DataFrame({
        "actual": test.values, "arima": arima_fc.values,
        "holt_winters": hw_fc.values, "ensemble": ens_fc,
    }, index=test.index)
    out.to_csv("forecast_output.csv")


if __name__ == "__main__":
    main()
