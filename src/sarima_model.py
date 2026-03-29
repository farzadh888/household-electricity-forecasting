import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from data_utils import chronological_split, regression_metrics

def clean_series(series: pd.Series, threshold: float = 0.05) -> pd.Series:
    series = series.copy()

    num_anomalies = (series < threshold).sum()
    print(f"[Cleaning] Detected {num_anomalies} low-consumption anomalies")

    series[series < threshold] = np.nan
    series = series.interpolate(method="time")
    series = series.dropna()

    return series

def run_sarima(hourly: pd.Series, train_ratio: float = 0.8) -> dict:
    """
    Train SARIMA on the training split and evaluate on the test split.
    """
    hourly = clean_series(hourly)
    train, test = chronological_split(hourly, train_ratio=train_ratio)
    print(f"Train size: {len(train)}")
    print(f"Test size: {len(test)}")

    model = SARIMAX(
        train,
        order=(1, 0, 1),
        seasonal_order=(1, 0, 1, 24),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    model_fit = model.fit(disp=False)

    forecast = model_fit.forecast(steps=len(test))
    forecast = pd.Series(forecast, index=test.index, name="Prediction")

    test_result = pd.DataFrame({
        "Actual": test,
        "Prediction": forecast,
    }, index=test.index)

    test_result["Error"] = test_result["Actual"] - test_result["Prediction"]
    test_result["Absolute_Error"] = test_result["Error"].abs()

    metrics = regression_metrics(test_result["Actual"], test_result["Prediction"])

    print("\n=== SARIMA Results ===")
    print(f"MAE: {metrics['MAE']:.4f} kWh")
    print(f"RMSE: {metrics['RMSE']:.4f} kWh")
    print(f"Normalized Accuracy: {metrics['Normalized_Accuracy'] * 100:.2f}%")

    plt.figure(figsize=(12, 5))
    plt.plot(test.index, test, label="Actual")
    plt.plot(test.index, forecast, label="SARIMA Forecast")
    plt.legend()
    plt.title("SARIMA Forecast vs Actual")
    plt.tight_layout()
    plt.savefig("sarima_plot.png")
    print("Plot saved to: sarima_plot.png")

    return {
        "model": model_fit,
        "train": train,
        "test": test,
        "test_result": test_result,
        "metrics": metrics,
    }