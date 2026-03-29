import pandas as pd
import numpy as np

from data_utils import chronological_split, regression_metrics

def clean_series(series: pd.Series, threshold: float = 0.05) -> pd.Series:
    series = series.copy()

    num_anomalies = (series < threshold).sum()
    print(f"[Cleaning] Detected {num_anomalies} low-consumption anomalies")

    series[series < threshold] = np.nan
    series = series.interpolate(method="time")
    series = series.dropna()

    return series

def naive_markov_predict(test: pd.Series) -> pd.DataFrame:
    """
    Naive / Markov-like baseline:
    prediction(t) = actual(t-1)

    It is called Markov-like because the next prediction depends only
    on the immediately previous observed value.
    """
    result = pd.DataFrame({
        "Actual": test,
        "Input_t_minus_1": test.shift(1),
    })

    result = result.dropna()
    result["Prediction"] = result["Input_t_minus_1"]
    result["Error"] = result["Actual"] - result["Prediction"]
    result["Absolute_Error"] = result["Error"].abs()

    return result


def run_naive_baseline(hourly: pd.Series, train_ratio: float = 0.8) -> dict:
    """
    Run the naive baseline on the test region.
    """
    hourly = clean_series(hourly)
    train, test = chronological_split(hourly, train_ratio=train_ratio)

    result = naive_markov_predict(test)
    metrics = regression_metrics(result["Actual"], result["Prediction"])

    return {
        "train": train,
        "test": test,
        "result": result,
        "metrics": metrics,
    }