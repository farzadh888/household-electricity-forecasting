from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


NUMERIC_COLS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]


def load_hourly_energy(file_path: str) -> pd.Series:
    """
    Load the UCI household power consumption dataset and convert
    minute-level active power (kW) into hourly energy consumption (kWh).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path, sep=";", low_memory=False)

    existing_numeric = [col for col in NUMERIC_COLS if col in df.columns]
    df[existing_numeric] = df[existing_numeric].apply(pd.to_numeric, errors="coerce")

    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        dayfirst=True,
        errors="coerce",
    )

    df = df.dropna(subset=["datetime", "Global_active_power"]).set_index("datetime")

    # Convert minute-level power to energy in kWh
    df["energy_kwh"] = df["Global_active_power"] / 60.0

    # Aggregate to hourly energy
    hourly = df["energy_kwh"].resample("h").sum()

    # Fill small gaps
    hourly = hourly.interpolate()

    hourly.name = "energy"
    return hourly


def chronological_split(series: pd.Series, train_ratio: float = 0.8):
    """
    Chronological split for time-series.
    """
    split_idx = int(len(series) * train_ratio)
    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:]
    return train, test


def regression_metrics(y_true, y_pred) -> dict:
    """
    Compute regression metrics for forecasting.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    mean_target = np.mean(y_true)
    normalized_accuracy = np.nan
    if mean_target != 0:
        normalized_accuracy = 1 - (mae / mean_target)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "Normalized_Accuracy": float(normalized_accuracy),
    }


def save_metrics_table(metrics: dict, output_path: str) -> None:
    """
    Save a metrics dictionary to CSV.
    """
    df = pd.DataFrame([metrics])
    df.to_csv(output_path, index=False)