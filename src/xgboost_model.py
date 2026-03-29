import pandas as pd
import numpy as np
from xgboost import XGBRegressor

from data_utils import regression_metrics


DEFAULT_LAGS = [1, 2, 3, 24, 48, 168]

def clean_series(series: pd.Series, threshold: float = 0.05) -> pd.Series:
    """
    Replace abnormal low values with NaN and interpolate.
    """
    series = series.copy()

    num_anomalies = (series < threshold).sum()
    print(f"[XGBoost] Detected {num_anomalies} low-consumption anomalies")

    # Replace with NaN
    series[series < threshold] = np.nan

    # Interpolate
    series = series.interpolate(method="time")

    # Drop remaining NaNs
    series = series.dropna()

    return series

def build_features(series: pd.Series, lags=None) -> pd.DataFrame:
    """
    Convert a univariate time-series into a supervised feature table.
    """
    if lags is None:
        lags = DEFAULT_LAGS

    df_feat = series.to_frame(name="energy")

    # Calendar features
    df_feat["hour"] = df_feat.index.hour
    df_feat["day_of_week"] = df_feat.index.dayofweek
    df_feat["month"] = df_feat.index.month
    df_feat["is_weekend"] = (df_feat.index.dayofweek >= 5).astype(int)

    # Lag features
    for lag in lags:
        df_feat[f"lag_{lag}"] = df_feat["energy"].shift(lag)

    df_feat = df_feat.dropna()
    return df_feat


def split_feature_table(df_feat: pd.DataFrame, train_ratio: float = 0.8):
    """
    Chronological split of the feature table.
    """
    split_idx = int(len(df_feat) * train_ratio)
    train_df = df_feat.iloc[:split_idx]
    test_df = df_feat.iloc[split_idx:]
    return train_df, test_df


def train_xgboost_model(X_train, y_train) -> XGBRegressor:
    """
    Train an XGBoost regressor.
    """
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def run_xgboost(hourly: pd.Series, train_ratio: float = 0.8, lags=None) -> dict:
    """
    Full XGBoost pipeline:
    - build lag/calendar features
    - split chronologically
    - train model
    - predict on train and test
    - return results and metrics
    """
    if lags is None:
        lags = DEFAULT_LAGS

    # ---------------------------
    # Clean anomalies (NEW)
    # ---------------------------
    hourly_clean = clean_series(hourly)
    print("\n--- Before cleaning ---")
    print(hourly.describe())

    print("\n--- After cleaning ---")
    print(hourly_clean.describe())
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12,4))
    plt.plot(hourly.index, hourly.values, label="Before")
    plt.plot(hourly_clean.index, hourly_clean.values, label="After")
    plt.legend()
    plt.title("Effect of Anomaly Cleaning")
    plt.show()

    df_feat = build_features(hourly_clean, lags=lags)

    train_df, test_df = split_feature_table(df_feat, train_ratio=train_ratio)

    feature_cols = [col for col in df_feat.columns if col != "energy"]

    X_train = train_df[feature_cols]
    y_train = train_df["energy"]

    X_test = test_df[feature_cols]
    y_test = test_df["energy"]

    model = train_xgboost_model(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_result = pd.DataFrame({
        "Actual": y_train,
        "Prediction": y_pred_train,
    }, index=train_df.index)
    train_result["Error"] = train_result["Actual"] - train_result["Prediction"]
    train_result["Absolute_Error"] = train_result["Error"].abs()

    test_result = pd.DataFrame({
        "Actual": y_test,
        "Prediction": y_pred_test,
    }, index=test_df.index)
    test_result["Error"] = test_result["Actual"] - test_result["Prediction"]
    test_result["Absolute_Error"] = test_result["Error"].abs()

    train_metrics = regression_metrics(y_train, y_pred_train)
    test_metrics = regression_metrics(y_test, y_pred_test)

    return {
        "model": model,
        "df_feat": df_feat,
        "train_df": train_df,
        "test_df": test_df,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "train_result": train_result,
        "test_result": test_result,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "lags": lags,
    }


def _build_single_feature_row(history: pd.Series, current_time: pd.Timestamp, lags) -> pd.DataFrame:
    """
    Build one feature row for recursive forecasting.
    """
    row = {
        "hour": current_time.hour,
        "day_of_week": current_time.dayofweek,
        "month": current_time.month,
        "is_weekend": int(current_time.dayofweek >= 5),
    }

    for lag in lags:
        row[f"lag_{lag}"] = history.iloc[-lag]

    return pd.DataFrame([row])


def forecast_future(model, history: pd.Series, steps: int, lags=None) -> pd.DataFrame:
    """
    Recursive multi-step forecasting.

    Parameters
    ----------
    model : trained XGBoost model
    history : pd.Series
        Full observed hourly series up to the last real timestamp.
    steps : int
        Number of future hourly steps to forecast.
    lags : list
        Lag values used during training.

    Returns
    -------
    forecast_df : pd.DataFrame
        Columns: Datetime, Forecast
    """
    if lags is None:
        lags = DEFAULT_LAGS

    history = history.copy()
    predictions = []
    future_times = []

    max_lag = max(lags)
    if len(history) < max_lag:
        raise ValueError(
            f"History is too short for lags={lags}. "
            f"Need at least {max_lag} observations, got {len(history)}."
        )

    for _ in range(steps):
        current_time = history.index[-1] + pd.Timedelta(hours=1)
        X_new = _build_single_feature_row(history, current_time, lags)
        y_hat = model.predict(X_new)[0]

        predictions.append(y_hat)
        future_times.append(current_time)

        history.loc[current_time] = y_hat

    forecast_df = pd.DataFrame({
        "Datetime": future_times,
        "Forecast": predictions,
    })

    return forecast_df