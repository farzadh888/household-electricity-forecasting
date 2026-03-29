import pandas as pd
import matplotlib.pyplot as plt

from naive_baseline import run_naive_baseline
from xgboost_model import run_xgboost
from sarima_model import run_sarima
from nbeats_model import run_nbeats
from patchtst_model import run_patchtst
from data_utils import load_hourly_energy


def main():
    # Load hourly series directly from raw dataset
    hourly = load_hourly_energy(
        "/content/drive/MyDrive/Electricity/Dataset/household_power_consumption.txt"
    )

    results = []
    preds = {}
    actual_series = None

    # =========================
    # Naive baseline
    # =========================
    naive_out = run_naive_baseline(hourly, train_ratio=0.8)

    naive_result = naive_out["result"]
    naive_metrics = naive_out["metrics"]

    results.append([
        "Naive",
        naive_metrics["MAE"],
        naive_metrics["RMSE"],
        naive_metrics["Normalized_Accuracy"] * 100,
    ])

    preds["Naive"] = naive_result["Prediction"]
    actual_series = naive_result["Actual"]

    # =========================
    # XGBoost
    # =========================
    xgb_out = run_xgboost(hourly, train_ratio=0.8)

    xgb_result = xgb_out["test_result"]
    xgb_metrics = xgb_out["test_metrics"]

    results.append([
        "XGBoost",
        xgb_metrics["MAE"],
        xgb_metrics["RMSE"],
        xgb_metrics["Normalized_Accuracy"] * 100,
    ])

    preds["XGBoost"] = xgb_result["Prediction"]

    # =========================
    # SARIMA
    # =========================
    sarima_out = run_sarima(hourly, train_ratio=0.8)

    sarima_result = sarima_out["test_result"]
    sarima_metrics = sarima_out["metrics"]

    results.append([
        "SARIMA",
        sarima_metrics["MAE"],
        sarima_metrics["RMSE"],
        sarima_metrics["Normalized_Accuracy"] * 100,
    ])

    preds["SARIMA"] = sarima_result["Prediction"]

    # =========================
    # N-BEATS
    # =========================
    nbeats_out = run_nbeats(hourly, train_ratio=0.8)

    nbeats_result = nbeats_out["test_result"]
    nbeats_metrics = nbeats_out["test_metrics"]

    results.append([
        "N-BEATS",
        nbeats_metrics["MAE"],
        nbeats_metrics["RMSE"],
        nbeats_metrics["Normalized_Accuracy"] * 100,
    ])

    preds["N-BEATS"] = nbeats_result["Prediction"]

    # =========================
    # PatchTST
    # =========================
    patchtst_out = run_patchtst(hourly, train_ratio=0.8)

    patchtst_result = patchtst_out["test_result"]
    patchtst_metrics = patchtst_out["test_metrics"]

    results.append([
        "PatchTST",
        patchtst_metrics["MAE"],
        patchtst_metrics["RMSE"],
        patchtst_metrics["Normalized_Accuracy"] * 100,
    ])

    preds["PatchTST"] = patchtst_result["Prediction"]

    # =========================
    # Align predictions with actual series
    # =========================
    if actual_series is not None:
        common_index = actual_series.index.copy()

        for name in list(preds.keys()):
            preds[name] = preds[name].reindex(common_index)

    # =========================
    # Results table
    # =========================
    results_df = pd.DataFrame(
        results,
        columns=["Model", "MAE", "RMSE", "Normalized_Accuracy (%)"]
    )

    print("\n=== Model Comparison ===")
    print(results_df.to_string(index=False))

    results_df.to_csv("model_comparison.csv", index=False)
    print("\nSaved metrics to: model_comparison.csv")

    # =========================
    # Better plots
    # =========================
    common_index = actual_series.index
    for name, pred in preds.items():
        common_index = common_index.intersection(pred.index)

    actual_aligned = actual_series.loc[common_index]

    # ---------------------------------
    # Plot 1: Main comparison (best models only)
    # ---------------------------------
    plt.figure(figsize=(14, 6))
    plt.plot(common_index, actual_aligned.values, label="Actual", linewidth=2)

    main_models = ["XGBoost", "N-BEATS", "Naive"]
    for name in main_models:
        if name in preds:
            plt.plot(common_index, preds[name].loc[common_index].values, label=name)

    plt.legend()
    plt.title("Main Model Comparison on Test Set")
    plt.xlabel("Time")
    plt.ylabel("Hourly Energy (kWh)")
    plt.tight_layout()
    plt.savefig("comparison_main.png")
    print("Saved plot to: comparison_main.png")

    # ---------------------------------
    # Plot 2: Zoomed comparison (all models)
    # ---------------------------------
    zoom_hours = 24 * 7   # last 7 days
    zoom_index = common_index[-zoom_hours:]

    plt.figure(figsize=(14, 6))
    plt.plot(zoom_index, actual_aligned.loc[zoom_index].values, label="Actual", linewidth=2)

    for name, pred in preds.items():
        plt.plot(zoom_index, pred.loc[zoom_index].values, label=name)

    plt.legend()
    plt.title("Zoomed Model Comparison on Last 7 Days of Test Set")
    plt.xlabel("Time")
    plt.ylabel("Hourly Energy (kWh)")
    plt.tight_layout()
    plt.savefig("comparison_zoom.png")
    print("Saved plot to: comparison_zoom.png")

if __name__ == "__main__":
    main()