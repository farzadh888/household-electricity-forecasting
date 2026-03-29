# ⚡ Household Energy Forecasting

### Baseline, Statistical, Machine Learning, and Deep Learning Models

This project develops a time-series forecasting system to predict **hourly household electricity consumption** using the UCI Household Power Consumption dataset. It evaluates multiple modeling paradigms, from simple baselines to advanced deep learning architectures.

---

## 🎯 Objective

To compare forecasting performance across different approaches:

* Baseline (Naive)
* Statistical (SARIMA)
* Machine Learning (XGBoost)
* Deep Learning (N-BEATS, PatchTST)

---

## ⚙️ Models

### 1. Naive (Markov-like) Baseline

Predicts the next value using the previous observation:

```
x̂_t = x_{t-1}
```

**Insight:**
A strong baseline due to the local smoothness of electricity consumption.

---

### 2. SARIMA (Statistical Model)

Seasonal ARIMA with daily seasonality:

* Order: (1, 0, 1)
* Seasonal order: (1, 1, 1, 24)

**Captures:**

* Linear temporal dependencies
* Daily repeating patterns

**Limitations:**

* Cannot model nonlinear behavior
* Cannot capture multiple seasonalities (e.g., weekly cycles)

---

### 3. XGBoost (Machine Learning Model)

Uses engineered features:

**Lag features:**

* 1, 2, 3, 24, 48, 168

**Calendar features:**

* Hour
* Day of week
* Month
* Weekend indicator

**Advantages:**

* Captures nonlinear relationships
* Handles multi-scale seasonality
* Achieves best overall performance

---

### 4. N-BEATS (Deep Learning)

A fully connected neural architecture for time-series forecasting:

* Learns temporal patterns directly from data
* No manual feature engineering required
* Uses backcast/forecast residual learning

---

### 5. PatchTST (Transformer Model)

Transformer-based model for time-series forecasting:

* Splits sequences into patches
* Applies attention mechanisms
* Designed for long-term dependencies

---

## 📊 Dataset Processing

The raw dataset is recorded at the **minute level**. Processing steps:

1. Load dataset
2. Convert numeric columns safely
3. Create datetime index from `Date` and `Time`
4. Remove invalid rows
5. Convert power to energy:

```python
energy_kwh = Global_active_power / 60
```

6. Resample to hourly frequency
7. Interpolate missing values

---

## 📊 Results Summary

| Model    | Performance        |
| -------- | ------------------ |
| XGBoost  | 🥇 Best overall    |
| N-BEATS  | 🥈 Best deep model |
| Naive    | Strong baseline    |
| PatchTST | Moderate           |
| SARIMA   | Weakest            |

---

## 📈 Model Comparison (Last 7 Days of Test Set)

Comparison of model predictions against actual energy consumption.

![Model Comparison](output/model_comparison_zoom.png)


## 🧠 Key Insights

* Electricity consumption is **locally smooth**, making Naive surprisingly strong
* Feature engineering + nonlinearity (XGBoost) provides the best performance
* Deep learning models help, but do not guarantee improvement
* SARIMA struggles due to linear assumptions and limited seasonality modeling
* Multi-seasonality (daily + weekly) is critical

---

## 🔮 Forecasting Capability

The system supports **recursive multi-step forecasting (up to 1 year)**:

* Short-term predictions → accurate
* Long-term predictions → pattern-based
* Error accumulates over time

---

## 🧩 Project Strengths

* Clean and reproducible pipeline
* Multiple modeling paradigms
* Fair comparison framework
* Real-world dataset
* Interpretable results

---

## 📥 Dataset

UCI Individual Household Electric Power Consumption Dataset
https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

**License:** CC BY 4.0

Place the dataset in:

```
data/household_power_consumption.txt
```

---

## 🚀 How to Run

Install dependencies:

```
pip install -r requirements.txt
```

Run the comparison:

```
python src/compare_models.py
```

---

## 🧾 Conclusion

XGBoost outperforms both statistical and deep learning models due to its ability to capture **nonlinear and multi-seasonal patterns**, while N-BEATS is the strongest deep learning alternative.

---

## 📌 Future Work

* Incorporate multi-seasonal statistical models (e.g., TBATS, Prophet)
* Improve Transformer performance with better tuning
* Add probabilistic forecasting (uncertainty estimation)
* Extend to real-time energy prediction systems

---
