import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from data_utils import regression_metrics


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ============================================================
# Config
# ============================================================

DEFAULT_BACKCAST_LENGTH = 168
DEFAULT_FORECAST_LENGTH = 24


# ============================================================
# Cleaning
# ============================================================

def clean_series(series: pd.Series, threshold: float = 0.05) -> pd.Series:
    """
    Replace abnormal low values with NaN and interpolate.
    """
    series = series.copy()

    num_anomalies = (series < threshold).sum()
    print(f"[N-BEATS] Detected {num_anomalies} low-consumption anomalies")

    series[series < threshold] = np.nan
    series = series.interpolate(method="time")
    series = series.dropna()

    return series


# ============================================================
# Windowing
# ============================================================

def create_windows(
    values: np.ndarray,
    backcast_length: int,
    forecast_length: int
) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []

    max_start = len(values) - backcast_length - forecast_length + 1
    for i in range(max_start):
        X.append(values[i:i + backcast_length])
        y.append(values[i + backcast_length:i + backcast_length + forecast_length])

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# Model
# ============================================================

class NBeatsBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        forecast_size: int,
        hidden_size: int = 128,
        n_layers: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()

        layers = []
        in_features = input_size
        for _ in range(n_layers):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden_size

        self.fc = nn.Sequential(*layers)
        self.theta = nn.Linear(hidden_size, input_size + forecast_size)

        self.input_size = input_size

    def forward(self, x: torch.Tensor):
        h = self.fc(x)
        theta = self.theta(h)
        backcast = theta[:, :self.input_size]
        forecast = theta[:, self.input_size:]
        return backcast, forecast


class NBeats(nn.Module):
    def __init__(
        self,
        input_size: int,
        forecast_size: int,
        num_blocks: int = 4,
        hidden_size: int = 128,
        n_layers: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            NBeatsBlock(
                input_size=input_size,
                forecast_size=forecast_size,
                hidden_size=hidden_size,
                n_layers=n_layers,
                dropout=dropout
            )
            for _ in range(num_blocks)
        ])

        self.forecast_size = forecast_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        forecast = torch.zeros(
            (x.size(0), self.forecast_size),
            dtype=x.dtype,
            device=x.device
        )

        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast

        return forecast


# ============================================================
# Training helpers
# ============================================================

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0
    total_samples = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        preds = model(X_batch)
        loss = loss_fn(preds, y_batch)
        loss.backward()
        optimizer.step()

        batch_size = X_batch.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_loss(model, loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    total_samples = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        preds = model(X_batch)
        loss = loss_fn(preds, y_batch)

        batch_size = X_batch.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / max(total_samples, 1)


@torch.no_grad()
def predict_model(model, loader, device):
    model.eval()
    all_preds = []
    all_true = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        preds = model(X_batch).cpu().numpy()

        all_preds.append(preds)
        all_true.append(y_batch.numpy())

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    return y_true, y_pred


# ============================================================
# Main runner
# ============================================================

def run_nbeats(
    hourly: pd.Series,
    train_ratio: float = 0.8,
    backcast_length: int = DEFAULT_BACKCAST_LENGTH,
    forecast_length: int = DEFAULT_FORECAST_LENGTH,
    val_ratio: float = 0.1,
    num_blocks: int = 4,
    hidden_size: int = 128,
    n_layers: int = 3,
    dropout: float = 0.2,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 5e-4,
    early_stopping_patience: int = 10,
    device: str | None = None,
) -> dict:
    """
    Full N-BEATS pipeline.
    """
    hourly = clean_series(hourly)

    split_idx = int(len(hourly) * train_ratio)
    train_series = hourly.iloc[:split_idx].copy()
    test_series = hourly.iloc[split_idx:].copy()

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_series.values.reshape(-1, 1)).flatten()
    test_scaled = scaler.transform(test_series.values.reshape(-1, 1)).flatten()

    X_train, y_train = create_windows(train_scaled, backcast_length, forecast_length)
    X_test, y_test = create_windows(test_scaled, backcast_length, forecast_length)

    val_size = int(len(X_train) * val_ratio)
    if val_size > 0:
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        X_train_final = X_train[:-val_size]
        y_train_final = y_train[:-val_size]
    else:
        X_val = np.empty((0, backcast_length), dtype=np.float32)
        y_val = np.empty((0, forecast_length), dtype=np.float32)
        X_train_final = X_train
        y_train_final = y_train

    train_loader = DataLoader(
        WindowDataset(X_train_final, y_train_final),
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        WindowDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False
    )

    val_loader = None
    if len(X_val) > 0:
        val_loader = DataLoader(
            WindowDataset(X_val, y_val),
            batch_size=batch_size,
            shuffle=False
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = NBeats(
        input_size=backcast_length,
        forecast_size=forecast_length,
        num_blocks=num_blocks,
        hidden_size=hidden_size,
        n_layers=n_layers,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    best_val_loss = math.inf
    best_state = None
    patience_counter = 0

    print("\nTraining N-BEATS...")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)

        if val_loader is not None:
            val_loss = evaluate_loss(model, val_loader, loss_fn, device)
            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break
        else:
            print(f"Epoch {epoch:03d}/{epochs} | Train Loss: {train_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print("Loaded best model from validation phase.")

    y_true_scaled, y_pred_scaled = predict_model(model, test_loader, device)

    # Use only 1-step-ahead forecast for aligned comparison
    y_true_1 = y_true_scaled[:, 0]
    y_pred_1 = y_pred_scaled[:, 0]

    y_true_inv = scaler.inverse_transform(y_true_1.reshape(-1, 1)).flatten()
    y_pred_inv = scaler.inverse_transform(y_pred_1.reshape(-1, 1)).flatten()

    test_metrics = regression_metrics(y_true_inv, y_pred_inv)

    # Align to hourly timestamps
    test_start_idx = backcast_length
    test_result_index = test_series.index[test_start_idx:test_start_idx + len(y_true_inv)]

    test_result = pd.DataFrame({
        "Actual": y_true_inv,
        "Prediction": y_pred_inv,
    }, index=test_result_index)
    test_result["Error"] = test_result["Actual"] - test_result["Prediction"]
    test_result["Absolute_Error"] = test_result["Error"].abs()

    return {
        "model": model,
        "scaler": scaler,
        "train_series": train_series,
        "test_series": test_series,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred_test": y_pred_inv,
        "test_result": test_result,
        "test_metrics": test_metrics,
    }