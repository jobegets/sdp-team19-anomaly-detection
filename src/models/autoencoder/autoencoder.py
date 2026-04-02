import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ARTIFACT_PATH = (Path(__file__).resolve().parents[3] / "artifacts" / "autoencoder.pkl").resolve()


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 16, latent_dim: int = 8) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def _format_score_percentiles(scores: np.ndarray) -> str:
    """Format compact score percentile diagnostics."""
    pcts = np.quantile(scores, [0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])
    labels = ("min", "p25", "p50", "p75", "p90", "p99", "max")
    return ", ".join(f"{label}={value:.6g}" for label, value in zip(labels, pcts))


def _top_feature_error_contributors(
    inputs: np.ndarray,
    reconstructed: np.ndarray,
    feature_cols: list[str],
    top_k: int = 5,
) -> str:
    """Return the top mean per-feature reconstruction errors as a compact string."""
    feature_errors = np.mean((reconstructed - inputs) ** 2, axis=0)
    top_indices = np.argsort(feature_errors)[::-1][: min(top_k, len(feature_cols))]
    return ", ".join(
        f"{feature_cols[idx]}={feature_errors[idx]:.6g}" for idx in top_indices
    )


def run_autoencoder(
    driving_df: pd.DataFrame,
    threshold_quantile: float = 0.80,
    num_epochs: int = 60,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    hidden_dim: int = 16,
    latent_dim: int = 8,
    random_state: int = 42,
) -> tuple[pd.DataFrame, float, MinMaxScaler, Autoencoder, list[str]]:
    """Train a simple autoencoder on normal data and score all points."""
    required_cols = ["Time", "BMS Disch Enable"]
    for col in required_cols:
        if col not in driving_df.columns:
            raise KeyError(f"Expected column '{col}' not found in driving data.")

    feature_cols = [
        col
        for col in driving_df.select_dtypes(include=["number"]).columns
        if col not in ("Time", "BMS Disch Enable")
    ]
    if not feature_cols:
        raise ValueError("No numeric feature columns found for model training.")

    fault_flags = (driving_df["BMS Disch Enable"] == 0).astype(int)
    normal_mask = fault_flags == 0
    if not normal_mask.any():
        raise ValueError("No normal samples found (all BMS discharge disabled).")

    X_full = driving_df[feature_cols].values.astype(np.float32)
    X_normal = X_full[normal_mask.values]

    scaler = MinMaxScaler()
    X_normal_scaled = scaler.fit_transform(X_normal).astype(np.float32)
    X_full_scaled = scaler.transform(X_full).astype(np.float32)

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    model = Autoencoder(
        input_dim=len(feature_cols),
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
    )

    dataset = TensorDataset(torch.from_numpy(X_normal_scaled))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(num_epochs):
        for (batch,) in loader:
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_normal_tensor = torch.from_numpy(X_normal_scaled)
        reconstructed_normal = model(X_normal_tensor)
        normal_scores = torch.mean(
            (reconstructed_normal - X_normal_tensor) ** 2, dim=1
        ).numpy()

        X_tensor = torch.from_numpy(X_full_scaled)
        reconstructed = model(X_tensor)
        anomaly_scores = torch.mean((reconstructed - X_tensor) ** 2, dim=1).numpy()
        reconstructed_full = reconstructed.numpy()

    threshold = float(np.quantile(anomaly_scores, threshold_quantile))

    print("\n[autoencoder] Training diagnostics")
    print(f"Threshold quantile: {threshold_quantile:.3f}")
    print(f"Threshold: {threshold:.6g}")
    print(
        "Normal-train score percentiles: "
        f"{_format_score_percentiles(normal_scores)}"
    )
    print(
        "Train/all score percentiles: "
        f"{_format_score_percentiles(anomaly_scores)}"
    )
    print(
        "Top train feature reconstruction errors: "
        f"{_top_feature_error_contributors(X_full_scaled, reconstructed_full, feature_cols)}"
    )

    result_df = driving_df.copy()
    result_df["anomaly_score"] = anomaly_scores
    result_df["any_bms_fault"] = fault_flags.values
    result_df["anomalous_flag"] = (result_df["anomaly_score"] >= threshold).astype(int)

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_name": "autoencoder",
            "model": model,
            "scaler": scaler,
            "feature_cols": feature_cols,
            "threshold": threshold,
            "threshold_quantile": threshold_quantile,
        },
        ARTIFACT_PATH,
    )

    return result_df, threshold, scaler, model, feature_cols
