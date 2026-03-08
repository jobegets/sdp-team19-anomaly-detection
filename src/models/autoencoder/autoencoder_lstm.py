import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ARTIFACT_PATH = (
    Path(__file__).resolve().parents[3] / "artifacts" / "autoencoder_lstm.pkl"
).resolve()


class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        latent_dim: int = 16,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder(x)
        latent = self.to_latent(hidden[-1])
        repeated = self.from_latent(latent).unsqueeze(1).repeat(1, x.size(1), 1)
        decoded, _ = self.decoder(repeated)
        return self.output(decoded)


def run_autoencoder(
    driving_df: pd.DataFrame,
    threshold_quantile: float = 0.80,
    window_size: int = 25,
    num_epochs: int = 50,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    hidden_dim: int = 32,
    latent_dim: int = 16,
    num_layers: int = 1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, float, MinMaxScaler, LSTMAutoencoder, list[str]]:
    """Train an LSTM autoencoder on sliding windows and score all windows."""
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

    if len(driving_df) < window_size:
        raise ValueError(
            f"Not enough samples ({len(driving_df)}) for window size {window_size}."
        )

    fault_flags = (driving_df["BMS Disch Enable"] == 0).astype(int)
    X_full = driving_df[feature_cols].values.astype(np.float32)

    sequences = np.array(
        [X_full[i : i + window_size] for i in range(len(X_full) - window_size + 1)],
        dtype=np.float32,
    )
    fault_flags_seq = fault_flags.values[window_size - 1 :]
    normal_mask = fault_flags_seq == 0
    if not normal_mask.any():
        raise ValueError("No normal windows found (all BMS discharge disabled).")

    normal_sequences = sequences[normal_mask]

    scaler = MinMaxScaler()
    scaler.fit(normal_sequences.reshape(-1, len(feature_cols)))

    normal_sequences_scaled = scaler.transform(
        normal_sequences.reshape(-1, len(feature_cols))
    ).reshape(normal_sequences.shape)
    full_sequences_scaled = scaler.transform(
        sequences.reshape(-1, len(feature_cols))
    ).reshape(sequences.shape)

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    model = LSTMAutoencoder(
        input_dim=len(feature_cols),
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_layers=num_layers,
    )

    dataset = TensorDataset(torch.from_numpy(normal_sequences_scaled.astype(np.float32)))
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
        X_tensor = torch.from_numpy(full_sequences_scaled.astype(np.float32))
        reconstructed = model(X_tensor)
        anomaly_scores = torch.mean((reconstructed - X_tensor) ** 2, dim=(1, 2)).numpy()

    threshold = float(np.quantile(anomaly_scores, threshold_quantile))

    result_df = driving_df.iloc[window_size - 1 :].copy()
    result_df["anomaly_score"] = anomaly_scores
    result_df["any_bms_fault"] = fault_flags_seq
    result_df["anomalous_flag"] = (result_df["anomaly_score"] >= threshold).astype(int)

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_name": "autoencoder-lstm",
            "model": model,
            "scaler": scaler,
            "feature_cols": feature_cols,
            "threshold": threshold,
            "threshold_quantile": threshold_quantile,
            "window_size": window_size,
        },
        ARTIFACT_PATH,
    )

    return result_df, threshold, scaler, model, feature_cols
