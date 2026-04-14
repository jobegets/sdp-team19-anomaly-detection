import sys
import time
from collections import deque
from pathlib import Path

import joblib
import numpy as np
# import serial
# import io

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.training_artifacts import ARTIFACT_PATHS
from src.utils import load_driving_data

repo_root = Path(__file__).resolve().parents[2]
DEFAULT_CSV = repo_root / "test-data" / "2025-07-12" / "2025_07_12-08.csv"
DEFAULT_ARTIFACT = ARTIFACT_PATHS["isolation-forest-timeseries"]

# # Serial setup
# ser = serial.Serial(port='/dev/tty.usbserial-58DD0009681', baudrate=115200, timeout=1)
#         if ser.in_waiting > 0:  # Check if there is data in the buffer
#             line = ser.readline().decode('utf-8').rstrip()
#             print(line)
# e


def _print_summary(
    inference_latencies_ms: list[float],
    preprocessing_latencies_ms: list[float],
    *,
    total_rows: int,
    scored_rows: int,
) -> None:
    """Print a compact profiling summary for edge deployment evaluation."""
    print("\nProfiling summary")
    print(f"Rows read: {total_rows}")
    print(f"Rows scored: {scored_rows}")

    if inference_latencies_ms:
        inference_arr = np.array(inference_latencies_ms)
        preprocessing_arr = np.array(preprocessing_latencies_ms)
        print(
            "End-to-end latency (ms): "
            f"mean={inference_arr.mean():.3f}, "
            f"p50={np.percentile(inference_arr, 50):.3f}, "
            f"p95={np.percentile(inference_arr, 95):.3f}, "
            f"max={inference_arr.max():.3f}"
        )
        print(
            "Preprocessing latency (ms): "
            f"mean={preprocessing_arr.mean():.3f}, "
            f"p95={np.percentile(preprocessing_arr, 95):.3f}, "
            f"max={preprocessing_arr.max():.3f}"
        )


def _score_isolation_forest(model: object, model_input: np.ndarray, scaler: object) -> float:
    """Score one feature vector with a classic isolation forest."""
    X_scaled = scaler.transform([model_input])
    return float(-model.decision_function(X_scaled)[0])


def _score_autoencoder(model: object, model_input: np.ndarray, scaler: object) -> float:
    """Score one feature vector with a dense autoencoder."""
    import torch

    X_scaled = scaler.transform([model_input]).astype(np.float32)
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_scaled)
        reconstructed = model(X_tensor)
        score = torch.mean((reconstructed - X_tensor) ** 2, dim=1).item()
    return float(score)


def _score_lstm_autoencoder(model: object, model_input: np.ndarray, scaler: object) -> float:
    """Score one feature sequence with an LSTM autoencoder."""
    import torch

    X_scaled = scaler.transform(model_input).astype(np.float32)
    X_batched = np.expand_dims(X_scaled, axis=0)
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_batched)
        reconstructed = model(X_tensor)
        score = torch.mean((reconstructed - X_tensor) ** 2, dim=(1, 2)).item()
    return float(score)


def stream_scores(csv_path: Path, artifact_path: Path, sleep_seconds: float = 0.05) -> None:
    """Simulate streaming sensor data and print anomaly scores using a pre-trained model.

    Args:
        csv_path: Path to the CSV containing sensor readings.
        artifact_path: Path to a joblib artifact with model, scaler, feature_cols, and threshold.
        sleep_seconds: Delay between emitted rows in seconds. Use 0 for no delay.

    Raises:
        FileNotFoundError: If the artifact path does not exist.
        
    Returns:
        None. Streams scores and logs inference results.
    """
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")

    cfg = joblib.load(artifact_path)

    model = cfg["model"]
    scaler = cfg["scaler"]
    feature_cols = cfg["feature_cols"]
    threshold = cfg["threshold"]
    model_name = cfg.get("model_name")
    window_size = cfg.get("window_size")

    df = load_driving_data(csv_path)

    if sleep_seconds < 0:
        raise ValueError("sleep_seconds must be >= 0.")

    if model_name is None:
        raise ValueError("Artifact is missing required 'model_name'.")

    sequence_model_names = {"isolation-forest-timeseries", "autoencoder-lstm"}
    if model_name in sequence_model_names and window_size is None:
        raise ValueError(
            f"{model_name} artifact is missing required 'window_size'."
        )

    window_buffer: deque[np.ndarray] | None = None
    if model_name in sequence_model_names:
        window_buffer = deque(maxlen=window_size)

    if model_name in {"autoencoder", "autoencoder-lstm"}:
        model.eval()

    inference_latencies_ms: list[float] = []
    preprocessing_latencies_ms: list[float] = []

    # Simulate streaming delay between samples.
    for _, row in df.iterrows():
        row_start = time.perf_counter()
        feature_values = row[feature_cols].to_numpy(dtype=np.float32)

        if model_name == "isolation-forest-timeseries":
            assert window_buffer is not None
            window_buffer.append(feature_values)
            if len(window_buffer) < window_size:
                time_val = row["Time"] if "Time" in row else _
                print(f"t={time_val} warming_up={len(window_buffer)}/{window_size}")
                if sleep_seconds:
                    time.sleep(sleep_seconds)
                continue
            model_input = np.mean(np.stack(window_buffer, axis=0), axis=0)
            score_fn = _score_isolation_forest
        elif model_name == "isolation-forest":
            model_input = feature_values
            score_fn = _score_isolation_forest
        elif model_name == "autoencoder":
            model_input = feature_values
            score_fn = _score_autoencoder
        elif model_name == "autoencoder-lstm":
            assert window_buffer is not None
            window_buffer.append(feature_values)
            if len(window_buffer) < window_size:
                time_val = row["Time"] if "Time" in row else _
                print(f"t={time_val} warming_up={len(window_buffer)}/{window_size}")
                if sleep_seconds:
                    time.sleep(sleep_seconds)
                continue
            model_input = np.stack(window_buffer, axis=0)
            score_fn = _score_lstm_autoencoder
        else:
            raise ValueError(f"Unsupported artifact model_name: {model_name}")

        preprocessing_latency_ms = (time.perf_counter() - row_start) * 1000
        score = score_fn(model, model_input, scaler)
        flag = int(score >= threshold)
        inference_latency_ms = (time.perf_counter() - row_start) * 1000

        preprocessing_latencies_ms.append(preprocessing_latency_ms)
        inference_latencies_ms.append(inference_latency_ms)

        time_val = row["Time"] if "Time" in row else _
        print(
            f"t={time_val} score={score:.4f} flag={flag} latency_ms={inference_latency_ms:.3f}"
        )
        if sleep_seconds:
            time.sleep(sleep_seconds)

    _print_summary(
        inference_latencies_ms,
        preprocessing_latencies_ms,
        total_rows=len(df),
        scored_rows=len(inference_latencies_ms),
    )

# TODO Receive input from ECE on coral board

if __name__ == "__main__":
    stream_scores(csv_path=DEFAULT_CSV, artifact_path=DEFAULT_ARTIFACT)
