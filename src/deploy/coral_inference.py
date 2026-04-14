import sys
import time
from pathlib import Path
import joblib
import serial
import io

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.training_artifacts import ARTIFACT_PATHS
from src.utils import load_driving_data

repo_root = Path(__file__).resolve().parents[2]
DEFAULT_CSV = repo_root / "test-data" / "2025-07-12" / "2025_07_12-08.csv"
DEFAULT_ARTIFACT = ARTIFACT_PATHS["isolation-forest"]

# # Serial setup
# ser = serial.Serial(port='/dev/tty.usbserial-58DD0009681', baudrate=115200, timeout=1)
#         if ser.in_waiting > 0:  # Check if there is data in the buffer
#             line = ser.readline().decode('utf-8').rstrip()
#             print(line)
# e

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
    iso = cfg["model"]
    scaler = cfg["scaler"]
    feature_cols = cfg["feature_cols"]
    threshold = cfg["threshold"]

    df = load_driving_data(csv_path)

    if sleep_seconds < 0:
        raise ValueError("sleep_seconds must be >= 0.")

    # Simulate streaming delay between samples.
    for _, row in df.iterrows():
        X = scaler.transform([row[feature_cols].values])
        score = -iso.decision_function(X)[0]
        flag = int(score >= threshold)
        time_val = row["Time"] if "Time" in row else _
        print(f"t={time_val} score={score:.4f} flag={flag}")
        if sleep_seconds:
            time.sleep(sleep_seconds)

# TODO Receive input from ECE on coral board

if __name__ == "__main__":
    stream_scores(csv_path=DEFAULT_CSV, artifact_path=DEFAULT_ARTIFACT)
