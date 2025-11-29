import sys
import time
from pathlib import Path
import joblib

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils import load_driving_data

repo_root = Path(__file__).resolve().parents[2]
DEFAULT_CSV = repo_root / "test-data" / "7-12-2025" / "fsae-7-12 (8).csv"
DEFAULT_ARTIFACT = repo_root / "artifacts" / "isolation_forest.pkl"

def stream_scores(csv_path: Path, artifact_path: Path) -> None:
    """Simulate streaming sensor data and print anomaly scores using a pre-trained model."""
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")

    cfg = joblib.load(artifact_path)
    iso = cfg["model"]
    scaler = cfg["scaler"]
    feature_cols = cfg["feature_cols"]
    threshold = cfg["threshold"]

    df = load_driving_data(csv_path)

    # simulate streaming at ~0.5 ms per sample
    for _, row in df.iterrows():
        X = scaler.transform([row[feature_cols].values])
        score = -iso.decision_function(X)[0]
        flag = int(score >= threshold)
        time_val = row["Time"] if "Time" in row else _
        print(f"t={time_val} score={score:.4f} flag={flag}")
        time.sleep(0.05)

if __name__ == "__main__":
    stream_scores(csv_path=DEFAULT_CSV, artifact_path=DEFAULT_ARTIFACT)
