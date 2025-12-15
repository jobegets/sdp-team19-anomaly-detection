import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

if __package__ is None or __package__ == "":
    # Allow running as a script from repo root: python src/models/isolation_forest.py
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils import (
    load_driving_data,
    plot_features_over_time,
    plot_results,
    print_evaluation,
)

repo_root = Path(__file__).resolve().parents[2]
CSV_PATH = (repo_root / "test-data" / "7-12-2025" / "fsae-7-12 (8).csv").expanduser().resolve()
ARTIFACT_PATH = (repo_root / "artifacts" / "isolation_forest.pkl").expanduser().resolve()

# Trains and runs isolation forest
# TODO figure out how to best tune hyperparameters. Also tuning threshold quantile.
def run_isolation_forest(
    driving_df: pd.DataFrame,
    contamination: float = 0.01,
    threshold_quantile: float = 0.80,
    random_state: int = 42,
) -> tuple[pd.DataFrame, float, MinMaxScaler, IsolationForest, list[str]]:
    """Train IsolationForest on normal data and score all points.

    Args:
        driving_df: Preprocessed driving data.
        contamination: Expected proportion of anomalies for the IF model.
        threshold_quantile: Quantile of anomaly scores used to set the alert threshold.
        random_state: Seed for reproducibility.

    Returns:
        result_df: Copy of input with new anomaly_score, any_bms_fault, anomalous_flag.
        threshold: Score cutoff derived from threshold_quantile.
        scaler: Fitted MinMaxScaler.
        iso: Fitted IsolationForest model.
        feature_cols: List of original feature column names used for training.

    Raises:
        KeyError: If required columns (Time, BMS Disch Enable) are missing.
        ValueError: If no usable numeric features or no normal samples are available.
    """
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

    # Normalize features on fault-free data and score across all points
    scaler = MinMaxScaler()
    X_full = driving_df[feature_cols].values
    X_normal = X_full[normal_mask.values]
    X_normal_scaled = scaler.fit_transform(X_normal)
    X_full_scaled = scaler.transform(X_full)

    iso = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )

    iso.fit(X_normal_scaled) # Train model
    
    # The normality score for every point
    # Initially: higher = more normal, lower = more anomalous
    decision_vals = iso.decision_function(X_full_scaled) 
    
    # By standard practice, flip the meanings
    anomaly_scores = -decision_vals

    result_df = driving_df.copy()
    result_df["anomaly_score"] = anomaly_scores
    result_df["any_bms_fault"] = fault_flags.values

    # Flag anomalies based on given threshold quantile
    threshold = float(np.quantile(anomaly_scores, threshold_quantile))
    result_df["anomalous_flag"] = (result_df["anomaly_score"] >= threshold).astype(int)

    return result_df, threshold, scaler, iso, feature_cols


if __name__ == "__main__":
    driving_df = load_driving_data(CSV_PATH)
    plot_features_over_time(driving_df)
    driving_df, threshold, scaler, iso, feature_cols = run_isolation_forest(driving_df)

    # Save model artifact for deployment
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": iso,
            "scaler": scaler,
            "feature_cols": feature_cols,
            "threshold": threshold,
            "threshold_quantile": 0.80,
        },
        ARTIFACT_PATH,
    )
    print(f"Saved model artifact to {ARTIFACT_PATH}")

    plot_results(driving_df, threshold)
    print_evaluation(driving_df, lookback_seconds=15.0)
