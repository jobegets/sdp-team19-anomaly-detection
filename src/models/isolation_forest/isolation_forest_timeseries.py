import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

from src.models.training_artifacts import save_training_artifact

# Trains and runs isolation forest on rolling-mean features
def run_isolation_forest(
    driving_df: pd.DataFrame,
    threshold_quantile: float = 0.80,
    window_size: int = 25,
    contamination: float = 0.01,
    random_state: int = 42,
    dataset_reference: str | None = None,
) -> tuple[pd.DataFrame, float, MinMaxScaler, IsolationForest, list[str]]:
    """Train IsolationForest on rolling-mean windows and flag anomalies.

    Args:
        driving_df: Preprocessed driving data.
        threshold_quantile: Quantile of anomaly scores used to set the alert threshold.
        window_size: Rolling window length for mean features.
        contamination: Expected proportion of anomalies for the IF model.
        random_state: Seed for reproducibility.

    Returns:
        result_df: Window-aligned copy with new anomaly_score, any_bms_fault, anomalous_flag.
        threshold: Score cutoff derived from threshold_quantile.
        scaler: Fitted MinMaxScaler.
        iso: Fitted IsolationForest model.
        feature_cols: List of original feature column names used for training.

    Raises:
        KeyError: If required columns (Time, BMS Disch Enable) are missing.
        ValueError: If no numeric features, no normal windows, or too few samples for the window_size.
    """
    
    if "Time" not in driving_df.columns:
        raise KeyError("Expected column 'Time' not found in driving data.")

    if "BMS Disch Enable" not in driving_df.columns:
        raise KeyError("Expected column 'BMS Disch Enable' not found in driving data.")

    # Binary label per timestep: 1 if BMS discharge disabled (fault), 0 otherwise
    fault_flags = (driving_df["BMS Disch Enable"] == 0).astype(int)

    # Feature columns for the window (exclude Time and label)
    feature_cols = [
        col
        for col in driving_df.select_dtypes(include=["number"]).columns
        if col not in ("Time", "BMS Disch Enable")
    ]

    if not feature_cols:
        raise ValueError("No numeric feature columns found for sliding window.")

    if len(driving_df) < window_size:
        raise ValueError(
            f"Not enough samples ({len(driving_df)}) for window size {window_size}."
        )

    # Rolling mean per feature; aligned to window end
    rolling_means = (
        driving_df[feature_cols]
        .rolling(window=window_size, min_periods=window_size)
        .mean()
        .iloc[window_size - 1 :]
    )

    fault_flags_seq = fault_flags.values[window_size - 1 :]
    times_seq = driving_df["Time"].values[window_size - 1 :]

    # Normal windows: end point is non-fault
    normal_mask = fault_flags_seq == 0
    if not normal_mask.any():
        raise ValueError("No normal windows found (all BMS discharge disabled).")

    X_full = rolling_means.values
    X_normal = X_full[normal_mask]

    scaler = MinMaxScaler()
    X_normal_scaled = scaler.fit_transform(X_normal)
    X_full_scaled = scaler.transform(X_full)

    iso = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )

    iso.fit(X_normal_scaled)

    # The normality score for every window
    decision_vals = iso.decision_function(X_full_scaled)
    anomaly_scores = -decision_vals

    # Result dataframe at window-level (aligned with window ends)
    result_df = driving_df.iloc[window_size - 1 :].copy()
    result_df["Time"] = times_seq
    result_df["anomaly_score"] = anomaly_scores
    result_df["any_bms_fault"] = fault_flags_seq

    # Flag anomalies based on given threshold quantile
    threshold = float(np.quantile(anomaly_scores, threshold_quantile))
    result_df["anomalous_flag"] = (result_df["anomaly_score"] >= threshold).astype(int)

    save_training_artifact(
        model_name="isolation-forest-timeseries",
        dataset_reference=dataset_reference,
        threshold=threshold,
        scaler=scaler,
        model=iso,
        feature_cols=feature_cols,
        contamination=contamination,
        threshold_quantile=threshold_quantile,
        window_size=window_size,
    )

    return result_df, threshold, scaler, iso, feature_cols
