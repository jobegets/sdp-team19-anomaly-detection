import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots


repo_root = Path(__file__).resolve().parents[2]
CSV_PATH = (repo_root / "test-data" / "7-12-2025" / "fsae-7-12 (8).csv").expanduser().resolve()


# Loads and cleans driving data
def load_driving_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    columns = [
        "Time",
        "BMS Disch Enable",
        # Battery
        "Pack Voltage","Pack Current","Pack Temp","State of Charge","Min Cell Voltage","BMS LV input",
        # Powertrain / inverter / motor
        "Torque Feedback",
        "RPM",
        "Flux Feedback",
        # Dynamics
        "InlineAcc","LateralAcc","VerticalAcc",
        "BrakeBias",
        "RollRate","PitchRate","YawRate",
    ]
    
    
    df = pd.read_csv(csv_path, skiprows=[1], usecols=columns)
    df.columns = df.columns.str.strip().str.replace('"', "")
    print(df.columns)
    df = df.dropna(subset=["Time"]).sort_values("Time")

    # Only want data where car is moving
    driving_df = df[df["RPM"] > 0.5].copy()
    
    return driving_df

# Trains and runs isolation forest using a sliding time window over features
def run_isolation_forest(
    driving_df: pd.DataFrame,
    contamination: float = 0.01,
    threshold_quantile: float = 0.80,
    random_state: int = 42,
    window_size: int = 25,
) -> tuple[pd.DataFrame, float]:
    
    if "Time" not in driving_df.columns:
        raise KeyError("Expected column 'Time' not found in driving data.")

    if "BMS Disch Enable" not in driving_df.columns:
        raise KeyError("Expected column 'BMS Disch Enable' not found in driving data.")

    # Binary label per timestep: 1 if BMS discharge disabled (fault), 0 otherwise
    labels_t = (driving_df["BMS Disch Enable"] == 0).astype(int)

    # Feature columns for the window (exclude Time and label)
    feature_cols = [
        col
        for col in driving_df.select_dtypes(include=["number"]).columns
        if col not in ("Time", "BMS Disch Enable")
    ]

    if not feature_cols:
        raise ValueError("No numeric feature columns found for sliding window.")

    values = driving_df[feature_cols].values
    times = driving_df["Time"].values

    if len(values) <= window_size:
        raise ValueError(
            f"Not enough samples ({len(values)}) for window size {window_size}."
        )

    # Build sliding windows: each row is the flattened last `window_size` samples
    num_windows = len(values) - window_size + 1
    num_features = len(feature_cols) * window_size

    X_seq = np.zeros((num_windows, num_features), dtype=float)
    for i in range(num_windows):
        window = values[i : i + window_size]
        X_seq[i] = window.flatten()

    # Align labels and times to the end of each window
    labels_seq = labels_t.values[window_size - 1 :]
    times_seq = times[window_size - 1 :]

    # Normal windows: end point is non-fault
    normal_mask = labels_seq == 0

    X_normal = X_seq[normal_mask]
    X_full = X_seq

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
    result_df["any_bms_fault"] = labels_seq

    # Flag anomalies based on given threshold quantile
    threshold = float(np.quantile(anomaly_scores, threshold_quantile))
    result_df["anomalous_flag"] = (result_df["anomaly_score"] >= threshold).astype(int)

    return result_df, threshold


def plot_results(driving_df: pd.DataFrame, threshold: float) -> None:
    N = len(driving_df)
    start_idx = 0
    end_idx = min(5000, N)

    t = driving_df["Time"].iloc[start_idx:end_idx].values

    plt.figure(figsize=(14, 8))

    ax1 = plt.subplot(3, 1, 1)
    plt.plot(t, driving_df["anomaly_score"].iloc[start_idx:end_idx], label="Anomaly score")
    plt.axhline(threshold, color="red", linestyle="--", label="Threshold")
    plt.ylabel("Score")
    plt.legend(loc="upper right")
    plt.plot(
        t,
        driving_df["anomalous_flag"].iloc[start_idx:end_idx],
        label="Anomalous flag",
    )
    plt.plot(
        t,
        driving_df["any_bms_fault"].iloc[start_idx:end_idx],
        label="BMS fault flag",
    )
    plt.ylabel("Flags")
    plt.legend(loc="upper right")

    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    if "Pack Voltage" in driving_df.columns:
        plt.plot(
            t,
            driving_df["Pack Voltage"].iloc[start_idx:end_idx],
            label="Pack Voltage",
        )
    if "Pack Current" in driving_df.columns:
        plt.plot(
            t,
            driving_df["Pack Current"].iloc[start_idx:end_idx],
            label="Pack Current",
        )
    plt.legend(loc="upper right")
    plt.xlabel("Time")
    plt.ylabel("Pack signals")

    plt.tight_layout()
    plt.show()

# Plots all features in df over time
def plot_features_over_time_plotly(df: pd.DataFrame) -> None:

    if "Time" not in df.columns:
        raise KeyError("Expected column 'Time' not found in DataFrame.")

    time_values = df["Time"].values
    feature_cols = [
        col
        for col in df.select_dtypes(include=["number"]).columns
        if col != "Time"
    ]

    if not feature_cols:
        return

    feature_cols.sort()
    n_features = len(feature_cols)

    if n_features == 1:
        fig = go.Figure()
        col = feature_cols[0]
        fig.add_trace(go.Scatter(x=time_values, y=df[col].values, name=col, mode="lines"))
        fig.update_layout(
            title=f"{col} over Time",
            xaxis_title="Time",
            yaxis_title="Value",
        )
        fig.show()
        return

    max_spacing = 1.0 / (n_features - 0.5)
    vertical_spacing = min(0.01, max_spacing - 1e-4) if n_features > 1 else 0.01

    fig = make_subplots(
        rows=n_features,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=vertical_spacing,
        subplot_titles=feature_cols,
    )

    for i, col in enumerate(feature_cols, start=1):
        fig.add_trace(
            go.Scatter(x=time_values, y=df[col].values, name=col, mode="lines"),
            row=i,
            col=1,
        )

    fig.update_xaxes(title_text="Time", row=n_features, col=1)
    fig.update_layout(
        # Slightly taller per subplot so titles remain readable,
        # but still capped to keep everything roughly in one view.
        height=min(150 * n_features, 1200),
        showlegend=False,
        title="Features over Time",
        margin=dict(l=60, r=20, t=80, b=40),
    )

    fig.show()

# Prints evaluation of our model
def print_evaluation(driving_df: pd.DataFrame) -> None:
    y_true = driving_df["any_bms_fault"].values
    y_pred = driving_df["anomalous_flag"].values

    total_fault = (y_true == 1).sum()
    total_normal = (y_true == 0).sum()

    fault_detected = ((y_true == 1) & (y_pred == 1)).sum()
    normal_flagged = ((y_true == 0) & (y_pred == 1)).sum()

    print("Total BMS-fault points:", total_fault)
    print("Fault points flagged anomalous:", fault_detected)
    print(
        "Coverage of faults: {:.2%}".format(
            fault_detected / total_fault if total_fault > 0 else 0
        )
    )

    print("\nTotal normal points:", total_normal)
    print("Normal points flagged anomalous:", normal_flagged)
    print(
        "False positive rate on normal: {:.2%}".format(
            normal_flagged / total_normal if total_normal > 0 else 0
        )
    )


if __name__ == "__main__":
    driving_df = load_driving_data(CSV_PATH)
    plot_features_over_time_plotly(driving_df)
    driving_df, threshold = run_isolation_forest(driving_df)
    plot_results(driving_df, threshold)
    print_evaluation(driving_df)
