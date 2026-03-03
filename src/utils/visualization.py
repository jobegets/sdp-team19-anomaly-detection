import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_results(
    driving_df: pd.DataFrame,
    threshold: float,
    # window_seconds: float = 1.0,
    # feature_cols: list[str] | None = None,
) -> None:
    """Visualize anomaly scores, flags, and key pack signals.

    Produces a multi-row Plotly figure with anomaly scores (and threshold), predicted flags,
    and selected pack signals. If model-only anomalies exist (flagged but no BMS fault),
    the first one is expanded via `plot_new_anomaly_windows`.

    Args:
        driving_df: Dataframe containing time-series data with anomaly_score, anomalous_flag, any_bms_fault.
        threshold: Anomaly score cutoff used for flagging.
        window_seconds: Time span window to show around the first model-only anomaly.
        feature_cols: Optional explicit list of numeric feature columns for window plots; defaults to all numeric.
    """
    t = driving_df["Time"]
    model_only = driving_df[
        (driving_df["anomalous_flag"] == 1) & (driving_df["any_bms_fault"] == 0)
    ]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=["Anomaly score", "Flags", "Pack signals"],
    )

    fig.add_trace(
        go.Scatter(
            x=t,
            y=driving_df["anomaly_score"],
            name="Anomaly score",
            mode="lines",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="red",
        row=1,
        col=1,
        annotation_text="Threshold",
    )

    fig.add_trace(
        go.Scatter(
            x=t,
            y=driving_df["anomalous_flag"],
            name="Anomalous flag",
            mode="lines",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=driving_df["any_bms_fault"],
            name="BMS fault flag",
            mode="lines",
        ),
        row=2,
        col=1,
    )

    if "Pack Voltage" in driving_df.columns:
        fig.add_trace(
            go.Scatter(
                x=t,
                y=driving_df["Pack Voltage"],
                name="Pack Voltage",
                mode="lines",
            ),
            row=3,
            col=1,
        )
    if "Pack Current" in driving_df.columns:
        fig.add_trace(
            go.Scatter(
                x=t,
                y=driving_df["Pack Current"],
                name="Pack Current",
                mode="lines",
            ),
            row=3,
            col=1,
        )
    
    if not model_only.empty:
        fig.add_trace(
            go.Scatter(
                x=model_only["Time"].astype(float),
                y=model_only["anomaly_score"],
                mode="markers",
                marker=dict(color="orange", size=6, symbol="diamond"),
                name="Model-only anomaly",
            ),
            row=1,
            col=1,
        )

    fig.update_layout(
        height=900,
        title="Anomaly Detection Results",
        showlegend=True,
        margin=dict(l=60, r=20, t=60, b=40),
    )
    fig.update_yaxes(title_text="Score", row=1, col=1)
    fig.update_yaxes(title_text="Flags", row=2, col=1)
    fig.update_yaxes(title_text="Pack signals", row=3, col=1)
    fig.update_xaxes(title_text="Time", row=3, col=1)

    fig.show()

    # if not model_only.empty:
    #     plot_new_anomaly_windows(
    #         driving_df,
    #         anomaly_times=model_only["Time"].astype(float).tolist(),
    #         window_seconds=window_seconds,
    #         feature_cols=feature_cols,
    #     )

# TODO: Keeping as an experimental feature.
# def plot_features_over_time(df: pd.DataFrame) -> None:
#     """Plot every numeric feature over time as stacked Plotly subplots.

#     Args:
#         df: DataFrame containing a Time column and numeric feature columns to plot.

#     Returns:
#         None. Displays an interactive Plotly figure.
#     """
#     if "Time" not in df.columns:
#         raise KeyError("Expected column 'Time' not found in DataFrame.")

#     time_values = df["Time"].values
#     feature_cols = [
#         col for col in df.select_dtypes(include=["number"]).columns if col != "Time"
#     ]

#     if not feature_cols:
#         return

#     feature_cols.sort()
#     n_features = len(feature_cols)

#     max_spacing = 1.0 / (n_features - 0.5)
#     vertical_spacing = min(0.01, max_spacing - 1e-4) if n_features > 1 else 0.01

#     fig = make_subplots(
#         rows=n_features,
#         cols=1,
#         shared_xaxes=True,
#         vertical_spacing=vertical_spacing,
#         subplot_titles=feature_cols,
#     )

#     for i, col in enumerate(feature_cols, start=1):
#         fig.add_trace(
#             go.Scatter(x=time_values, y=df[col].values, name=col, mode="lines"),
#             row=i,
#             col=1,
#         )

#     fig.update_xaxes(title_text="Time", row=n_features, col=1)
#     fig.update_layout(
#         height=min(150 * n_features, 1200),
#         showlegend=False,
#         title="Features over Time",
#         margin=dict(l=60, r=20, t=80, b=40),
#     )

#     fig.show()

# TODO: Keeping as an experimental feature.
#
# def plot_new_anomaly_windows(
#     driving_df: pd.DataFrame,
#     anomaly_times: list[float],
#     window_seconds: float = 1.0,
#     feature_cols: list[str] | None = None,
# ) -> None:
#     """Expand the first model-only anomaly into per-feature windows. 
    
#     This is a proof of concept that models what the FSAE team would see for each fault found.

#     Args:
#         driving_df: Full driving dataframe with Time and numeric feature columns.
#         anomaly_times: List of timestamps (seconds) where the model flagged anomalies without BMS faults.
#         window_seconds: Window size in seconds to display before and after an anomaly.
#         feature_cols: Optional subset of numeric columns to plot; defaults to all numeric columns except Time.

#     Returns:
#         None. Displays an interactive Plotly figure with one column per anomaly and one row per feature.
#     """
#     if "Time" not in driving_df.columns:
#         raise KeyError("Expected column 'Time' not found in DataFrame.")

#     if feature_cols is None:
#         feature_cols = [
#             col
#             for col in driving_df.select_dtypes(include=["number"]).columns
#             if col != "Time"
#         ]

#     if not feature_cols:
#         print("No numeric feature columns available to plot.")
#         return

#     if not anomaly_times:
#         print("No new anomalies to visualize.")
#         return

#     times = driving_df["Time"].astype(float)
#     event_times = sorted(anomaly_times)[:1]  # only the first anomaly

#     fig = make_subplots(
#         rows=len(feature_cols),
#         cols=len(event_times),
#         shared_xaxes=False,
#         shared_yaxes=False,
#         vertical_spacing=0.02,
#         horizontal_spacing=0.04,
#         column_titles=[f"t={t:.3f}s" for t in event_times],
#     )

#     for c, event_time in enumerate(event_times, start=1):
#         mask = (times >= event_time - window_seconds) & (times <= event_time + window_seconds)
#         window_df = driving_df.loc[mask, ["Time"] + feature_cols]

#         if window_df.empty:
#             continue

#         for r, col in enumerate(feature_cols, start=1):
#             fig.add_trace(
#                 go.Scatter(
#                     x=window_df["Time"],
#                     y=window_df[col],
#                     name=col if c == 1 else None,  # legend once
#                     mode="lines",
#                     showlegend=(c == 1),
#                 ),
#                 row=r,
#                 col=c,
#             )
#             fig.add_vline(
#                 x=event_time,
#                 line_width=1,
#                 line_dash="dot",
#                 line_color="orange",
#                 row=r,
#                 col=c,
#             )
#             if c == 1:
#                 fig.update_yaxes(title_text=col, row=r, col=c)

#         fig.update_xaxes(title_text="Time", row=len(feature_cols), col=c)

#     fig.update_layout(
#         height=5000,
#         title=f"Model-only anomalies (±{window_seconds:.2f}s)",
#         showlegend=True,
#         margin=dict(l=60, r=20, t=80, b=40),
#     )
#     fig.show()


def _fault_start_indices(labels: np.ndarray) -> list[int]:
    """Private helper function that computes rising-edge indices for fault labels.

    Args:
        labels: Array of binary labels over time.

    Returns:
        List of indices where labels transition from 0 to 1 (fault onset).
    """
    starts = []
    for i, val in enumerate(labels):
        if val == 1 and (i == 0 or labels[i - 1] == 0):
            starts.append(i)
    return starts


def early_detection_stats(
    driving_df: pd.DataFrame, lookback_seconds: float = 1.0
) -> tuple[int, int, float]:
    """Count faults detected before their flag and compute average lead time (seconds).

    Args:
        driving_df: DataFrame containing Time, any_bms_fault, and anomalous_flag columns.
        lookback_seconds: Time window before each fault start to credit early detections.

    Returns:
        A tuple of (num_fault_segments, num_detected_early, mean_lead_time_seconds).
    """
    times = driving_df["Time"].astype(float).values
    y_true = driving_df["any_bms_fault"].values
    y_pred = driving_df["anomalous_flag"].values

    starts = _fault_start_indices(y_true)
    if not starts:
        return 0, 0, 0.0

    detected_before = 0
    lead_times = []

    for start_idx in starts:
        t_start = times[start_idx]
        window_mask = (times >= t_start - lookback_seconds) & (times <= t_start)
        candidate_idxs = np.where(window_mask & (y_pred == 1))[0]
        if candidate_idxs.size:
            earliest = candidate_idxs[0]
            lead = t_start - times[earliest]
            detected_before += 1
            lead_times.append(lead)

    mean_lead = float(np.mean(lead_times)) if lead_times else 0.0
    return len(starts), detected_before, mean_lead


def print_evaluation(driving_df: pd.DataFrame, lookback_seconds: float = 1.0) -> None:
    """Print high-level evaluation metrics for anomaly flags vs. BMS faults.

    Includes recall on faults, false positives on normal operation, a confusion matrix,
    model-only anomaly count, and early detection stats within the lookback window.

    Args:
        driving_df: DataFrame with any_bms_fault and anomalous_flag columns.
        lookback_seconds: Time window before each fault start to count early detections.

    Returns:
        None. Metrics are logged.
    """
    # Binary labels (ground truth) 
    y_true = driving_df["any_bms_fault"].values
    # and predictions from the model
    y_pred = driving_df["anomalous_flag"].values

    # Basic counts for faulty and normal operational data
    total_fault = int((y_true == 1).sum())
    total_normal = int((y_true == 0).sum())

    # Confusion matrix components
    fault_detected = int(((y_true == 1) & (y_pred == 1)).sum())
    normal_flagged = int(((y_true == 0) & (y_pred == 1)).sum())
    fault_missed = int(((y_true == 1) & (y_pred == 0)).sum())
    normal_correct = int(((y_true == 0) & (y_pred == 0)).sum())

    # Recall on faults: are we catching the real faults?
    print("Total BMS-fault points:", total_fault)
    print("Fault points flagged anomalous:", fault_detected)
    print(
        "Coverage of faults: {:.2%}".format(
            fault_detected / total_fault if total_fault > 0 else 0
        )
    )

    # False positives: how noisy are we on normal data?
    print("\nTotal normal points:", total_normal)
    print("Normal points flagged anomalous:", normal_flagged)
    print(
        "False positive rate on normal: {:.2%}".format(
            normal_flagged / total_normal if total_normal > 0 else 0
        )
    )

    # Quick confusion matrix for a full view of classification behavior
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(f"           Pred 0    Pred 1")
    print(f"True 0 | {normal_correct:7d} {normal_flagged:8d}")
    print(f"True 1 | {fault_missed:7d} {fault_detected:8d}")

    # Model-only anomalies = flags with no corresponding fault flag
    print("\nModel-only anomalies (flagged with no fault flag):", normal_flagged)

    # Early detection: count how often we flag before the fault flag starts, and by how much
    total_segments, early_hits, mean_lead = early_detection_stats(
        driving_df, lookback_seconds=lookback_seconds
    )
    if total_segments:
        print(
            "\nEarly detections (<= {:.2f}s before fault flag): {} / {} ({:.2%})".format(
                lookback_seconds, early_hits, total_segments, early_hits / total_segments
            )
        )
        if early_hits:
            print("Mean early lead time: {:.3f} s".format(mean_lead))
    else:
        print("\nEarly detection: no fault segments to evaluate.")
