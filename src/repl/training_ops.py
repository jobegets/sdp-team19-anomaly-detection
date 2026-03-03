from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.isolation_forest.isolation_forest import run_isolation_forest as run_iforest
from src.models.isolation_forest.isolation_forest_timeseries import (
    run_isolation_forest as run_iforest_timeseries,
)
from src.repl.dataset_ops import build_auto_split, format_repo_relative, refresh_dataset_summaries
from src.repl.prompts import prompt_optional_index
from src.repl.types import AppConfig, SessionState, TrainingRun
from src.utils import early_detection_stats, load_driving_data, plot_results, print_evaluation


def choose_model_name(config: AppConfig) -> str | None:
    """Prompt for model choice by number."""
    print("\nModels:")
    for idx, (_, label) in enumerate(config.model_options, start=1):
        print(f"{idx:>2}. {label}")

    selected_index = prompt_optional_index(
        len(config.model_options), "Choose model number (Enter to cancel): "
    )
    if selected_index is None:
        print("Model selection canceled.")
        return None
    return config.model_options[selected_index - 1][0]


def save_training_artifact(
    config: AppConfig,
    *,
    model_name: str,
    dataset_reference: str,
    threshold: float,
    scaler: object,
    model: object,
    feature_cols: list[str],
    contamination: float,
    threshold_quantile: float,
    window_size: int | None,
) -> None:
    """Persist training outputs. This is always called after training."""
    payload: dict[str, object] = {
        "model_name": model_name,
        "dataset_path": dataset_reference,
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "threshold": threshold,
        "contamination": contamination,
        "threshold_quantile": threshold_quantile,
    }
    if window_size is not None:
        payload["window_size"] = window_size

    config.artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, config.artifact_path)
    print(f"Artifact updated: {config.artifact_path}")


def concatenate_driving_data(csv_paths: list[Path]) -> pd.DataFrame:
    """Load and concatenate multiple driving datasets."""
    frames = [load_driving_data(path) for path in csv_paths]
    if not frames:
        raise ValueError("No datasets were provided.")
    return pd.concat(frames, ignore_index=True)


def score_isolation_forest(
    driving_df: pd.DataFrame,
    *,
    threshold: float,
    scaler: object,
    model: object,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Score a dataframe with an already-trained IF model."""
    required_cols = ["Time", "BMS Disch Enable"]
    for col in required_cols:
        if col not in driving_df.columns:
            raise KeyError(f"Expected column '{col}' not found in driving data.")

    X = driving_df[feature_cols].values
    X_scaled = scaler.transform(X)
    anomaly_scores = -model.decision_function(X_scaled)

    result_df = driving_df.copy()
    result_df["anomaly_score"] = anomaly_scores
    result_df["any_bms_fault"] = (driving_df["BMS Disch Enable"] == 0).astype(int).values
    result_df["anomalous_flag"] = (result_df["anomaly_score"] >= threshold).astype(int)
    return result_df


def score_isolation_forest_timeseries(
    driving_df: pd.DataFrame,
    *,
    threshold: float,
    scaler: object,
    model: object,
    feature_cols: list[str],
    window_size: int,
) -> pd.DataFrame:
    """Score a dataframe with an already-trained timeseries IF model."""
    X_windows, fault_flags, result_df = _build_timeseries_windows(
        driving_df, feature_cols, window_size
    )
    scores = -model.decision_function(scaler.transform(X_windows))
    result_df["anomaly_score"] = scores
    result_df["any_bms_fault"] = fault_flags
    result_df["anomalous_flag"] = (result_df["anomaly_score"] >= threshold).astype(int)
    return result_df


def _build_timeseries_windows(
    driving_df: pd.DataFrame, feature_cols: list[str], window_size: int
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build rolling-mean windows and aligned labels for timeseries IF."""
    if len(driving_df) < window_size:
        raise ValueError(
            f"Not enough samples ({len(driving_df)}) for window size {window_size}."
        )

    rolling_means = (
        driving_df[feature_cols]
        .rolling(window=window_size, min_periods=window_size)
        .mean()
        .iloc[window_size - 1 :]
    )
    fault_flags = (driving_df["BMS Disch Enable"] == 0).astype(int).values[window_size - 1 :]
    times = driving_df["Time"].values[window_size - 1 :]

    result_df = driving_df.iloc[window_size - 1 :].copy()
    result_df["Time"] = times
    return rolling_means.values, fault_flags, result_df


def run_auto_split_timeseries(
    *,
    train_paths: list[Path],
    test_paths: list[Path],
    threshold_quantile: float,
    contamination: float,
    window_size: int,
    random_state: int = 42,
) -> tuple[pd.DataFrame, float, object, object, list[str]]:
    """Train timeseries IF on train paths and evaluate on test paths."""
    train_frames = [load_driving_data(path) for path in train_paths]
    test_frames = [load_driving_data(path) for path in test_paths]

    if not train_frames or not test_frames:
        raise ValueError("Train/test frames could not be loaded.")

    feature_cols = [
        col
        for col in train_frames[0].select_dtypes(include=["number"]).columns
        if col not in ("Time", "BMS Disch Enable")
    ]
    if not feature_cols:
        raise ValueError("No numeric feature columns found for sliding window.")

    X_train_full_list: list[np.ndarray] = []
    X_train_normal_list: list[np.ndarray] = []

    for frame in train_frames:
        X_full, fault_flags, _ = _build_timeseries_windows(frame, feature_cols, window_size)
        normal_mask = fault_flags == 0
        if not normal_mask.any():
            continue
        X_train_full_list.append(X_full)
        X_train_normal_list.append(X_full[normal_mask])

    if not X_train_normal_list:
        raise ValueError("No normal training windows found in auto split train set.")

    X_train_full = np.vstack(X_train_full_list)
    X_train_normal = np.vstack(X_train_normal_list)

    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    X_train_normal_scaled = scaler.fit_transform(X_train_normal)
    X_train_full_scaled = scaler.transform(X_train_full)

    iso = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    iso.fit(X_train_normal_scaled)

    train_scores = -iso.decision_function(X_train_full_scaled)
    threshold = float(np.quantile(train_scores, threshold_quantile))

    test_results: list[pd.DataFrame] = []
    for frame in test_frames:
        X_test, fault_flags_test, test_df = _build_timeseries_windows(
            frame, feature_cols, window_size
        )
        scores = -iso.decision_function(scaler.transform(X_test))
        test_df["anomaly_score"] = scores
        test_df["any_bms_fault"] = fault_flags_test
        test_df["anomalous_flag"] = (test_df["anomaly_score"] >= threshold).astype(int)
        test_results.append(test_df)

    if not test_results:
        raise ValueError("No test windows available after applying window size.")

    return pd.concat(test_results, ignore_index=True), threshold, scaler, iso, feature_cols


def train_on_selected_dataset(config: AppConfig, state: SessionState) -> None:
    """Train user-selected model on the selected dataset and store session run."""
    if state.selected_training_dataset is None:
        print("No training dataset selected. Choose option 1 first.")
        return

    model_name = choose_model_name(config)
    if model_name is None:
        return

    dataset_label = format_repo_relative(config, state.selected_training_dataset)
    print(f"\nTraining {model_name} on {dataset_label}...")
    driving_df = load_driving_data(state.selected_training_dataset)

    if model_name == "isolation-forest":
        _, threshold, scaler, model, feature_cols = run_iforest(
            driving_df=driving_df,
            contamination=config.default_contamination,
            threshold_quantile=config.default_threshold_quantile,
        )
        window_size: int | None = None
    else:
        _, threshold, scaler, model, feature_cols = run_iforest_timeseries(
            driving_df=driving_df,
            contamination=config.default_contamination,
            threshold_quantile=config.default_threshold_quantile,
            window_size=config.default_window_size,
        )
        window_size = config.default_window_size

    save_training_artifact(
        config,
        model_name=model_name,
        dataset_reference=dataset_label,
        threshold=threshold,
        scaler=scaler,
        model=model,
        feature_cols=feature_cols,
        contamination=config.default_contamination,
        threshold_quantile=config.default_threshold_quantile,
        window_size=window_size,
    )

    state.trained_runs.append(
        TrainingRun(
            run_id=len(state.trained_runs) + 1,
            model_name=model_name,
            training_dataset_label=dataset_label,
            evaluation_dataset_label=None,
            threshold=threshold,
            contamination=config.default_contamination,
            threshold_quantile=config.default_threshold_quantile,
            window_size=window_size,
            scaler=scaler,
            model=model,
            feature_cols=feature_cols,
            result_df=None,
        )
    )
    print(f"Training complete. Session run #{state.trained_runs[-1].run_id} stored.")


def run_auto_split_training(config: AppConfig, state: SessionState) -> None:
    """Train on NO_FAULTS datasets and evaluate on HAS_FAULTS datasets."""
    refresh_dataset_summaries(config, state)
    usable = [s for s in state.dataset_summaries if s.driving_rows > 0]

    no_fault_count = sum(1 for s in usable if s.label == "no_faults")
    has_fault_count = sum(1 for s in usable if s.label == "has_faults")
    if no_fault_count == 0 or has_fault_count == 0:
        print(
            "Auto split requires at least one NO_FAULTS dataset and one HAS_FAULTS dataset."
        )
        return

    train_paths, test_paths = build_auto_split(state.dataset_summaries)
    print("\nAuto split plan:")
    print(f"Train files (NO_FAULTS): {len(train_paths)}")
    print(f"Test files (HAS_FAULTS): {len(test_paths)}")

    model_name = choose_model_name(config)
    if model_name is None:
        return

    if model_name == "isolation-forest":
        train_df = concatenate_driving_data(train_paths)
        test_df = concatenate_driving_data(test_paths)
        _, threshold, scaler, model, feature_cols = run_iforest(
            driving_df=train_df,
            contamination=config.default_contamination,
            threshold_quantile=config.default_threshold_quantile,
        )
        result_df = score_isolation_forest(
            test_df,
            threshold=threshold,
            scaler=scaler,
            model=model,
            feature_cols=feature_cols,
        )
        window_size: int | None = None
    else:
        result_df, threshold, scaler, model, feature_cols = run_auto_split_timeseries(
            train_paths=train_paths,
            test_paths=test_paths,
            threshold_quantile=config.default_threshold_quantile,
            contamination=config.default_contamination,
            window_size=config.default_window_size,
        )
        window_size = config.default_window_size

    split_label = (
        f"auto_split(train={len(train_paths)} NO_FAULTS, "
        f"test={len(test_paths)} HAS_FAULTS)"
    )

    save_training_artifact(
        config,
        model_name=model_name,
        dataset_reference=split_label,
        threshold=threshold,
        scaler=scaler,
        model=model,
        feature_cols=feature_cols,
        contamination=config.default_contamination,
        threshold_quantile=config.default_threshold_quantile,
        window_size=window_size,
    )

    state.trained_runs.append(
        TrainingRun(
            run_id=len(state.trained_runs) + 1,
            model_name=model_name,
            training_dataset_label=split_label,
            evaluation_dataset_label=split_label,
            threshold=threshold,
            contamination=config.default_contamination,
            threshold_quantile=config.default_threshold_quantile,
            window_size=window_size,
            scaler=scaler,
            model=model,
            feature_cols=feature_cols,
            result_df=result_df,
        )
    )

    print(f"Auto split training complete. Session run #{state.trained_runs[-1].run_id} stored.")
    print_evaluation(result_df, lookback_seconds=config.default_lookback_seconds)


def evaluate_last_trained_model(config: AppConfig, state: SessionState) -> None:
    """Print evaluation for the most recent trained model in this session."""
    if not state.trained_runs:
        print("No trained model in this session. Train first with option 3.")
        return

    if state.selected_evaluation_dataset is None:
        print("No evaluation dataset selected. Choose option 2 first.")
        return

    last_run = state.trained_runs[-1]
    eval_label = format_repo_relative(config, state.selected_evaluation_dataset)
    eval_df = load_driving_data(state.selected_evaluation_dataset)

    if last_run.model_name == "isolation-forest":
        result_df = score_isolation_forest(
            eval_df,
            threshold=last_run.threshold,
            scaler=last_run.scaler,
            model=last_run.model,
            feature_cols=last_run.feature_cols,
        )
    else:
        if last_run.window_size is None:
            raise ValueError("Timeseries run is missing window_size.")
        result_df = score_isolation_forest_timeseries(
            eval_df,
            threshold=last_run.threshold,
            scaler=last_run.scaler,
            model=last_run.model,
            feature_cols=last_run.feature_cols,
            window_size=last_run.window_size,
        )

    last_run.result_df = result_df
    last_run.evaluation_dataset_label = eval_label

    print(
        f"\nEvaluating run #{last_run.run_id} "
        f"({last_run.model_name}) on {eval_label}"
    )
    print_evaluation(result_df, lookback_seconds=config.default_lookback_seconds)


def plot_last_results(state: SessionState) -> None:
    """Plot the most recent evaluated run in this session."""
    if not state.trained_runs:
        print("No trained model in this session. Train first with option 3.")
        return

    last_run = state.trained_runs[-1]
    if last_run.result_df is None:
        print("No evaluation results available for the last run. Evaluate first with option 4.")
        return

    print(
        f"\nPlotting run #{last_run.run_id} "
        f"({last_run.model_name}) "
        f"on {last_run.evaluation_dataset_label or 'UNKNOWN_DATASET'}"
    )
    plot_results(last_run.result_df, threshold=last_run.threshold)


def summarize_run_metrics(config: AppConfig, run: TrainingRun) -> dict[str, float | int]:
    """Compute compact metrics used by compare output."""
    if run.result_df is None:
        return {
            "coverage": float("nan"),
            "fp_rate": float("nan"),
            "early_hits": -1,
            "mean_lead": float("nan"),
        }

    y_true = run.result_df["any_bms_fault"].values
    y_pred = run.result_df["anomalous_flag"].values

    total_fault = int((y_true == 1).sum())
    total_normal = int((y_true == 0).sum())
    fault_detected = int(((y_true == 1) & (y_pred == 1)).sum())
    normal_flagged = int(((y_true == 0) & (y_pred == 1)).sum())
    coverage = fault_detected / total_fault if total_fault else 0.0
    fp_rate = normal_flagged / total_normal if total_normal else 0.0
    _, early_hits, mean_lead = early_detection_stats(
        run.result_df, lookback_seconds=config.default_lookback_seconds
    )

    return {
        "coverage": coverage,
        "fp_rate": fp_rate,
        "early_hits": early_hits,
        "mean_lead": mean_lead,
    }


def compare_session_runs(config: AppConfig, state: SessionState) -> None:
    """Print metrics summary for all models trained in the current session."""
    if not state.trained_runs:
        print("No trained models to compare in this session.")
        return

    print("\nSession comparison:")
    print(
        "Run | Model                          | Train Dataset                        | Eval Dataset                         "
        "| Fault Coverage | FP Rate | Early Hits | Mean Lead (s)"
    )
    print("-" * 170)

    for run in state.trained_runs:
        metrics = summarize_run_metrics(config, run)
        train_name = run.training_dataset_label
        eval_name = run.evaluation_dataset_label or "NOT_EVALUATED"
        if run.result_df is None:
            coverage_str = "N/A"
            fp_rate_str = "N/A"
            early_hits_str = "N/A"
            mean_lead_str = "N/A"
        else:
            coverage_str = f"{metrics['coverage'] * 100:>13.2f}%"
            fp_rate_str = f"{metrics['fp_rate'] * 100:>7.2f}%"
            early_hits_str = f"{int(metrics['early_hits']):>10d}"
            mean_lead_str = f"{float(metrics['mean_lead']):>13.3f}"

        print(
            f"{run.run_id:>3} | "
            f"{run.model_name:<30} | "
            f"{train_name[:36]:<36} | "
            f"{eval_name[:36]:<36} | "
            f"{coverage_str:>13} | "
            f"{fp_rate_str:>7} | "
            f"{early_hits_str:>10} | "
            f"{mean_lead_str:>13}"
        )
