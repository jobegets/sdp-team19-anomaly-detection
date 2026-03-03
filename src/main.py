from dataclasses import dataclass, field
from pathlib import Path
import sys

import joblib
import pandas as pd

if __package__ is None or __package__ == "":
    # Allow running as a script from repo root: python src/main.py
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.isolation_forest.isolation_forest import run_isolation_forest as run_iforest
from src.models.isolation_forest.isolation_forest_timeseries import (
    run_isolation_forest as run_iforest_timeseries,
)
from src.utils import early_detection_stats, load_driving_data, print_evaluation

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = (REPO_ROOT / "test-data").resolve()
ARTIFACT_PATH = (REPO_ROOT / "artifacts" / "isolation_forest.pkl").resolve()

DEFAULT_CONTAMINATION = 0.01
DEFAULT_THRESHOLD_QUANTILE = 0.80
DEFAULT_WINDOW_SIZE = 25
DEFAULT_LOOKBACK_SECONDS = 15.0

MODEL_OPTIONS: list[tuple[str, str]] = [
    ("isolation-forest", "Isolation Forest"),
    ("isolation-forest-timeseries", "Isolation Forest (timeseries rolling mean)"),
]


@dataclass
class TrainingRun:
    run_id: int
    model_name: str
    dataset_path: Path
    threshold: float
    contamination: float
    threshold_quantile: float
    window_size: int | None
    result_df: pd.DataFrame


@dataclass
class SessionState:
    selected_dataset: Path | None = None
    trained_runs: list[TrainingRun] = field(default_factory=list)


def list_available_datasets() -> list[Path]:
    """Return all CSV files under test-data, sorted by relative path."""
    if not DATASET_ROOT.exists():
        return []
    return sorted(path.resolve() for path in DATASET_ROOT.rglob("*.csv"))


def format_repo_relative(path: Path) -> str:
    """Format path relative to repo root when possible."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def prompt_menu_choice(prompt: str, valid_choices: set[str]) -> str:
    """Prompt for a menu choice until valid input is given."""
    while True:
        choice = input(prompt).strip()
        if choice in valid_choices:
            return choice
        print(f"Invalid choice. Choose one of: {', '.join(sorted(valid_choices))}")


def prompt_optional_index(total: int, prompt: str) -> int | None:
    """Prompt for a 1-based index, or blank to cancel."""
    while True:
        raw = input(prompt).strip()
        if raw == "":
            return None
        if not raw.isdigit():
            print("Please enter a number or press Enter to cancel.")
            continue
        value = int(raw)
        if 1 <= value <= total:
            return value
        print(f"Please choose a value between 1 and {total}.")


def choose_dataset(state: SessionState) -> None:
    """Show dataset options and update the session-selected dataset."""
    datasets = list_available_datasets()
    if not datasets:
        print(f"No datasets found under {DATASET_ROOT}")
        return

    print("\nDatasets:")
    for idx, path in enumerate(datasets, start=1):
        marker = " (selected)" if state.selected_dataset == path else ""
        print(f"{idx:>2}. {format_repo_relative(path)}{marker}")

    selected_index = prompt_optional_index(
        len(datasets), "Choose dataset number (Enter to cancel): "
    )
    if selected_index is None:
        print("Dataset selection canceled.")
        return

    state.selected_dataset = datasets[selected_index - 1]
    print(f"Selected dataset: {format_repo_relative(state.selected_dataset)}")


def choose_model_name() -> str | None:
    """Prompt for model choice by number."""
    print("\nModels:")
    for idx, (_, label) in enumerate(MODEL_OPTIONS, start=1):
        print(f"{idx:>2}. {label}")

    selected_index = prompt_optional_index(
        len(MODEL_OPTIONS), "Choose model number (Enter to cancel): "
    )
    if selected_index is None:
        print("Model selection canceled.")
        return None
    return MODEL_OPTIONS[selected_index - 1][0]


def save_training_artifact(
    *,
    model_name: str,
    dataset_path: Path,
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
        "dataset_path": str(dataset_path),
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "threshold": threshold,
        "contamination": contamination,
        "threshold_quantile": threshold_quantile,
    }
    if window_size is not None:
        payload["window_size"] = window_size

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, ARTIFACT_PATH)
    print(f"Artifact updated: {ARTIFACT_PATH}")


def train_on_selected_dataset(state: SessionState) -> None:
    """Train user-selected model on the selected dataset and store session run."""
    if state.selected_dataset is None:
        print("No dataset selected. Choose option 1 first.")
        return

    model_name = choose_model_name()
    if model_name is None:
        return

    print(
        f"\nTraining {model_name} on {format_repo_relative(state.selected_dataset)}..."
    )

    driving_df = load_driving_data(state.selected_dataset)

    if model_name == "isolation-forest":
        result_df, threshold, scaler, model, feature_cols = run_iforest(
            driving_df=driving_df,
            contamination=DEFAULT_CONTAMINATION,
            threshold_quantile=DEFAULT_THRESHOLD_QUANTILE,
        )
        window_size: int | None = None
    else:
        result_df, threshold, scaler, model, feature_cols = run_iforest_timeseries(
            driving_df=driving_df,
            contamination=DEFAULT_CONTAMINATION,
            threshold_quantile=DEFAULT_THRESHOLD_QUANTILE,
            window_size=DEFAULT_WINDOW_SIZE,
        )
        window_size = DEFAULT_WINDOW_SIZE

    save_training_artifact(
        model_name=model_name,
        dataset_path=state.selected_dataset,
        threshold=threshold,
        scaler=scaler,
        model=model,
        feature_cols=feature_cols,
        contamination=DEFAULT_CONTAMINATION,
        threshold_quantile=DEFAULT_THRESHOLD_QUANTILE,
        window_size=window_size,
    )

    run = TrainingRun(
        run_id=len(state.trained_runs) + 1,
        model_name=model_name,
        dataset_path=state.selected_dataset,
        threshold=threshold,
        contamination=DEFAULT_CONTAMINATION,
        threshold_quantile=DEFAULT_THRESHOLD_QUANTILE,
        window_size=window_size,
        result_df=result_df,
    )
    state.trained_runs.append(run)

    print(f"Training complete. Session run #{run.run_id} stored.")


def evaluate_last_trained_model(state: SessionState) -> None:
    """Print evaluation for the most recent trained model in this session."""
    if not state.trained_runs:
        print("No trained model in this session. Train first with option 2.")
        return

    last_run = state.trained_runs[-1]
    print(
        f"\nEvaluating run #{last_run.run_id} "
        f"({last_run.model_name}) on {format_repo_relative(last_run.dataset_path)}"
    )
    print_evaluation(last_run.result_df, lookback_seconds=DEFAULT_LOOKBACK_SECONDS)


def summarize_run_metrics(run: TrainingRun) -> dict[str, float | int]:
    """Compute compact metrics used by compare output."""
    y_true = run.result_df["any_bms_fault"].values
    y_pred = run.result_df["anomalous_flag"].values

    total_fault = int((y_true == 1).sum())
    total_normal = int((y_true == 0).sum())
    fault_detected = int(((y_true == 1) & (y_pred == 1)).sum())
    normal_flagged = int(((y_true == 0) & (y_pred == 1)).sum())
    coverage = fault_detected / total_fault if total_fault else 0.0
    fp_rate = normal_flagged / total_normal if total_normal else 0.0
    _, early_hits, mean_lead = early_detection_stats(
        run.result_df, lookback_seconds=DEFAULT_LOOKBACK_SECONDS
    )

    return {
        "coverage": coverage,
        "fp_rate": fp_rate,
        "early_hits": early_hits,
        "mean_lead": mean_lead,
    }


def compare_session_runs(state: SessionState) -> None:
    """Print metrics summary for all models trained in the current session."""
    if not state.trained_runs:
        print("No trained models to compare in this session.")
        return

    print("\nSession comparison:")
    print(
        "Run | Model                          | Dataset                              "
        "| Fault Coverage | FP Rate | Early Hits | Mean Lead (s)"
    )
    print("-" * 120)

    for run in state.trained_runs:
        metrics = summarize_run_metrics(run)
        dataset_name = format_repo_relative(run.dataset_path)
        print(
            f"{run.run_id:>3} | "
            f"{run.model_name:<30} | "
            f"{dataset_name[:36]:<36} | "
            f"{metrics['coverage'] * 100:>13.2f}% | "
            f"{metrics['fp_rate'] * 100:>7.2f}% | "
            f"{int(metrics['early_hits']):>10d} | "
            f"{float(metrics['mean_lead']):>13.3f}"
        )


def print_repl_menu(state: SessionState) -> None:
    """Render REPL menu with session context."""
    selected_dataset = (
        format_repo_relative(state.selected_dataset)
        if state.selected_dataset is not None
        else "None"
    )
    last_run = state.trained_runs[-1] if state.trained_runs else None
    last_run_label = (
        f"#{last_run.run_id} {last_run.model_name} "
        f"({format_repo_relative(last_run.dataset_path)})"
        if last_run
        else "None"
    )

    print("\n=== Anomaly Detection REPL ===")
    print(f"Selected dataset: {selected_dataset}")
    print(f"Last trained model: {last_run_label}")
    print("1. Choose dataset")
    print("2. Train model on chosen dataset (always updates artifact)")
    print("3. Evaluate on last trained model using its dataset")
    print("4. Compare trained models in this session")
    print("5. Exit")


def run_repl() -> int:
    """Start and run the interactive REPL loop."""
    state = SessionState()
    print("Starting anomaly detection REPL.")

    while True:
        print_repl_menu(state)
        choice = prompt_menu_choice("Select option [1-5]: ", {"1", "2", "3", "4", "5"})

        try:
            if choice == "1":
                choose_dataset(state)
            elif choice == "2":
                train_on_selected_dataset(state)
            elif choice == "3":
                evaluate_last_trained_model(state)
            elif choice == "4":
                compare_session_runs(state)
            else:
                print("Exiting REPL.")
                return 0
        except Exception as exc:
            print(f"Error: {exc}")


def main() -> int:
    """Entrypoint."""
    return run_repl()


if __name__ == "__main__":
    raise SystemExit(main())
