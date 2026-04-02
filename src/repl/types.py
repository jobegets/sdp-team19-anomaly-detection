from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.utils.data import DatasetSummary


@dataclass(frozen=True)
class AppConfig:
    repo_root: Path
    dataset_root: Path
    default_contamination: float
    default_threshold_quantile: float
    default_window_size: int
    default_lookback_seconds: float
    model_options: tuple[tuple[str, str], ...]


@dataclass
class TrainingRun:
    run_id: int
    model_name: str
    training_dataset_label: str
    evaluation_dataset_label: str | None
    threshold: float
    contamination: float
    threshold_quantile: float
    window_size: int | None
    scaler: object
    model: object
    feature_cols: list[str]
    result_df: pd.DataFrame | None = None


@dataclass
class SessionState:
    selected_training_dataset: Path | None = None
    selected_evaluation_dataset: Path | None = None
    dataset_summaries: list[DatasetSummary] = field(default_factory=list)
    trained_runs: list[TrainingRun] = field(default_factory=list)
