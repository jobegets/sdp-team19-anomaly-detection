from pathlib import Path

from src.repl import AppConfig, run_repl

# Keep primary app constants centralized here.
REPO_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = (REPO_ROOT / "test-data").resolve()
DEFAULT_CONTAMINATION = 0.01
DEFAULT_THRESHOLD_QUANTILE = 0.99
DEFAULT_WINDOW_SIZE = 25
DEFAULT_LOOKBACK_SECONDS = 15.0

MODEL_OPTIONS: tuple[tuple[str, str], ...] = (
    ("isolation-forest", "Isolation Forest"),
    ("isolation-forest-timeseries", "Isolation Forest (timeseries rolling mean)"),
    ("autoencoder", "Autoencoder"),
    ("autoencoder-lstm", "Autoencoder (LSTM)"),
)


def build_config() -> AppConfig:
    """Create runtime config."""
    return AppConfig(
        repo_root=REPO_ROOT,
        dataset_root=DATASET_ROOT,
        default_contamination=DEFAULT_CONTAMINATION,
        default_threshold_quantile=DEFAULT_THRESHOLD_QUANTILE,
        default_window_size=DEFAULT_WINDOW_SIZE,
        default_lookback_seconds=DEFAULT_LOOKBACK_SECONDS,
        model_options=MODEL_OPTIONS,
    )


if __name__ == "__main__":
    run_repl(build_config())
