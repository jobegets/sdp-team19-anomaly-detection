from pathlib import Path

from src.repl.prompts import prompt_optional_index
from src.repl.types import AppConfig, SessionState
from src.utils import DatasetSummary, save_dataset_manifest, summarize_dataset


def list_available_datasets(config: AppConfig) -> list[Path]:
    """Return all CSV files under test-data, sorted by relative path."""
    if not config.dataset_root.exists():
        return []
    return sorted(path.resolve() for path in config.dataset_root.rglob("*.csv"))


def format_repo_relative(config: AppConfig, path: Path) -> str:
    """Format path relative to repo root when possible."""
    try:
        return str(path.relative_to(config.repo_root))
    except ValueError:
        return str(path)


def dataset_label_display(summary: DatasetSummary) -> str:
    """Short human-readable label for dataset quality."""
    if summary.label == "has_faults":
        return "HAS_FAULTS"
    if summary.label == "no_faults":
        return "NO_FAULTS"
    return "EMPTY_AFTER_FILTER"


def dataset_sort_key(config: AppConfig, summary: DatasetSummary) -> tuple[int, str]:
    """Sort faults-first for split planning, then path."""
    rank_map = {
        "has_faults": 0,
        "no_faults": 1,
        "empty_after_filter": 2,
    }
    return rank_map[summary.label], format_repo_relative(config, summary.csv_path)


def refresh_dataset_summaries(config: AppConfig, state: SessionState) -> None:
    """Rescan dataset folder and cache per-file summaries in session state."""
    summaries: list[DatasetSummary] = []
    errored: list[tuple[Path, Exception]] = []

    for csv_path in list_available_datasets(config):
        try:
            summaries.append(summarize_dataset(csv_path))
        except Exception as exc:
            errored.append((csv_path, exc))

    summaries.sort(key=lambda summary: dataset_sort_key(config, summary))
    state.dataset_summaries = summaries
    save_dataset_manifest(summaries, config.dataset_manifest_path)

    if errored:
        print(f"Skipped {len(errored)} dataset(s) that could not be summarized:")
        for path, exc in errored:
            print(f"  - {format_repo_relative(config, path)}: {exc}")


def get_summary_for_path(state: SessionState, csv_path: Path) -> DatasetSummary | None:
    """Find cached summary for a path."""
    for summary in state.dataset_summaries:
        if summary.csv_path == csv_path:
            return summary
    return None


def choose_dataset(config: AppConfig, state: SessionState) -> None:
    """Show dataset options and update the session-selected dataset."""
    refresh_dataset_summaries(config, state)
    if not state.dataset_summaries:
        print(f"No datasets found under {config.dataset_root}")
        return

    usable_summaries = [
        summary for summary in state.dataset_summaries if summary.driving_rows > 0
    ]
    ignored_count = len(state.dataset_summaries) - len(usable_summaries)

    if not usable_summaries:
        print(
            "No usable datasets found after removing non-moving rows."
            " Check RPM data quality."
        )
        return

    if ignored_count:
        print(
            f"Ignoring {ignored_count} dataset(s) with zero moving samples "
            f"(RPM <= 0.5 after filtering)."
        )

    print("\nDatasets:")
    for idx, summary in enumerate(usable_summaries, start=1):
        marker = " (selected)" if state.selected_dataset == summary.csv_path else ""
        print(
            f"{idx:>2}. {format_repo_relative(config, summary.csv_path)} "
            f"[{dataset_label_display(summary)}] "
            f"moving={summary.driving_rows} "
            f"faults={summary.fault_rows} "
            f"ignored_non_moving={summary.non_moving_rows}{marker}"
        )

    selected_index = prompt_optional_index(
        len(usable_summaries), "Choose dataset number (Enter to cancel): "
    )
    if selected_index is None:
        print("Dataset selection canceled.")
        return

    selected_summary = usable_summaries[selected_index - 1]
    state.selected_dataset = selected_summary.csv_path
    print(
        "Selected dataset: "
        f"{format_repo_relative(config, state.selected_dataset)} "
        f"({dataset_label_display(selected_summary)}, "
        f"moving={selected_summary.driving_rows}, faults={selected_summary.fault_rows})"
    )


def build_auto_split(summaries: list[DatasetSummary]) -> tuple[list[Path], list[Path]]:
    """Build train/test paths from dataset labels."""
    usable = [s for s in summaries if s.driving_rows > 0]
    train_summaries = [s for s in usable if s.label == "no_faults"]
    test_summaries = [s for s in usable if s.label == "has_faults"]

    train_paths = [s.csv_path for s in train_summaries]
    test_paths = [s.csv_path for s in test_summaries]

    if not train_paths:
        raise ValueError("Auto split produced empty train set.")
    if not test_paths:
        raise ValueError("Auto split produced empty test set.")

    return train_paths, test_paths
