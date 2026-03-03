from src.repl.dataset_ops import (
    choose_dataset,
    dataset_label_display,
    format_repo_relative,
    get_summary_for_path,
    refresh_dataset_summaries,
)
from src.repl.prompts import prompt_menu_choice
from src.repl.training_ops import (
    compare_session_runs,
    evaluate_last_trained_model,
    run_auto_split_training,
    train_on_selected_dataset,
)
from src.repl.types import AppConfig, SessionState


def print_repl_menu(config: AppConfig, state: SessionState) -> None:
    """Render REPL menu with session context."""
    if state.selected_dataset is not None:
        summary = get_summary_for_path(state, state.selected_dataset)
        if summary is None:
            selected_dataset = format_repo_relative(config, state.selected_dataset)
        else:
            selected_dataset = (
                f"{format_repo_relative(config, state.selected_dataset)} "
                f"[{dataset_label_display(summary)} | "
                f"moving={summary.driving_rows} faults={summary.fault_rows}]"
            )
    else:
        selected_dataset = "None"

    last_run = state.trained_runs[-1] if state.trained_runs else None
    last_run_label = (
        f"#{last_run.run_id} {last_run.model_name} ({last_run.dataset_label})"
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
    print("5. Auto split train/test (NO_FAULTS train, HAS_FAULTS test)")
    print("6. Exit")


def run_repl(config: AppConfig) -> int:
    """Start and run the interactive REPL loop."""
    state = SessionState()
    print("Starting anomaly detection REPL.")
    refresh_dataset_summaries(config, state)

    total = len(state.dataset_summaries)
    has_faults = sum(1 for s in state.dataset_summaries if s.label == "has_faults")
    no_faults = sum(1 for s in state.dataset_summaries if s.label == "no_faults")
    empty = sum(1 for s in state.dataset_summaries if s.label == "empty_after_filter")
    if total:
        print(
            "Dataset inventory loaded: "
            f"{total} files ({has_faults} has_faults, {no_faults} no_faults, {empty} empty_after_filter)."
        )
        print(f"Manifest written to: {config.dataset_manifest_path}")
    else:
        print(f"No datasets found under {config.dataset_root}")

    while True:
        print_repl_menu(config, state)
        choice = prompt_menu_choice(
            "Select option [1-6]: ", {"1", "2", "3", "4", "5", "6"}
        )

        try:
            if choice == "1":
                choose_dataset(config, state)
            elif choice == "2":
                train_on_selected_dataset(config, state)
            elif choice == "3":
                evaluate_last_trained_model(config, state)
            elif choice == "4":
                compare_session_runs(config, state)
            elif choice == "5":
                run_auto_split_training(config, state)
            else:
                print("Exiting REPL.")
                return 0
        except Exception as exc:
            print(f"Error: {exc}")
