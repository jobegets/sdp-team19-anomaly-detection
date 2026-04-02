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
    plot_last_results,
    train_on_selected_dataset,
)
from src.repl.types import AppConfig, SessionState


def print_repl_menu(config: AppConfig, state: SessionState) -> None:
    """Render REPL menu with session context."""
    if state.selected_training_dataset is not None:
        summary = get_summary_for_path(state, state.selected_training_dataset)
        if summary is None:
            selected_training_dataset = format_repo_relative(
                config, state.selected_training_dataset
            )
        else:
            selected_training_dataset = (
                f"{format_repo_relative(config, state.selected_training_dataset)} "
                f"[{dataset_label_display(summary)} | "
                f"moving={summary.driving_rows} faults={summary.fault_rows}]"
            )
    else:
        selected_training_dataset = "None"

    if state.selected_evaluation_dataset is not None:
        summary = get_summary_for_path(state, state.selected_evaluation_dataset)
        if summary is None:
            selected_evaluation_dataset = format_repo_relative(
                config, state.selected_evaluation_dataset
            )
        else:
            selected_evaluation_dataset = (
                f"{format_repo_relative(config, state.selected_evaluation_dataset)} "
                f"[{dataset_label_display(summary)} | "
                f"moving={summary.driving_rows} faults={summary.fault_rows}]"
            )
    else:
        selected_evaluation_dataset = "None"

    last_run = state.trained_runs[-1] if state.trained_runs else None
    last_run_label = (
        f"#{last_run.run_id} {last_run.model_name} "
        f"(train={last_run.training_dataset_label}, "
        f"eval={last_run.evaluation_dataset_label or 'NOT_EVALUATED'})"
        if last_run
        else "None"
    )

    print("\n=== Anomaly Detection REPL ===")
    print(f"Selected training dataset: {selected_training_dataset}")
    print(f"Selected evaluation dataset: {selected_evaluation_dataset}")
    print(f"Last trained model: {last_run_label}")
    print("1. Choose training dataset")
    print("2. Choose evaluation dataset")
    print("3. Train model on chosen training dataset (always updates artifact)")
    print("4. Evaluate last trained model on chosen evaluation dataset")
    print("5. Compare trained models in this session")
    print("6. Plot results from last evaluated run")
    print("7. Exit")


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
    else:
        print(f"No datasets found under {config.dataset_root}")

    while True:
        print_repl_menu(config, state)
        choice = prompt_menu_choice(
            "Select option [1-7]: ", {"1", "2", "3", "4", "5", "6", "7"}
        )

        try:
            if choice == "1":
                choose_dataset(config, state, target="train")
            elif choice == "2":
                choose_dataset(config, state, target="eval")
            elif choice == "3":
                train_on_selected_dataset(config, state)
            elif choice == "4":
                evaluate_last_trained_model(config, state)
            elif choice == "5":
                compare_session_runs(config, state)
            elif choice == "6":
                plot_last_results(state)
            else:
                print("Exiting REPL.")
                return 0
        except Exception as exc:
            print(f"Error: {exc}")
