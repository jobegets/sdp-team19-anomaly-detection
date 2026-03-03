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
