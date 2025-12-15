from pathlib import Path
import pandas as pd


def load_driving_data(csv_path: Path) -> pd.DataFrame:
    """Load and clean a driving CSV to a filtered DataFrame.

    Args:
        csv_path: Path to the raw sensor CSV.

    Returns:
        DataFrame with cleaned column names, sorted by time, and limited to samples where car is driving (RPM >0.5).

    Raises:
        FileNotFoundError: If the CSV path does not exist.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    columns = [
        "Time",
        "BMS Disch Enable",
        # Battery
        "Pack Voltage", "Pack Current", "Pack Temp", "State of Charge", "Min Cell Voltage", "BMS LV input",
        # Powertrain / inverter / motor
        "Torque Feedback",
        "RPM",
        "Flux Feedback",
        # Dynamics
        "InlineAcc", "LateralAcc", "VerticalAcc",
        "BrakeBias",
        "RollRate", "PitchRate", "YawRate",
    ]

    df = pd.read_csv(csv_path, skiprows=[1], usecols=columns)
    df.columns = df.columns.str.strip().str.replace('"', "")
    df = df.dropna(subset=["Time"]).sort_values("Time")

    # Only want data where car is moving
    # TODO. This is most likely problematic for calculating rolling mean. The resulting df could be discontinuous, which
    # would effect our model's performance (probably?)
    driving_df = df[df["RPM"] > 0.5].copy()

    return driving_df
