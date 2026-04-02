from pathlib import Path
from dataclasses import dataclass
import pandas as pd

MOVING_RPM_THRESHOLD = 0.5

DATA_COLUMNS = [
    "Time",
    "BMS Disch Enable",
    # Battery
    "Pack Voltage",
    "Pack Current",
    # "Pack Temp", High MSE
    "State of Charge",
    "Min Cell Voltage",
    # "BMS LV input", High MSE
    # Powertrain / inverter / motor
    "Torque Feedback",
    "RPM",
    "Flux Feedback",
    # Dynamics
    "InlineAcc",
    "LateralAcc",
    "VerticalAcc",
    "RollRate",
    "PitchRate",
    "YawRate",
]


@dataclass(frozen=True)
class DatasetSummary:
    """Summary stats used to organize datasets before splitting."""

    csv_path: Path
    total_rows: int
    driving_rows: int
    non_moving_rows: int
    fault_rows: int
    normal_rows: int

    @property
    def label(self) -> str:
        """Dataset label for split planning."""
        if self.driving_rows == 0:
            return "empty_after_filter"
        if self.fault_rows > 0:
            return "has_faults"
        return "no_faults"


def _load_raw_data(csv_path: Path) -> pd.DataFrame:
    """Load CSV with expected sensor columns and standard cleaning."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    df = pd.read_csv(csv_path, skiprows=[1], usecols=DATA_COLUMNS)
    df.columns = df.columns.str.strip().str.replace('"', "")
    df = df.dropna(subset=["Time"]).sort_values("Time")
    return df


def summarize_dataset(
    csv_path: Path, moving_rpm_threshold: float = MOVING_RPM_THRESHOLD
) -> DatasetSummary:
    """Summarize a dataset after removing non-moving samples."""
    df = _load_raw_data(csv_path)
    moving_mask = df["RPM"] > moving_rpm_threshold
    driving_df = df[moving_mask]
    fault_mask = driving_df["BMS Disch Enable"] == 0

    driving_rows = int(driving_df.shape[0])
    fault_rows = int(fault_mask.sum())
    normal_rows = int(driving_rows - fault_rows)

    return DatasetSummary(
        csv_path=csv_path.resolve(),
        total_rows=int(df.shape[0]),
        driving_rows=driving_rows,
        non_moving_rows=int((~moving_mask).sum()),
        fault_rows=fault_rows,
        normal_rows=normal_rows,
    )
def load_driving_data(csv_path: Path) -> pd.DataFrame:
    """Load and clean a driving CSV to a filtered DataFrame.

    Args:
        csv_path: Path to the raw sensor CSV.

    Returns:
        DataFrame with cleaned column names, sorted by time, and limited to samples where car is driving (RPM >0.5).

    Raises:
        FileNotFoundError: If the CSV path does not exist.
    """
    df = _load_raw_data(csv_path)

    # Only want data where car is moving
    # TODO. This is most likely problematic for calculating rolling mean. The resulting df could be discontinuous, which
    # would effect our model's performance (probably?)
    driving_df = df[df["RPM"] > MOVING_RPM_THRESHOLD].copy()

    return driving_df
