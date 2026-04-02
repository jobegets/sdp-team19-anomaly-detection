import numpy as np
import pandas as pd

SEVERITY_ORDER: tuple[str, ...] = ("normal", "low", "medium", "high")
SEVERITY_TO_RANK = {label: idx for idx, label in enumerate(SEVERITY_ORDER)}


def apply_severity_levels(
    result_df: pd.DataFrame,
    *,
    threshold: float,
    medium_multiplier: float = 1.5,
    high_multiplier: float = 2.5,
) -> pd.DataFrame:
    """Add binary anomaly flags and severity levels derived from anomaly scores."""
    if "anomaly_score" not in result_df.columns:
        raise KeyError("Expected column 'anomaly_score' not found in result dataframe.")

    scores = result_df["anomaly_score"].astype(float).to_numpy()
    flagged = scores >= float(threshold)
    severity = np.full(scores.shape, "normal", dtype=object)
    severity[flagged] = "low"

    if flagged.any():
        if threshold > 0:
            medium_threshold = threshold * medium_multiplier
            high_threshold = threshold * high_multiplier
        else:
            flagged_scores = scores[flagged]
            medium_threshold = float(np.quantile(flagged_scores, 0.75))
            high_threshold = float(np.quantile(flagged_scores, 0.90))

        medium_mask = flagged & (scores >= medium_threshold)
        high_mask = flagged & (scores >= high_threshold)
        severity[medium_mask] = "medium"
        severity[high_mask] = "high"

    severity_rank = np.array([SEVERITY_TO_RANK[label] for label in severity], dtype=int)

    labeled_df = result_df.copy()
    labeled_df["anomalous_flag"] = flagged.astype(int)
    labeled_df["severity"] = pd.Categorical(
        severity,
        categories=list(SEVERITY_ORDER),
        ordered=True,
    )
    labeled_df["severity_rank"] = severity_rank
    return labeled_df
