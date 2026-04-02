# Utility package for shared helpers.
from . import visualization as _visualization
from .data import (
    DatasetSummary,
    load_driving_data,
    summarize_dataset,
)

plot_results = _visualization.plot_results
print_evaluation = _visualization.print_evaluation
early_detection_stats = _visualization.early_detection_stats

# These visualization helpers may be intentionally disabled in some branches.
plot_features_over_time = getattr(_visualization, "plot_features_over_time", None)
plot_new_anomaly_windows = getattr(_visualization, "plot_new_anomaly_windows", None)

__all__ = [
    "plot_results",
    "print_evaluation",
    "early_detection_stats",
    "DatasetSummary",
    "load_driving_data",
    "summarize_dataset",
]
