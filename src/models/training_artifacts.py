from pathlib import Path

import joblib

REPO_ROOT = Path(__file__).resolve().parents[2]

ARTIFACT_PATHS: dict[str, Path] = {
    "isolation-forest": (REPO_ROOT / "artifacts" / "isolation_forest.pkl").resolve(),
    "isolation-forest-timeseries": (
        REPO_ROOT / "artifacts" / "isolation_forest_timeseries.pkl"
    ).resolve(),
    "autoencoder": (REPO_ROOT / "artifacts" / "autoencoder.pkl").resolve(),
    "autoencoder-lstm": (REPO_ROOT / "artifacts" / "autoencoder_lstm.pkl").resolve(),
}


def save_training_artifact(
    *,
    model_name: str,
    threshold: float,
    scaler: object,
    model: object,
    feature_cols: list[str],
    threshold_quantile: float,
    dataset_reference: str | None = None,
    contamination: float | None = None,
    window_size: int | None = None,
) -> Path:
    """Persist a trained model artifact using the canonical path for the model."""
    try:
        artifact_path = ARTIFACT_PATHS[model_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported model artifact path: {model_name}") from exc

    payload: dict[str, object] = {
        "model_name": model_name,
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "threshold": threshold,
        "threshold_quantile": threshold_quantile,
    }
    if dataset_reference is not None:
        payload["dataset_path"] = dataset_reference
    if contamination is not None:
        payload["contamination"] = contamination
    if window_size is not None:
        payload["window_size"] = window_size

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, artifact_path)
    print(f"Artifact updated: {artifact_path}")
    return artifact_path
