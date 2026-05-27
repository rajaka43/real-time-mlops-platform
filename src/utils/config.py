"""
Configuration Management — Loads settings from environment variables.
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    # API
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    log_level: str = "INFO"
    debug: bool = False

    # Model
    model_registry_path: str = "models/registry"
    model_artifacts_path: str = "models/artifacts"
    min_accuracy_to_promote: float = 0.005

    # Drift Detection
    drift_window_size: int = 500
    drift_check_interval: int = 50
    drift_psi_threshold: float = 0.20
    drift_ks_alpha: float = 0.05
    retraining_drift_threshold: int = 3

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "mlops-platform"

    # Redis
    redis_url: str = "redis://localhost:6379"
    redis_ttl_seconds: int = 3600

    # CI/CD Quality Gate
    min_accuracy_gate: float = 0.75
    min_f1_gate: float = 0.70


def load_config() -> Config:
    """Load configuration from environment variables with defaults."""
    return Config(
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        workers=int(os.getenv("API_WORKERS", "4")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        debug=os.getenv("DEBUG", "false").lower() == "true",

        model_registry_path=os.getenv("MODEL_REGISTRY_PATH", "models/registry"),
        model_artifacts_path=os.getenv("MODEL_ARTIFACTS_PATH", "models/artifacts"),
        min_accuracy_to_promote=float(os.getenv("MIN_ACCURACY_TO_PROMOTE", "0.005")),

        drift_window_size=int(os.getenv("DRIFT_WINDOW_SIZE", "500")),
        drift_check_interval=int(os.getenv("DRIFT_CHECK_INTERVAL", "50")),
        drift_psi_threshold=float(os.getenv("DRIFT_PSI_THRESHOLD", "0.20")),
        retraining_drift_threshold=int(os.getenv("RETRAINING_DRIFT_THRESHOLD", "3")),

        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        mlflow_experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "mlops-platform"),

        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        redis_ttl_seconds=int(os.getenv("REDIS_TTL_SECONDS", "3600")),

        min_accuracy_gate=float(os.getenv("MIN_ACCURACY", "0.75")),
        min_f1_gate=float(os.getenv("MIN_F1_SCORE", "0.70")),
    )


# Global config instance
config = load_config()
