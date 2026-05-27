"""
Experiment Tracker — MLflow integration for logging runs, params, and metrics.
"""
import logging
import os
from typing import Dict, Any, Optional
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)

# Graceful import — MLflow is optional in minimal environments
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not installed. Experiment tracking disabled.")


class ExperimentTracker:
    """
    Wraps MLflow for experiment tracking.
    Falls back to local logging if MLflow is unavailable.
    """

    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "mlops-platform"
    ):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self._runs: list = []  # Fallback local store
        self._active_run = None

        if MLFLOW_AVAILABLE:
            try:
                mlflow.set_tracking_uri(tracking_uri)
                mlflow.set_experiment(experiment_name)
                logger.info(f"MLflow tracking: {tracking_uri} | experiment: {experiment_name}")
            except Exception as e:
                logger.warning(f"MLflow connection failed: {e}. Using local logging.")

    @contextmanager
    def start_run(self, run_name: Optional[str] = None):
        """Context manager for a training run."""
        run_name = run_name or f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        if MLFLOW_AVAILABLE:
            try:
                with mlflow.start_run(run_name=run_name) as run:
                    self._active_run = run
                    logger.info(f"MLflow run started: {run.info.run_id}")
                    yield self
                    self._active_run = None
                return
            except Exception as e:
                logger.warning(f"MLflow run failed: {e}. Falling back to local.")

        # Local fallback
        local_run = {"run_name": run_name, "params": {}, "metrics": {}, "tags": {}}
        self._runs.append(local_run)
        self._active_run = local_run
        yield self
        self._active_run = None
        logger.info(f"Local run completed: {run_name} | metrics: {local_run['metrics']}")

    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        if self._active_run is None:
            return
        if MLFLOW_AVAILABLE and hasattr(self._active_run, 'info'):
            try:
                mlflow.log_params({k: str(v) for k, v in params.items()})
                return
            except Exception:
                pass
        if isinstance(self._active_run, dict):
            self._active_run["params"].update(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log evaluation metrics."""
        if self._active_run is None:
            return
        if MLFLOW_AVAILABLE and hasattr(self._active_run, 'info'):
            try:
                mlflow.log_metrics(metrics, step=step)
                return
            except Exception:
                pass
        if isinstance(self._active_run, dict):
            self._active_run["metrics"].update(metrics)

    def log_tags(self, tags: Dict[str, str]):
        """Log run tags."""
        if self._active_run is None:
            return
        if MLFLOW_AVAILABLE and hasattr(self._active_run, 'info'):
            try:
                mlflow.set_tags(tags)
                return
            except Exception:
                pass
        if isinstance(self._active_run, dict):
            self._active_run["tags"].update(tags)

    def log_artifact(self, local_path: str):
        """Log a file artifact."""
        if MLFLOW_AVAILABLE:
            try:
                mlflow.log_artifact(local_path)
            except Exception as e:
                logger.warning(f"Artifact log failed: {e}")

    def get_run_history(self) -> list:
        """Return local run history (fallback mode)."""
        return self._runs


# Global singleton
_tracker: Optional[ExperimentTracker] = None


def get_tracker() -> ExperimentTracker:
    global _tracker
    if _tracker is None:
        _tracker = ExperimentTracker(
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
            experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "mlops-platform")
        )
    return _tracker
