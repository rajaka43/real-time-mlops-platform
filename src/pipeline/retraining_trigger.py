"""
Automated Retraining Pipeline - Trains new models when triggered
"""
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)


class RetrainingTrigger:
    """
    Manages automated model retraining.
    Triggered by: drift detection, scheduled jobs, manual trigger, or performance degradation.
    """

    def __init__(self):
        self.status = {
            "state": "idle",
            "last_run": None,
            "last_job_id": None,
            "runs_completed": 0,
            "last_accuracy": None
        }
        self._retraining_threshold = 0.05  # 5% accuracy drop triggers retraining
        self._consecutive_drift_count = 0
        self._drift_threshold = 3  # 3 consecutive drift alerts = retrain

    async def evaluate_retraining_need(self) -> bool:
        """Evaluate whether retraining should be triggered."""
        self._consecutive_drift_count += 1

        if self._consecutive_drift_count >= self._drift_threshold:
            logger.info(f"Drift threshold exceeded ({self._consecutive_drift_count} events). Triggering retraining...")
            self._consecutive_drift_count = 0
            job_id = f"auto_retrain_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            asyncio.create_task(self.run_retraining_pipeline(job_id))
            return True
        return False

    async def run_retraining_pipeline(self, job_id: str):
        """
        Full automated retraining pipeline:
        1. Data collection & validation
        2. Feature engineering
        3. Hyperparameter optimization (Optuna)
        4. Model training & evaluation
        5. Model comparison & promotion
        """
        logger.info(f"Starting retraining pipeline: {job_id}")
        self.status["state"] = "running"
        self.status["last_job_id"] = job_id
        start_time = datetime.utcnow()

        try:
            # Step 1: Data Collection
            logger.info(f"[{job_id}] Step 1/5: Collecting training data...")
            X, y, feature_names = await self._collect_training_data()
            logger.info(f"[{job_id}] Collected {len(X)} samples with {X.shape[1]} features")

            # Step 2: Data Validation
            logger.info(f"[{job_id}] Step 2/5: Validating data quality...")
            is_valid, issues = self._validate_data(X, y)
            if not is_valid:
                raise ValueError(f"Data validation failed: {issues}")

            # Step 3: Feature Engineering
            logger.info(f"[{job_id}] Step 3/5: Engineering features...")
            X_engineered = self._engineer_features(X)

            # Step 4: Hyperparameter Optimization
            logger.info(f"[{job_id}] Step 4/5: Running hyperparameter optimization...")
            best_params, best_algorithm = await self._optimize_hyperparameters(
                X_engineered, y
            )
            logger.info(f"[{job_id}] Best algorithm: {best_algorithm}, params: {best_params}")

            # Step 5: Train Final Model
            logger.info(f"[{job_id}] Step 5/5: Training final model...")
            model_pipeline, metrics, version = await self._train_final_model(
                X_engineered, y, best_params, best_algorithm, feature_names
            )

            # Register and conditionally promote
            from src.models.model_registry import ModelRegistry
            registry = ModelRegistry()
            await registry.initialize()

            model_id = await registry.register_model(
                model_pipeline=model_pipeline,
                version=version,
                algorithm=best_algorithm,
                metrics=metrics,
                feature_names=feature_names,
                hyperparameters=best_params,
                training_samples=len(X)
            )

            # Auto-promote if better than current
            if await self._should_promote(metrics, registry):
                await registry.promote_model(model_id)
                logger.info(f"[{job_id}] New model auto-promoted: {model_id}")
            else:
                logger.info(f"[{job_id}] New model in staging (does not outperform current)")

            elapsed = (datetime.utcnow() - start_time).seconds
            self.status.update({
                "state": "idle",
                "last_run": datetime.utcnow().isoformat(),
                "runs_completed": self.status["runs_completed"] + 1,
                "last_accuracy": metrics["accuracy"],
                "last_duration_seconds": elapsed
            })
            logger.info(f"[{job_id}] Retraining complete in {elapsed}s. Accuracy: {metrics['accuracy']}")

        except Exception as e:
            logger.error(f"[{job_id}] Retraining failed: {str(e)}")
            self.status["state"] = "failed"
            self.status["last_error"] = str(e)

    async def _collect_training_data(self):
        """Collect and prepare training data."""
        # In production: query data warehouse, feature store, labeled feedback
        np.random.seed(int(datetime.utcnow().timestamp()) % 1000)
        n_samples = np.random.randint(4000, 8000)

        X = np.random.randn(n_samples, 6)
        # Add some concept drift simulation
        drift_magnitude = np.random.uniform(0.05, 0.15)
        X[:, 0] += drift_magnitude  # Simulate feature distribution shift

        y = (
            0.35 * X[:, 0] + 0.38 * X[:, 1] - 0.22 * X[:, 2] +
            0.15 * X[:, 3] + np.random.randn(n_samples) * 0.25
        ) > 0
        y = y.astype(int)

        feature_names = [
            "age_normalized", "income_normalized", "credit_score_normalized",
            "loan_amount_normalized", "employment_years", "debt_ratio"
        ]
        return X, y, feature_names

    def _validate_data(self, X: np.ndarray, y: np.ndarray):
        """Validate data quality before training."""
        issues = []

        if len(X) < 100:
            issues.append(f"Insufficient samples: {len(X)}")

        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            issues.append(f"NaN values found: {nan_count}")

        class_balance = np.bincount(y) / len(y)
        if min(class_balance) < 0.05:
            issues.append(f"Severe class imbalance: {class_balance}")

        return len(issues) == 0, issues

    def _engineer_features(self, X: np.ndarray) -> np.ndarray:
        """Apply feature engineering transformations."""
        # Interaction features
        interaction_1 = (X[:, 0] * X[:, 1]).reshape(-1, 1)
        interaction_2 = (X[:, 2] / (np.abs(X[:, 3]) + 1e-8)).reshape(-1, 1)

        # Polynomial features for top predictors
        poly_1 = (X[:, 0] ** 2).reshape(-1, 1)
        poly_2 = (X[:, 1] ** 2).reshape(-1, 1)

        X_engineered = np.hstack([X, interaction_1, interaction_2, poly_1, poly_2])
        return X_engineered

    async def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray):
        """Run Optuna hyperparameter optimization."""
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        def objective(trial):
            algorithm = trial.suggest_categorical(
                "algorithm", ["GradientBoosting", "RandomForest"]
            )

            if algorithm == "GradientBoosting":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                    "max_depth": trial.suggest_int("max_depth", 2, 6),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                }
                clf = GradientBoostingClassifier(**params, random_state=42)
            else:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                }
                clf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)

            pipeline = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
            pipeline.fit(X_train, y_train)
            return accuracy_score(y_val, pipeline.predict(X_val))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20, timeout=60)

        best_params = study.best_params.copy()
        best_algorithm = best_params.pop("algorithm")
        return best_params, best_algorithm

    async def _train_final_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: Dict,
        algorithm: str,
        feature_names: List[str]
    ):
        """Train the final model with best hyperparameters."""
        if algorithm == "GradientBoosting":
            clf = GradientBoostingClassifier(**params, random_state=42)
        else:
            clf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)

        pipeline = Pipeline([("scaler", StandardScaler()), ("clf", clf)])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        }

        # Generate version from timestamp
        ts = datetime.utcnow().strftime("%Y%m%d%H%M")
        version = f"2.{ts[:8]}.{ts[8:]}"

        return pipeline, metrics, version

    async def _should_promote(self, new_metrics: Dict, registry) -> bool:
        """Compare new model to current production model."""
        models = await registry.list_models()
        production_models = [m for m in models if m["status"] == "production"]

        if not production_models:
            return True  # No production model exists, promote automatically

        current_accuracy = production_models[0]["accuracy"]
        improvement = new_metrics["accuracy"] - current_accuracy

        logger.info(f"Model comparison: current={current_accuracy}, new={new_metrics['accuracy']}, delta={improvement:.4f}")
        return improvement >= 0.005  # Promote if at least 0.5% better

    async def get_status(self) -> Dict:
        """Return current pipeline status."""
        return {
            "pipeline": "automated_mlops",
            **self.status,
            "drift_count": self._consecutive_drift_count,
            "drift_threshold": self._drift_threshold
        }
