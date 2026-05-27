"""
Model Registry - Manages model versions, promotion, and lifecycle
"""
import os
import json
import pickle
import hashlib
import logging
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    model_id: str
    version: str
    algorithm: str
    status: str  # training, staging, production, archived
    accuracy: float
    f1_score: float
    created_at: str
    promoted_at: Optional[str]
    training_samples: int
    feature_names: List[str]
    hyperparameters: Dict[str, Any]
    artifact_path: str


class MLModel:
    """Wrapper for ML model with preprocessing and prediction logic."""

    def __init__(self, metadata: ModelMetadata, pipeline: Pipeline):
        self.metadata = metadata
        self.pipeline = pipeline
        self.version = metadata.version
        self.model_id = metadata.model_id

    def preprocess(self, features: Dict[str, Any]) -> np.ndarray:
        """Preprocess raw features for prediction."""
        feature_vector = []
        for feat_name in self.metadata.feature_names:
            value = features.get(feat_name, 0)
            feature_vector.append(float(value))
        return np.array(feature_vector).reshape(1, -1)

    def predict(
        self,
        features: np.ndarray,
        return_proba: bool = False
    ) -> Tuple[Any, float, Optional[Dict]]:
        """Run prediction and return result with confidence."""
        try:
            prediction = self.pipeline.predict(features)[0]
            probabilities = self.pipeline.predict_proba(features)[0]
            confidence = float(np.max(probabilities))

            proba_dict = None
            if return_proba:
                classes = self.pipeline.classes_
                proba_dict = {
                    str(cls): round(float(prob), 4)
                    for cls, prob in zip(classes, probabilities)
                }

            return int(prediction), confidence, proba_dict

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise


class ModelRegistry:
    """
    Central registry for all model versions.
    Handles versioning, promotion, and artifact storage.
    """

    REGISTRY_DIR = Path("models/registry")
    ARTIFACTS_DIR = Path("models/artifacts")

    def __init__(self):
        self.registry: Dict[str, ModelMetadata] = {}
        self.active_model: Optional[MLModel] = None
        self._initialized = False

    async def initialize(self):
        """Initialize registry, load or create default model."""
        self.REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        self.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        # Load existing registry
        registry_file = self.REGISTRY_DIR / "registry.json"
        if registry_file.exists():
            with open(registry_file) as f:
                data = json.load(f)
                self.registry = {k: ModelMetadata(**v) for k, v in data.items()}
            logger.info(f"Loaded {len(self.registry)} models from registry")
        else:
            # Bootstrap with a default model
            await self._create_bootstrap_model()

        # Load active (production) model
        await self._load_active_model()
        self._initialized = True
        logger.info("Model registry initialized")

    async def _create_bootstrap_model(self):
        """Create and register an initial model for bootstrapping."""
        logger.info("Creating bootstrap model...")

        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 5000
        X = np.random.randn(n_samples, 6)
        y = (
            0.3 * X[:, 0] + 0.4 * X[:, 1] - 0.2 * X[:, 2] +
            0.1 * X[:, 3] + np.random.randn(n_samples) * 0.3
        ) > 0
        y = y.astype(int)

        feature_names = [
            "age_normalized", "income_normalized", "credit_score_normalized",
            "loan_amount_normalized", "employment_years", "debt_ratio"
        ]

        # Train model
        model_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42
            ))
        ])
        model_pipeline.fit(X, y)

        # Evaluate
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import f1_score
        cv_scores = cross_val_score(model_pipeline, X, y, cv=5)
        y_pred = model_pipeline.predict(X)
        f1 = f1_score(y, y_pred)

        # Create metadata
        model_id = "gbc_v1_0_0"
        version = "1.0.0"
        artifact_path = str(self.ARTIFACTS_DIR / f"{model_id}.pkl")

        # Save artifact
        with open(artifact_path, "wb") as f:
            pickle.dump(model_pipeline, f)

        metadata = ModelMetadata(
            model_id=model_id,
            version=version,
            algorithm="GradientBoostingClassifier",
            status="production",
            accuracy=round(float(cv_scores.mean()), 4),
            f1_score=round(float(f1), 4),
            created_at=datetime.utcnow().isoformat(),
            promoted_at=datetime.utcnow().isoformat(),
            training_samples=n_samples,
            feature_names=feature_names,
            hyperparameters={
                "n_estimators": 100,
                "max_depth": 4,
                "learning_rate": 0.1
            },
            artifact_path=artifact_path
        )

        self.registry[model_id] = metadata
        await self._save_registry()
        logger.info(f"Bootstrap model created: {model_id} (acc={metadata.accuracy})")

    async def _load_active_model(self):
        """Load the production model into memory."""
        for model_id, metadata in self.registry.items():
            if metadata.status == "production":
                try:
                    with open(metadata.artifact_path, "rb") as f:
                        pipeline = pickle.load(f)
                    self.active_model = MLModel(metadata, pipeline)
                    logger.info(f"Active model loaded: {model_id}")
                    return
                except Exception as e:
                    logger.error(f"Failed to load model {model_id}: {e}")

        logger.warning("No production model found in registry")

    async def get_active_model(self, version: Optional[str] = None) -> Optional[MLModel]:
        """Get the active model (or specific version)."""
        if version:
            for model_id, metadata in self.registry.items():
                if metadata.version == version:
                    with open(metadata.artifact_path, "rb") as f:
                        pipeline = pickle.load(f)
                    return MLModel(metadata, pipeline)
        return self.active_model

    async def register_model(
        self,
        model_pipeline: Pipeline,
        version: str,
        algorithm: str,
        metrics: Dict[str, float],
        feature_names: List[str],
        hyperparameters: Dict[str, Any],
        training_samples: int
    ) -> str:
        """Register a new model in the registry."""
        model_id = f"{algorithm.lower().replace(' ', '_')}_v{version.replace('.', '_')}"
        artifact_path = str(self.ARTIFACTS_DIR / f"{model_id}.pkl")

        # Save artifact
        with open(artifact_path, "wb") as f:
            pickle.dump(model_pipeline, f)

        metadata = ModelMetadata(
            model_id=model_id,
            version=version,
            algorithm=algorithm,
            status="staging",
            accuracy=round(metrics.get("accuracy", 0), 4),
            f1_score=round(metrics.get("f1_score", 0), 4),
            created_at=datetime.utcnow().isoformat(),
            promoted_at=None,
            training_samples=training_samples,
            feature_names=feature_names,
            hyperparameters=hyperparameters,
            artifact_path=artifact_path
        )

        self.registry[model_id] = metadata
        await self._save_registry()
        logger.info(f"Model registered: {model_id} (status=staging)")
        return model_id

    async def promote_model(self, model_id: str) -> bool:
        """Promote a staging model to production."""
        if model_id not in self.registry:
            return False

        # Archive current production model
        for mid, meta in self.registry.items():
            if meta.status == "production":
                self.registry[mid].status = "archived"

        # Promote new model
        self.registry[model_id].status = "production"
        self.registry[model_id].promoted_at = datetime.utcnow().isoformat()

        await self._save_registry()
        await self._load_active_model()

        logger.info(f"Model promoted to production: {model_id}")
        return True

    async def list_models(self) -> List[Dict]:
        """List all registered models."""
        return [
            {**asdict(meta), "is_active": meta.status == "production"}
            for meta in self.registry.values()
        ]

    async def get_status(self) -> str:
        """Get registry health status."""
        if self.active_model:
            return f"healthy (active: {self.active_model.version})"
        return "degraded (no active model)"

    async def _save_registry(self):
        """Persist registry to disk."""
        registry_file = self.REGISTRY_DIR / "registry.json"
        data = {k: asdict(v) for k, v in self.registry.items()}
        with open(registry_file, "w") as f:
            json.dump(data, f, indent=2)
