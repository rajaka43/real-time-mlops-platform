"""
Test Suite for MLOps Prediction Platform
Tests: API endpoints, model registry, drift detection, retraining pipeline
"""
import pytest
import asyncio
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────

@pytest.fixture
def mock_model():
    """Create a mock ML model for testing."""
    model = MagicMock()
    model.version = "1.0.0"
    model.preprocess = MagicMock(return_value=np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]))
    model.predict = MagicMock(return_value=(1, 0.87, {"0": 0.13, "1": 0.87}))
    model.metadata = MagicMock()
    model.metadata.feature_names = [
        "age_normalized", "income_normalized", "credit_score_normalized",
        "loan_amount_normalized", "employment_years", "debt_ratio"
    ]
    return model


@pytest.fixture
def sample_features():
    return {
        "age_normalized": 0.5,
        "income_normalized": 0.7,
        "credit_score_normalized": 0.8,
        "loan_amount_normalized": -0.3,
        "employment_years": 1.2,
        "debt_ratio": -0.5
    }


@pytest.fixture
def sample_prediction_request(sample_features):
    return {
        "features": sample_features,
        "return_probabilities": True
    }


# ─────────────────────────────────────────
# Unit Tests: Drift Detector
# ─────────────────────────────────────────

class TestDriftDetector:

    @pytest.mark.asyncio
    async def test_drift_detector_initializes(self):
        """Drift detector should initialize with reference distribution."""
        from src.monitoring.drift_detector import DriftDetector
        detector = DriftDetector()
        assert detector.reference_distribution is not None
        assert detector.reference_distribution.shape == (1000, 6)
        assert len(detector.feature_names) == 6

    @pytest.mark.asyncio
    async def test_no_drift_with_similar_data(self):
        """Similar data should not trigger drift alerts."""
        from src.monitoring.drift_detector import DriftDetector
        detector = DriftDetector()

        # Feed data similar to reference distribution
        for _ in range(60):
            features = {name: float(np.random.randn()) for name in detector.feature_names}
            await detector.check_drift(features)

        report = await detector.get_drift_report()
        assert detector.drift_alerts == 0 or detector.drift_alerts <= 1  # Tolerant check

    @pytest.mark.asyncio
    async def test_drift_detected_with_shifted_data(self):
        """Strongly shifted data should trigger drift detection."""
        from src.monitoring.drift_detector import DriftDetector
        detector = DriftDetector()
        detector.CHECK_INTERVAL = 10  # Speed up for tests

        # Feed data with 5-sigma shift (extreme drift)
        for _ in range(60):
            features = {name: float(np.random.randn() + 10.0) for name in detector.feature_names}
            await detector.check_drift(features)

        report = await detector.get_drift_report()
        assert detector.drift_alerts >= 1, "Should have detected drift with extreme shift"

    def test_psi_calculation(self):
        """PSI should be 0 for identical distributions, higher for different."""
        from src.monitoring.drift_detector import DriftDetector
        detector = DriftDetector()

        # Same distribution → low PSI
        ref = np.random.randn(500)
        psi_same = detector._calculate_psi(ref, ref)
        assert psi_same < 0.1, f"Same distribution PSI too high: {psi_same}"

        # Very different distribution → high PSI
        different = np.random.randn(500) + 5.0
        psi_diff = detector._calculate_psi(ref, different)
        assert psi_diff > 0.2, f"Different distribution PSI too low: {psi_diff}"

    def test_severity_classification(self):
        """Severity should be correctly classified from PSI and p-values."""
        from src.monitoring.drift_detector import DriftDetector
        detector = DriftDetector()

        assert detector._classify_severity(0.05, 0.5) == "none"
        assert detector._classify_severity(0.12, 0.03) == "low"
        assert detector._classify_severity(0.22, 0.008) == "medium"
        assert detector._classify_severity(0.30, 0.0001) == "high"


# ─────────────────────────────────────────
# Unit Tests: Metrics Collector
# ─────────────────────────────────────────

class TestMetricsCollector:

    @pytest.mark.asyncio
    async def test_metrics_collector_records_predictions(self):
        """Should correctly record prediction metrics."""
        from src.monitoring.metrics_collector import MetricsCollector
        collector = MetricsCollector()
        await collector.initialize()

        await collector.record_prediction(
            prediction_id="test-001",
            prediction=1,
            confidence=0.85,
            latency_ms=12.5,
            model_version="1.0.0"
        )

        assert collector.total_predictions == 1
        assert len(collector.predictions) == 1
        assert collector.predictions[0]["prediction_id"] == "test-001"

    @pytest.mark.asyncio
    async def test_metrics_summary_empty(self):
        """Should handle empty state gracefully."""
        from src.monitoring.metrics_collector import MetricsCollector
        collector = MetricsCollector()
        await collector.initialize()

        summary = await collector.get_summary()
        assert summary["status"] == "no_data"

    @pytest.mark.asyncio
    async def test_latency_percentiles(self):
        """Latency percentiles should be correctly calculated."""
        from src.monitoring.metrics_collector import MetricsCollector
        collector = MetricsCollector()
        await collector.initialize()

        # Record predictions with known latencies
        for i, latency in enumerate([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]):
            await collector.record_prediction(
                prediction_id=f"p{i}",
                prediction=1,
                confidence=0.9,
                latency_ms=float(latency),
                model_version="1.0.0"
            )

        summary = await collector.get_summary()
        assert summary["latency_ms"]["p50"] == 55.0
        assert summary["latency_ms"]["p95"] >= 90.0

    @pytest.mark.asyncio
    async def test_feedback_recording(self):
        """Feedback should be stored and linked to predictions."""
        from src.monitoring.metrics_collector import MetricsCollector
        collector = MetricsCollector()

        await collector.record_feedback("pred-001", 1, "correction")
        assert "pred-001" in collector.feedback_store
        assert collector.feedback_store["pred-001"]["actual_label"] == 1


# ─────────────────────────────────────────
# Unit Tests: Retraining Pipeline
# ─────────────────────────────────────────

class TestRetrainingTrigger:

    @pytest.mark.asyncio
    async def test_drift_threshold_triggers_retraining(self):
        """Consecutive drift events should trigger retraining evaluation."""
        from src.pipeline.retraining_trigger import RetrainingTrigger
        trigger = RetrainingTrigger()
        trigger._drift_threshold = 3

        with patch.object(trigger, 'run_retraining_pipeline', new_callable=AsyncMock) as mock_retrain:
            # First 2 drift events — should not trigger
            await trigger.evaluate_retraining_need()
            await trigger.evaluate_retraining_need()
            mock_retrain.assert_not_called()

            # Third event — should trigger
            await trigger.evaluate_retraining_need()
            # Note: asyncio.create_task is used, so we check the counter reset
            assert trigger._consecutive_drift_count == 0

    @pytest.mark.asyncio
    async def test_data_validation_passes_clean_data(self):
        """Clean data should pass validation."""
        from src.pipeline.retraining_trigger import RetrainingTrigger
        trigger = RetrainingTrigger()

        X = np.random.randn(500, 6)
        y = np.random.randint(0, 2, 500)

        is_valid, issues = trigger._validate_data(X, y)
        assert is_valid, f"Expected valid data but got issues: {issues}"

    @pytest.mark.asyncio
    async def test_data_validation_rejects_insufficient_samples(self):
        """Should reject datasets that are too small."""
        from src.pipeline.retraining_trigger import RetrainingTrigger
        trigger = RetrainingTrigger()

        X = np.random.randn(50, 6)
        y = np.random.randint(0, 2, 50)

        is_valid, issues = trigger._validate_data(X, y)
        assert not is_valid
        assert any("Insufficient" in issue for issue in issues)

    def test_feature_engineering_adds_features(self):
        """Feature engineering should expand feature count."""
        from src.pipeline.retraining_trigger import RetrainingTrigger
        trigger = RetrainingTrigger()

        X = np.random.randn(100, 6)
        X_engineered = trigger._engineer_features(X)

        assert X_engineered.shape[1] > X.shape[1], "Should have more features after engineering"
        assert X_engineered.shape[0] == X.shape[0], "Sample count should not change"

    @pytest.mark.asyncio
    async def test_pipeline_status_initial_state(self):
        """Pipeline status should start as idle."""
        from src.pipeline.retraining_trigger import RetrainingTrigger
        trigger = RetrainingTrigger()
        status = await trigger.get_status()
        assert status["state"] == "idle"
        assert status["runs_completed"] == 0


# ─────────────────────────────────────────
# Integration Tests: Model Registry
# ─────────────────────────────────────────

class TestModelRegistry:

    @pytest.mark.asyncio
    async def test_registry_initializes_and_creates_model(self, tmp_path, monkeypatch):
        """Registry should initialize and create bootstrap model."""
        from src.models.model_registry import ModelRegistry
        registry = ModelRegistry()
        registry.REGISTRY_DIR = tmp_path / "registry"
        registry.ARTIFACTS_DIR = tmp_path / "artifacts"

        await registry.initialize()

        assert registry.active_model is not None
        assert len(registry.registry) >= 1

    @pytest.mark.asyncio
    async def test_active_model_makes_predictions(self, tmp_path):
        """Active model should produce valid predictions."""
        from src.models.model_registry import ModelRegistry
        registry = ModelRegistry()
        registry.REGISTRY_DIR = tmp_path / "registry"
        registry.ARTIFACTS_DIR = tmp_path / "artifacts"

        await registry.initialize()
        model = await registry.get_active_model()

        assert model is not None

        features = {
            "age_normalized": 0.5,
            "income_normalized": 0.7,
            "credit_score_normalized": 0.8,
            "loan_amount_normalized": -0.3,
            "employment_years": 1.2,
            "debt_ratio": -0.5
        }

        processed = model.preprocess(features)
        prediction, confidence, _ = model.predict(processed)

        assert prediction in [0, 1]
        assert 0.0 <= confidence <= 1.0


# ─────────────────────────────────────────
# Main test runner
# ─────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
