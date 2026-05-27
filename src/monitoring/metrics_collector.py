"""
Metrics Collector - Records and aggregates model performance metrics
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Collects real-time metrics for:
    - Prediction latency
    - Model accuracy (from feedback)
    - Request volume
    - Confidence distributions
    - Error rates
    """

    def __init__(self):
        self.predictions: deque = deque(maxlen=10000)
        self.feedback_store: Dict[str, Dict] = {}
        self.latency_history: deque = deque(maxlen=1000)
        self.hourly_counts: defaultdict = defaultdict(int)
        self.error_count: int = 0
        self.total_predictions: int = 0

    async def initialize(self):
        logger.info("Metrics collector initialized")

    async def record_prediction(
        self,
        prediction_id: str,
        prediction: Any,
        confidence: float,
        latency_ms: float,
        model_version: str
    ):
        """Record a prediction event."""
        record = {
            "prediction_id": prediction_id,
            "prediction": prediction,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "model_version": model_version,
            "timestamp": datetime.utcnow().isoformat(),
            "hour_bucket": datetime.utcnow().strftime("%Y-%m-%d %H:00")
        }
        self.predictions.append(record)
        self.latency_history.append(latency_ms)
        self.hourly_counts[record["hour_bucket"]] += 1
        self.total_predictions += 1

    async def record_feedback(
        self,
        prediction_id: str,
        actual_label: Any,
        feedback_type: str
    ):
        """Record ground truth feedback for accuracy tracking."""
        self.feedback_store[prediction_id] = {
            "actual_label": actual_label,
            "feedback_type": feedback_type,
            "recorded_at": datetime.utcnow().isoformat()
        }
        logger.info(f"Feedback recorded for prediction {prediction_id}")

    async def get_summary(self) -> Dict:
        """Generate comprehensive metrics summary."""
        if not self.predictions:
            return {"status": "no_data", "total_predictions": 0}

        recent = list(self.predictions)
        latencies = [r["latency_ms"] for r in recent]
        confidences = [r["confidence"] for r in recent]

        # Accuracy from feedback
        accuracy_metrics = self._calculate_accuracy()

        # Latency percentiles
        latency_p50 = float(np.percentile(latencies, 50))
        latency_p95 = float(np.percentile(latencies, 95))
        latency_p99 = float(np.percentile(latencies, 99))

        # Confidence distribution
        conf_mean = float(np.mean(confidences))
        low_confidence = sum(1 for c in confidences if c < 0.6) / len(confidences)

        # Request rate (last hour)
        recent_hour = datetime.utcnow().strftime("%Y-%m-%d %H:00")
        requests_last_hour = self.hourly_counts.get(recent_hour, 0)

        # Version distribution
        version_counts = defaultdict(int)
        for r in recent:
            version_counts[r["model_version"]] += 1

        return {
            "summary": {
                "total_predictions": self.total_predictions,
                "predictions_with_feedback": len(self.feedback_store),
                "error_rate": round(self.error_count / max(self.total_predictions, 1), 4),
                "requests_last_hour": requests_last_hour
            },
            "latency_ms": {
                "p50": round(latency_p50, 2),
                "p95": round(latency_p95, 2),
                "p99": round(latency_p99, 2),
                "mean": round(float(np.mean(latencies)), 2)
            },
            "confidence": {
                "mean": round(conf_mean, 3),
                "low_confidence_rate": round(low_confidence, 3),
                "distribution": self._bucket_distribution(confidences)
            },
            "accuracy": accuracy_metrics,
            "model_versions": dict(version_counts),
            "throughput": {
                "total": self.total_predictions,
                "last_hour": requests_last_hour,
                "rolling_window": len(recent)
            }
        }

    def _calculate_accuracy(self) -> Dict:
        """Calculate accuracy from feedback records."""
        if len(self.feedback_store) < 10:
            return {"status": "insufficient_feedback", "samples": len(self.feedback_store)}

        correct = 0
        evaluated = 0
        for pred in self.predictions:
            pid = pred["prediction_id"]
            if pid in self.feedback_store:
                actual = self.feedback_store[pid]["actual_label"]
                if str(pred["prediction"]) == str(actual):
                    correct += 1
                evaluated += 1

        if evaluated == 0:
            return {"status": "no_matched_feedback"}

        return {
            "status": "calculated",
            "accuracy": round(correct / evaluated, 4),
            "evaluated_samples": evaluated
        }

    def _bucket_distribution(self, values: List[float], buckets: int = 5) -> List[Dict]:
        """Create bucketed distribution for visualization."""
        if not values:
            return []
        hist, edges = np.histogram(values, bins=buckets, range=(0, 1))
        return [
            {
                "range": f"{edges[i]:.1f}-{edges[i+1]:.1f}",
                "count": int(hist[i])
            }
            for i in range(len(hist))
        ]
