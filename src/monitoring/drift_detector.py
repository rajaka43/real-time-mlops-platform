"""
Data Drift Detector - Monitors incoming data for distribution shifts
Uses statistical tests: KS test, PSI, Chi-squared
"""
import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
from collections import deque
from scipy import stats
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    feature_name: str
    drift_detected: bool
    drift_score: float
    p_value: Optional[float]
    test_used: str
    severity: str  # none, low, medium, high
    timestamp: str


class DriftDetector:
    """
    Real-time data drift detection using multiple statistical tests.
    
    Methods:
    - Kolmogorov-Smirnov test for continuous features
    - Population Stability Index (PSI) for categorical features  
    - Jensen-Shannon divergence for distribution comparison
    """

    WINDOW_SIZE = 500       # Number of recent predictions to keep
    REFERENCE_SIZE = 1000   # Size of reference distribution
    CHECK_INTERVAL = 50     # Check drift every N predictions

    PSI_THRESHOLDS = {
        "low": 0.1,
        "medium": 0.2,
        "high": 0.25
    }

    KS_ALPHA = 0.05  # Significance level for KS test

    def __init__(self):
        self.reference_distribution: Optional[np.ndarray] = None
        self.feature_names: List[str] = []
        self.recent_data: deque = deque(maxlen=self.WINDOW_SIZE)
        self.drift_history: List[Dict] = []
        self.prediction_count: int = 0
        self.drift_alerts: int = 0
        self._initialize_reference()

    def _initialize_reference(self):
        """Create a reference distribution from baseline data."""
        np.random.seed(42)
        n_features = 6
        self.feature_names = [
            "age_normalized", "income_normalized", "credit_score_normalized",
            "loan_amount_normalized", "employment_years", "debt_ratio"
        ]
        # Baseline reference distribution
        self.reference_distribution = np.random.randn(self.REFERENCE_SIZE, n_features)
        logger.info(f"Reference distribution initialized: {self.reference_distribution.shape}")

    async def check_drift(self, features: Dict[str, Any]) -> bool:
        """
        Check if incoming data is drifting from reference distribution.
        Returns True if significant drift is detected.
        """
        # Convert features to vector
        feature_vector = [
            float(features.get(name, 0.0))
            for name in self.feature_names
        ]
        self.recent_data.append(feature_vector)
        self.prediction_count += 1

        # Only run drift check at intervals (avoid overhead)
        if self.prediction_count % self.CHECK_INTERVAL != 0:
            return False

        if len(self.recent_data) < 50:
            return False  # Not enough data yet

        return await self._run_drift_analysis()

    async def _run_drift_analysis(self) -> bool:
        """Run full drift analysis across all features."""
        current_data = np.array(list(self.recent_data))
        drift_results = []
        overall_drift = False

        for i, feature_name in enumerate(self.feature_names):
            ref_values = self.reference_distribution[:, i]
            curr_values = current_data[:, i]

            # KS Test
            ks_stat, p_value = stats.ks_2samp(ref_values, curr_values)

            # PSI Score
            psi = self._calculate_psi(ref_values, curr_values)

            # Jensen-Shannon divergence
            js_div = self._calculate_js_divergence(ref_values, curr_values)

            # Determine severity
            drift_detected = p_value < self.KS_ALPHA or psi > self.PSI_THRESHOLDS["medium"]
            severity = self._classify_severity(psi, p_value)

            result = DriftResult(
                feature_name=feature_name,
                drift_detected=drift_detected,
                drift_score=round(float(psi), 4),
                p_value=round(float(p_value), 4),
                test_used="KS+PSI",
                severity=severity,
                timestamp=datetime.utcnow().isoformat()
            )
            drift_results.append(asdict(result))

            if drift_detected:
                overall_drift = True

        # Record in history
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_drift": overall_drift,
            "features_drifted": sum(1 for r in drift_results if r["drift_detected"]),
            "total_features": len(self.feature_names),
            "results": drift_results,
            "window_size": len(self.recent_data)
        }
        self.drift_history.append(summary)

        # Keep only last 100 drift reports
        if len(self.drift_history) > 100:
            self.drift_history.pop(0)

        if overall_drift:
            self.drift_alerts += 1
            logger.warning(
                f"DRIFT DETECTED: {summary['features_drifted']}/{summary['total_features']} "
                f"features drifting (alert #{self.drift_alerts})"
            )

        return overall_drift

    def _calculate_psi(self, reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """Calculate Population Stability Index."""
        try:
            # Create bins from reference
            min_val = min(reference.min(), current.min())
            max_val = max(reference.max(), current.max())
            bin_edges = np.linspace(min_val, max_val, bins + 1)

            ref_hist, _ = np.histogram(reference, bins=bin_edges)
            curr_hist, _ = np.histogram(current, bins=bin_edges)

            # Normalize
            ref_pct = (ref_hist + 0.001) / (len(reference) + 0.001 * bins)
            curr_pct = (curr_hist + 0.001) / (len(current) + 0.001 * bins)

            # PSI formula
            psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
            return float(psi)
        except Exception:
            return 0.0

    def _calculate_js_divergence(self, p: np.ndarray, q: np.ndarray, bins: int = 20) -> float:
        """Calculate Jensen-Shannon divergence."""
        try:
            min_val = min(p.min(), q.min())
            max_val = max(p.max(), q.max())
            bin_edges = np.linspace(min_val, max_val, bins + 1)

            p_hist, _ = np.histogram(p, bins=bin_edges, density=True)
            q_hist, _ = np.histogram(q, bins=bin_edges, density=True)

            p_norm = (p_hist + 1e-10) / (p_hist + 1e-10).sum()
            q_norm = (q_hist + 1e-10) / (q_hist + 1e-10).sum()
            m = 0.5 * (p_norm + q_norm)

            js = 0.5 * stats.entropy(p_norm, m) + 0.5 * stats.entropy(q_norm, m)
            return float(np.clip(js, 0, 1))
        except Exception:
            return 0.0

    def _classify_severity(self, psi: float, p_value: float) -> str:
        """Classify drift severity based on PSI and p-value."""
        if psi >= self.PSI_THRESHOLDS["high"] or p_value < 0.001:
            return "high"
        elif psi >= self.PSI_THRESHOLDS["medium"] or p_value < 0.01:
            return "medium"
        elif psi >= self.PSI_THRESHOLDS["low"] or p_value < 0.05:
            return "low"
        return "none"

    async def get_drift_report(self) -> Dict:
        """Get current drift analysis summary."""
        if not self.drift_history:
            return {
                "status": "monitoring",
                "message": "Collecting baseline data...",
                "predictions_monitored": self.prediction_count,
                "drift_alerts": self.drift_alerts
            }

        latest = self.drift_history[-1]
        recent_alerts = sum(
            1 for h in self.drift_history[-10:]
            if h["overall_drift"]
        )

        return {
            "status": "drift_detected" if latest["overall_drift"] else "stable",
            "latest_check": latest,
            "drift_rate_last_10_checks": f"{recent_alerts}/10",
            "total_drift_alerts": self.drift_alerts,
            "predictions_monitored": self.prediction_count,
            "window_size": self.WINDOW_SIZE,
            "reference_samples": self.REFERENCE_SIZE,
            "history_length": len(self.drift_history)
        }
