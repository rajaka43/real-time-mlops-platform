"""
Drift Check Script — CI/CD Drift Report
Generates a drift report and warns if features are drifting.
Run: python scripts/check_drift.py
"""
import sys
import os
import asyncio
import numpy as np
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


async def check():
    from src.monitoring.drift_detector import DriftDetector

    logger.info("=" * 50)
    logger.info("MLOps Drift Detection Report")
    logger.info("=" * 50)

    detector = DriftDetector()

    # Simulate recent predictions feeding into detector
    np.random.seed(int.from_bytes(os.urandom(4), 'big') % 10000)
    for _ in range(200):
        features = {name: float(np.random.randn()) for name in detector.feature_names}
        await detector.check_drift(features)

    report = await detector.get_drift_report()
    logger.info(f"Predictions monitored: {report.get('predictions_monitored', 0)}")
    logger.info(f"Total drift alerts:    {report.get('total_drift_alerts', 0)}")
    logger.info(f"Status:                {report.get('status', 'unknown')}")

    if report.get("latest_check"):
        latest = report["latest_check"]
        drifted = latest.get("features_drifted", 0)
        total = latest.get("total_features", 6)
        logger.info(f"Features drifting:     {drifted}/{total}")

        if drifted > 0:
            logger.warning(f"WARNING: {drifted} feature(s) showing drift — consider retraining")
        else:
            logger.info("All features stable ✓")

    logger.info("Drift check complete")


if __name__ == "__main__":
    asyncio.run(check())
