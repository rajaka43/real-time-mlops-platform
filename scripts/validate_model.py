"""
Model Validation Script — CI/CD Quality Gate
Ensures model meets minimum performance thresholds before deployment.
Run: python scripts/validate_model.py
"""
import sys
import os
import asyncio
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

MIN_ACCURACY = float(os.getenv("MIN_ACCURACY", "0.75"))
MIN_F1_SCORE = float(os.getenv("MIN_F1_SCORE", "0.70"))


async def validate():
    from src.models.model_registry import ModelRegistry
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    logger.info("=" * 50)
    logger.info("MLOps Model Validation — Quality Gate")
    logger.info("=" * 50)
    logger.info(f"Thresholds: accuracy >= {MIN_ACCURACY}, f1 >= {MIN_F1_SCORE}")

    registry = ModelRegistry()
    await registry.initialize()

    model = await registry.get_active_model()
    if model is None:
        logger.error("FAIL: No active model found in registry")
        sys.exit(1)

    logger.info(f"Validating model: {model.model_id} (v{model.version})")

    # Generate holdout test set
    np.random.seed(99)
    n = 2000
    X = np.random.randn(n, 6)
    y = (0.3*X[:,0] + 0.4*X[:,1] - 0.2*X[:,2] + np.random.randn(n)*0.3) > 0
    y = y.astype(int)

    processed = model.pipeline.predict(X)
    acc = accuracy_score(y, processed)
    f1 = f1_score(y, processed)

    logger.info(f"Results on {n} holdout samples:")
    logger.info(f"  Accuracy : {acc:.4f}  (min: {MIN_ACCURACY})")
    logger.info(f"  F1 Score : {f1:.4f}  (min: {MIN_F1_SCORE})")

    passed = True
    if acc < MIN_ACCURACY:
        logger.error(f"FAIL: Accuracy {acc:.4f} below threshold {MIN_ACCURACY}")
        passed = False
    if f1 < MIN_F1_SCORE:
        logger.error(f"FAIL: F1 {f1:.4f} below threshold {MIN_F1_SCORE}")
        passed = False

    if passed:
        logger.info("PASS: Model meets all quality thresholds ✓")
        sys.exit(0)
    else:
        logger.error("FAIL: Model does not meet quality thresholds")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(validate())
