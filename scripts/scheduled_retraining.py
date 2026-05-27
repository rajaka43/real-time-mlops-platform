"""
Scheduled Retraining Script
Runs as a background worker to check model performance and trigger retraining.
Can be invoked via: cron, Celery, GitHub Actions schedule, or Kubernetes CronJob.
"""
import asyncio
import logging
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main retraining orchestration."""
    logger.info("=" * 60)
    logger.info("Scheduled Retraining Check Started")
    logger.info(f"Timestamp: {datetime.utcnow().isoformat()}")
    logger.info("=" * 60)

    from src.pipeline.retraining_trigger import RetrainingTrigger
    from src.models.model_registry import ModelRegistry

    registry = ModelRegistry()
    await registry.initialize()

    # Get current production model metrics
    models = await registry.list_models()
    production = [m for m in models if m["status"] == "production"]

    if production:
        current = production[0]
        logger.info(f"Current production model: {current['model_id']}")
        logger.info(f"  Accuracy: {current['accuracy']}")
        logger.info(f"  F1 Score: {current['f1_score']}")
        logger.info(f"  Training samples: {current['training_samples']}")
        logger.info(f"  Promoted at: {current.get('promoted_at', 'N/A')}")
    else:
        logger.warning("No production model found!")

    # Run retraining
    trigger = RetrainingTrigger()
    job_id = f"scheduled_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    logger.info(f"\nLaunching retraining job: {job_id}")
    await trigger.run_retraining_pipeline(job_id)

    # Final status
    status = await trigger.get_status()
    logger.info("\nRetraining Complete:")
    logger.info(f"  State: {status['state']}")
    logger.info(f"  Last run: {status['last_run']}")
    logger.info(f"  New accuracy: {status.get('last_accuracy', 'N/A')}")
    logger.info(f"  Duration: {status.get('last_duration_seconds', 'N/A')}s")


if __name__ == "__main__":
    asyncio.run(main())
