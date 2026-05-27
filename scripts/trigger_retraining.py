"""
Trigger Retraining Script — Manual / CI invocation
Can be called from GitHub Actions or manually via CLI.
Usage: python scripts/trigger_retraining.py
"""
import sys
import os
import asyncio
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


async def main():
    api_url = os.getenv("API_URL", "")

    if api_url:
        # Trigger via live API
        import urllib.request
        import json
        logger.info(f"Triggering retraining via API: {api_url}")
        req = urllib.request.Request(
            f"{api_url}/retrain",
            method="POST",
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                logger.info(f"Retraining triggered: job_id={data.get('job_id')}")
        except Exception as e:
            logger.error(f"API call failed: {e}")
            sys.exit(1)
    else:
        # Run pipeline directly
        logger.info("No API_URL set — running pipeline directly")
        from src.pipeline.retraining_trigger import RetrainingTrigger
        trigger = RetrainingTrigger()
        job_id = f"ci_manual_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        await trigger.run_retraining_pipeline(job_id)
        status = await trigger.get_status()
        logger.info(f"Done. State={status['state']}, Accuracy={status.get('last_accuracy')}")


if __name__ == "__main__":
    asyncio.run(main())
