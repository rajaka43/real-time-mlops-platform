"""
MLOps Platform - FastAPI Real-Time Prediction API
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn
import logging
import time
import uuid
from datetime import datetime

from src.models.model_registry import ModelRegistry
from src.monitoring.drift_detector import DriftDetector
from src.monitoring.metrics_collector import MetricsCollector
from src.pipeline.retraining_trigger import RetrainingTrigger
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

app = FastAPI(
    title="MLOps Prediction Platform",
    description="Real-time AI prediction with automated MLOps pipeline",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
model_registry = ModelRegistry()
drift_detector = DriftDetector()
metrics_collector = MetricsCollector()
retraining_trigger = RetrainingTrigger()


class PredictionRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="Input features for prediction")
    model_version: Optional[str] = Field(None, description="Specific model version to use")
    return_probabilities: bool = Field(False, description="Return class probabilities")

    class Config:
        json_schema_extra = {
            "example": {
                "features": {
                    "age": 35,
                    "income": 75000,
                    "credit_score": 720,
                    "loan_amount": 25000
                },
                "return_probabilities": True
            }
        }


class PredictionResponse(BaseModel):
    prediction_id: str
    prediction: Any
    confidence: float
    model_version: str
    latency_ms: float
    timestamp: str
    probabilities: Optional[Dict[str, float]] = None


class BatchPredictionRequest(BaseModel):
    records: List[Dict[str, Any]]
    model_version: Optional[str] = None


class FeedbackRequest(BaseModel):
    prediction_id: str
    actual_label: Any
    feedback_type: str = "correction"


@app.on_event("startup")
async def startup_event():
    """Initialize all services on startup."""
    logger.info("Starting MLOps Prediction Platform...")
    await model_registry.initialize()
    await metrics_collector.initialize()
    logger.info("Platform ready for predictions")


@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "MLOps Prediction Platform",
        "status": "operational",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Comprehensive health check."""
    model_status = await model_registry.get_status()
    return {
        "status": "healthy",
        "components": {
            "api": "healthy",
            "model_registry": model_status,
            "drift_detector": "healthy",
            "metrics_collector": "healthy"
        },
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks
):
    """
    Make a real-time prediction.
    
    - Loads the active model from registry
    - Monitors for data drift
    - Records metrics for observability
    - Triggers retraining if drift detected
    """
    start_time = time.time()
    prediction_id = str(uuid.uuid4())

    try:
        # Get active model
        model = await model_registry.get_active_model(request.model_version)
        if not model:
            raise HTTPException(status_code=503, detail="No active model available")

        # Preprocess features
        processed_features = model.preprocess(request.features)

        # Make prediction
        prediction, confidence, probabilities = model.predict(
            processed_features,
            return_proba=request.return_probabilities
        )

        latency_ms = (time.time() - start_time) * 1000

        response = PredictionResponse(
            prediction_id=prediction_id,
            prediction=prediction,
            confidence=confidence,
            model_version=model.version,
            latency_ms=round(latency_ms, 2),
            timestamp=datetime.utcnow().isoformat(),
            probabilities=probabilities if request.return_probabilities else None
        )

        # Background: record metrics & check drift
        background_tasks.add_task(
            _post_prediction_tasks,
            prediction_id,
            request.features,
            prediction,
            confidence,
            latency_ms,
            model.version
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", tags=["Predictions"])
async def batch_predict(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks
):
    """Batch prediction endpoint for multiple records."""
    if len(request.records) > 1000:
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 1000 records")

    model = await model_registry.get_active_model(request.model_version)
    if not model:
        raise HTTPException(status_code=503, detail="No active model available")

    results = []
    for record in request.records:
        pred_id = str(uuid.uuid4())
        processed = model.preprocess(record)
        prediction, confidence, _ = model.predict(processed)
        results.append({
            "prediction_id": pred_id,
            "prediction": prediction,
            "confidence": confidence,
            "model_version": model.version
        })

    return {
        "batch_id": str(uuid.uuid4()),
        "total_records": len(results),
        "predictions": results,
        "model_version": model.version,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/feedback", tags=["Feedback"])
async def submit_feedback(request: FeedbackRequest, background_tasks: BackgroundTasks):
    """Submit ground truth feedback for model evaluation."""
    background_tasks.add_task(
        metrics_collector.record_feedback,
        request.prediction_id,
        request.actual_label,
        request.feedback_type
    )
    return {
        "status": "feedback_recorded",
        "prediction_id": request.prediction_id,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/models", tags=["Model Management"])
async def list_models():
    """List all registered models and their status."""
    models = await model_registry.list_models()
    return {"models": models, "total": len(models)}


@app.post("/models/{model_id}/promote", tags=["Model Management"])
async def promote_model(model_id: str):
    """Promote a model version to active/production."""
    success = await model_registry.promote_model(model_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    return {"status": "promoted", "model_id": model_id}


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Get current system and model metrics."""
    return await metrics_collector.get_summary()


@app.get("/drift", tags=["Monitoring"])
async def get_drift_status():
    """Get current data drift analysis."""
    return await drift_detector.get_drift_report()


@app.post("/retrain", tags=["Pipeline"])
async def trigger_retraining(background_tasks: BackgroundTasks):
    """Manually trigger model retraining."""
    job_id = str(uuid.uuid4())
    background_tasks.add_task(retraining_trigger.run_retraining_pipeline, job_id)
    return {
        "status": "retraining_triggered",
        "job_id": job_id,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/pipeline/status", tags=["Pipeline"])
async def get_pipeline_status():
    """Get current pipeline execution status."""
    return await retraining_trigger.get_status()


async def _post_prediction_tasks(
    prediction_id: str,
    features: Dict,
    prediction: Any,
    confidence: float,
    latency_ms: float,
    model_version: str
):
    """Background tasks after each prediction."""
    try:
        # Record metrics
        await metrics_collector.record_prediction(
            prediction_id, prediction, confidence, latency_ms, model_version
        )

        # Check for drift
        drift_detected = await drift_detector.check_drift(features)
        if drift_detected:
            logger.warning(f"Data drift detected! Evaluating retraining need...")
            await retraining_trigger.evaluate_retraining_need()

    except Exception as e:
        logger.error(f"Post-prediction task failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
