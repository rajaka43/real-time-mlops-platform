# Architecture Documentation

## System Components

### 1. Prediction API (`src/api/main.py`)
FastAPI application exposing all prediction and management endpoints. Handles async background tasks for metrics recording and drift checking after every prediction.

### 2. Model Registry (`src/models/model_registry.py`)
Manages the full lifecycle of ML models:
- Registers new models with metadata and artifacts
- Handles staging → production promotion
- Loads and caches the active production model in memory
- Persists registry state to `models/registry/registry.json`

### 3. Retraining Pipeline (`src/pipeline/retraining_trigger.py`)
Automated pipeline triggered by drift, schedule, or manual call:
1. Data collection (simulated; replace with your data warehouse query)
2. Data validation (sample count, NaN check, class balance)
3. Feature engineering (interaction + polynomial features)
4. Hyperparameter optimization with Optuna (20 trials)
5. Final model training + evaluation
6. Auto-promotion if ≥ 0.5% accuracy improvement

### 4. Drift Detector (`src/monitoring/drift_detector.py`)
Real-time statistical drift detection per feature:
- **KS Test** — detects shape changes in continuous distributions
- **PSI (Population Stability Index)** — standard industry metric (> 0.2 = medium drift)
- **Jensen-Shannon Divergence** — symmetric probability distance

### 5. Metrics Collector (`src/monitoring/metrics_collector.py`)
Records per-prediction telemetry: latency, confidence, prediction value, model version. Computes rolling accuracy from feedback submissions.

### 6. Experiment Tracker (`src/monitoring/experiment_tracker.py`)
MLflow integration for logging hyperparameters, metrics, and artifacts per training run. Falls back to local logging if MLflow is unreachable.

## Data Flow

```
Incoming Request
      ↓
FastAPI (/predict)
      ↓
Model Registry → get active model
      ↓
Preprocess features
      ↓
Model.predict() → prediction + confidence
      ↓
Return response to client
      ↓ (background)
MetricsCollector.record()
DriftDetector.check()
      ↓ (if drift threshold hit)
RetrainingPipeline.run()
      ↓
ModelRegistry.register() + promote()
```

## Deployment Topology

```
Internet → Load Balancer → FastAPI (2-10 pods, HPA)
                                ↓
                         Redis (feature cache)
                         MLflow (experiment DB)
                         Model Artifacts (PVC / S3)
                                ↓
Prometheus ← /metrics scrape
Grafana ← Prometheus datasource
Alerts → Slack / PagerDuty
```
