# 🤖 MLOps Prediction Platform

> Production-grade AI prediction system with automated retraining, drift monitoring, CI/CD, and cloud deployment.

[![CI/CD](https://github.com/yourorg/mlops-platform/actions/workflows/mlops_pipeline.yml/badge.svg)](https://github.com/yourorg/mlops-platform/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Architecture Overview](#-architecture-overview)
- [Features](#-features)
- [Quick Start](#-quick-start)
- [API Reference](#-api-reference)
- [MLOps Pipeline](#-mlops-pipeline)
- [Monitoring](#-monitoring)
- [CI/CD](#-cicd)
- [Deployment](#-deployment)
- [Configuration](#-configuration)

---

## 🏗️ Architecture Overview

<img width="1408" height="768" alt="Gemini_Generated_Image_j091euj091euj091" src="https://github.com/user-attachments/assets/011f7bb3-2972-4f8b-9c9f-d074a5f68c17" />


---

## ✨ Features

| Feature | Implementation |
|---------|---------------|
| **Real-time Predictions** | FastAPI with async endpoints, <50ms P95 latency |
| **Automated Retraining** | Triggered by drift, schedule, or performance drop |
| **Experiment Tracking** | MLflow with parameter and metric logging |
| **Data Drift Detection** | KS test + PSI + Jensen-Shannon divergence |
| **Model Registry** | Versioned artifacts with staging/production promotion |
| **CI/CD Pipeline** | GitHub Actions: test → build → staging → production |
| **Containerization** | Multi-stage Docker build, Docker Compose stack |
| **Monitoring Dashboard** | Prometheus + Grafana with custom ML metrics |
| **Hyperparameter Optimization** | Optuna with multi-algorithm search |
| **Batch Predictions** | Bulk endpoint supporting up to 1000 records |
| **Feedback Loop** | Ground truth ingestion for continuous evaluation |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Git

### Local Development

```bash
# Clone the repository
git clone https://github.com/yourorg/mlops-platform.git
cd mlops-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the API server
uvicorn src.api.main:app --reload --port 8000

# API is now available at http://localhost:8000
# Docs: http://localhost:8000/docs
```

### Docker Compose (Full Stack)

```bash
# Start all services
docker compose -f docker/docker-compose.yml up -d

# Services started:
# API:        http://localhost:8000
# MLflow:     http://localhost:5000
# Grafana:    http://localhost:3000  (admin / mlops_admin)
# Prometheus: http://localhost:9090
```

---

## 📡 API Reference

### Make a Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "age_normalized": 0.5,
      "income_normalized": 0.7,
      "credit_score_normalized": 0.8,
      "loan_amount_normalized": -0.3,
      "employment_years": 1.2,
      "debt_ratio": -0.5
    },
    "return_probabilities": true
  }'
```

**Response:**
```json
{
  "prediction_id": "8f4e2a1c-...",
  "prediction": 1,
  "confidence": 0.87,
  "model_version": "1.0.0",
  "latency_ms": 12.4,
  "timestamp": "2025-01-15T10:30:00Z",
  "probabilities": {"0": 0.13, "1": 0.87}
}
```

### Batch Predictions

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"records": [{"age_normalized": 0.5, ...}, ...]}'
```

### Submit Feedback

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"prediction_id": "8f4e2a1c-...", "actual_label": 1}'
```

### Check Drift Status

```bash
curl http://localhost:8000/drift
```

### Trigger Manual Retraining

```bash
curl -X POST http://localhost:8000/retrain
```

### Full API Docs
Visit `http://localhost:8000/docs` for interactive Swagger UI.

---

## 🔄 MLOps Pipeline

### Retraining Triggers

The platform automatically retrains when:

1. **Data Drift** — 3 consecutive drift detections across features
2. **Scheduled** — Every Sunday at 2am UTC (GitHub Actions cron)
3. **Manual** — `POST /retrain` endpoint
4. **Performance Drop** — Accuracy drops below threshold (from feedback loop)

### Pipeline Steps

```
Trigger Detected
      │
      ▼
1. Data Collection      ← Query feature store / data warehouse
      │
      ▼
2. Data Validation      ← Check quality, sample count, class balance
      │
      ▼
3. Feature Engineering  ← Interaction features, polynomial transforms
      │
      ▼
4. Hyperparameter Opt   ← Optuna (20 trials, multi-algorithm)
      │
      ▼
5. Model Training       ← Train on full dataset with best params
      │
      ▼
6. Model Evaluation     ← Accuracy, F1, ROC-AUC on holdout set
      │
      ▼
7. Auto-Promotion       ← Promote if ≥0.5% better than production
      │
      ▼
8. Registry Update      ← Version, artifact, and metadata stored
```

### Drift Detection Methods

| Method | Detects | Threshold |
|--------|---------|-----------|
| Kolmogorov-Smirnov test | Distribution shape changes | p < 0.05 |
| Population Stability Index | Feature distribution shift | PSI > 0.2 |
| Jensen-Shannon divergence | Probability distribution drift | JS > 0.1 |

---

## 📊 Monitoring

### Grafana Dashboard

Navigate to `http://localhost:3000` (admin/mlops_admin) to access:

- **Model Performance**: Accuracy, F1, AUC over time
- **API Health**: Request rate, latency percentiles, error rate
- **Drift Monitoring**: Feature-level drift scores with alert history
- **Business Metrics**: Prediction confidence distribution
- **Infrastructure**: CPU, memory, container health

### Key Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `mlops_latency_p95` | 95th percentile prediction latency | > 200ms |
| `mlops_drift_score` | PSI drift score per feature | > 0.25 |
| `mlops_accuracy` | Rolling model accuracy | < 0.70 |
| `mlops_error_rate` | API error rate | > 1% |
| `mlops_low_confidence_rate` | % predictions below 60% confidence | > 20% |

---

## 🔧 CI/CD

### Pipeline Jobs

```
Push to main
    │
    ├── test              Unit tests, coverage, linting
    │
    ├── model-validation  Quality gate (accuracy ≥ 0.75)
    │
    ├── security          Bandit scan, dependency audit
    │
    ├── build             Docker image → GitHub Container Registry
    │
    ├── deploy-staging    Auto-deploy to staging environment
    │
    └── deploy-production  Manual approval → production deployment
```

### Environment Secrets Required

```bash
PRODUCTION_API_URL    # Your production API URL
API_KEY               # API authentication key
```

---



### Kubernetes (Recommended)

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/namespace.yml
kubectl apply -f k8s/deployment.yml
kubectl apply -f k8s/service.yml
kubectl apply -f k8s/hpa.yml  # Horizontal Pod Autoscaler

# Verify deployment
kubectl get pods -n mlops
kubectl get svc -n mlops
```

### AWS (ECS + ECR)

```bash
# Push to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_URI
docker tag mlops-api:latest $ECR_URI/mlops-api:latest
docker push $ECR_URI/mlops-api:latest

# Deploy to ECS
aws ecs update-service --cluster mlops --service api --force-new-deployment
```

### GCP 

```bash
gcloud run deploy mlops-api \
  --image gcr.io/PROJECT/mlops-api:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 1 \
  --max-instances 10
```

---

## ⚙️ Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Application log level |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | MLflow server URL |
| `REDIS_URL` | `redis://redis:6379` | Redis connection string |
| `DRIFT_WINDOW_SIZE` | `500` | Predictions in drift window |
| `DRIFT_CHECK_INTERVAL` | `50` | Check drift every N predictions |
| `RETRAINING_DRIFT_THRESHOLD` | `3` | Consecutive drift alerts to retrain |
| `MIN_ACCURACY_TO_PROMOTE` | `0.005` | Minimum improvement to auto-promote |

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v --cov=src --cov-report=html

# Run specific test classes
pytest tests/test_platform.py::TestDriftDetector -v
pytest tests/test_platform.py::TestRetrainingTrigger -v

# Run with coverage report
pytest --cov=src --cov-report=term-missing
```

---
##Project Screenshots
```
<img width="1468" height="878" alt="Screenshot 2026-05-27 at 07 30 21" src="https://github.com/user-attachments/assets/7da67824-6f39-4607-acc7-a8cffc8f050b" />
<img width="1441" height="683" alt="Screenshot 2026-05-27 at 07 31 24" src="https://github.com/user-attachments/assets/cfa531b8-f587-4f01-a984-86f2cdcdda08" />
<img width="1467" height="880" alt="Screenshot 2026-05-27 at 07 31 31" src="https://github.com/user-attachments/assets/b6aea7cd-df6f-41bf-80bc-2518d18bad64" />
<img width="1465" height="881" alt="Screenshot 2026-05-27 at 07 31 41" src="https://github.com/user-attachments/assets/a4bba50d-4e91-4822-9d67-d507bc70de40" />
<img width="1467" height="877" alt="Screenshot 2026-05-27 at 07 31 47" src="https://github.com/user-attachments/assets/52c58ab1-3222-4262-a59a-2b1f7aa32034" />
<img width="1469" height="879" alt="Screenshot 2026-05-27 at 07 31 50" src="https://github.com/user-attachments/assets/17d92f04-f458-4e27-a0b8-fea8e1eaf4a9" />
<img width="1467" height="881" alt="Screenshot 2026-05-27 at 07 32 01" src="https://github.com/user-attachments/assets/1ecd05e0-68bd-4a54-bc35-f5e91682e9bd" />
<img width="1466" height="879" alt="Screenshot 2026-05-27 at 07 32 06" src="https://github.com/user-attachments/assets/fccd3eb3-bfaf-4264-8098-3d044288b3d9" />
<img width="1468" height="879" alt="Screenshot 2026-05-27 at 07 32 23" src="https://github.com/user-attachments/assets/e2197dce-1101-4f1c-ade3-63c82035a809" />
<img width="1470" height="956" alt="Screenshot 2026-05-27 at 07 29 36" src="https://github.com/user-attachments/assets/9673d2f6-c4ea-4420-9e2f-6f85c5848c71" />

```


## 📁 Project Structure

```
mlops-platform/
├── src/
│   ├── api/
│   │   └── main.py              # FastAPI application
│   ├── models/
│   │   └── model_registry.py    # Model versioning & registry
│   ├── pipeline/
│   │   └── retraining_trigger.py # Automated retraining
│   ├── monitoring/
│   │   ├── drift_detector.py    # Statistical drift detection
│   │   └── metrics_collector.py # Metrics aggregation
│   └── utils/
│       └── logger.py            # Structured logging
├── tests/
│   └── test_platform.py         # Comprehensive test suite
├── docker/
│   ├── Dockerfile               # Multi-stage build
│   └── docker-compose.yml       # Full stack orchestration
├── .github/
│   └── workflows/
│       └── mlops_pipeline.yml   # CI/CD pipeline
├── scripts/
│   └── scheduled_retraining.py  # Retraining worker
├── configs/
│   └── prometheus.yml           # Monitoring config
├── requirements.txt
└── README.md
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built as a production-grade MLOps internship project demonstrating enterprise AI system design.*
