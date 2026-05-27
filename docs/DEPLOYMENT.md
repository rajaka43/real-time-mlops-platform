# Deployment Guide

## Local Development

```bash
# 1. Clone and install
git clone https://github.com/yourname/mlops-platform.git
cd mlops-platform
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Copy environment file
cp .env.example .env

# 3. Start the API
make run
# or: uvicorn src.api.main:app --reload --port 8000

# 4. Test it
make smoke
```

## Docker (Full Stack)

```bash
# Start everything
make docker-up

# Services:
#   API        → http://localhost:8000
#   API Docs   → http://localhost:8000/docs
#   MLflow     → http://localhost:5000
#   Grafana    → http://localhost:3000  (admin / mlops_admin)
#   Prometheus → http://localhost:9090

# Stop everything
make docker-down
```

## Kubernetes

```bash
# Prerequisites: kubectl configured, image pushed to registry

# 1. Update image in k8s/deployment.yml with your registry URL

# 2. Create secrets
kubectl create secret generic mlops-secrets \
  --from-literal=api-key=your-secret-key \
  -n mlops

# 3. Apply manifests
kubectl apply -f k8s/deployment.yml

# 4. Check status
kubectl get pods -n mlops
kubectl get svc -n mlops

# 5. Get external IP
kubectl get svc mlops-api-svc -n mlops
```

## AWS ECS

```bash
# 1. Authenticate to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin $AWS_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com

# 2. Build and push
docker build -f docker/Dockerfile -t mlops-api .
docker tag mlops-api:latest $AWS_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/mlops-api:latest
docker push $AWS_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/mlops-api:latest

# 3. Update ECS service
aws ecs update-service --cluster mlops --service api --force-new-deployment
```

## Google Cloud Run

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT_ID/mlops-api

# Deploy
gcloud run deploy mlops-api \
  --image gcr.io/PROJECT_ID/mlops-api \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 1 \
  --max-instances 10 \
  --allow-unauthenticated
```

## GitHub Actions Secrets Required

Go to your GitHub repo → Settings → Secrets → Actions, and add:

| Secret | Value |
|--------|-------|
| `PRODUCTION_API_URL` | Your deployed API URL |
| `API_KEY` | Your API authentication key |

The CI/CD pipeline runs automatically on every push to `main`.
