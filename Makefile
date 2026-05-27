# ─────────────────────────────────────────────────────────
# MLOps Platform — Makefile
# Usage: make <target>
# ─────────────────────────────────────────────────────────

.PHONY: install run test lint format docker-up docker-down retrain validate clean help

## Install all Python dependencies
install:
	pip install -r requirements.txt

## Run the API server locally (hot reload)
run:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

## Run all tests with coverage
test:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

## Lint code
lint:
	flake8 src/ tests/ --max-line-length=120
	mypy src/ --ignore-missing-imports

## Format code
format:
	black src/ tests/ scripts/

## Start full Docker stack (API + MLflow + Redis + Prometheus + Grafana)
docker-up:
	docker compose -f docker/docker-compose.yml up -d --build
	@echo ""
	@echo "  ✅  Stack is running:"
	@echo "  API:        http://localhost:8000"
	@echo "  API Docs:   http://localhost:8000/docs"
	@echo "  MLflow:     http://localhost:5000"
	@echo "  Grafana:    http://localhost:3000  (admin / mlops_admin)"
	@echo "  Prometheus: http://localhost:9090"

## Stop all Docker services
docker-down:
	docker compose -f docker/docker-compose.yml down

## View Docker logs
docker-logs:
	docker compose -f docker/docker-compose.yml logs -f api

## Trigger manual retraining
retrain:
	python scripts/trigger_retraining.py

## Run model validation quality gate
validate:
	python scripts/validate_model.py

## Check data drift
drift:
	python scripts/check_drift.py

## Run scheduled retraining worker
worker:
	python scripts/scheduled_retraining.py

## Quick API smoke test (requires running server)
smoke:
	@echo "Testing /health..."
	curl -s http://localhost:8000/health | python -m json.tool
	@echo "\nTesting /predict..."
	curl -s -X POST http://localhost:8000/predict \
	  -H "Content-Type: application/json" \
	  -d '{"features":{"age_normalized":0.5,"income_normalized":0.7,"credit_score_normalized":0.8,"loan_amount_normalized":-0.3,"employment_years":1.2,"debt_ratio":-0.5},"return_probabilities":true}' \
	  | python -m json.tool

## Clean up generated files
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .coverage htmlcov/ .pytest_cache/ dist/ build/ *.egg-info/

## Show this help
help:
	@echo "MLOps Platform — Available Commands:"
	@echo ""
	@grep -E '^## ' Makefile | sed 's/## /  /'
	@echo ""
	@echo "Usage: make <command>"
