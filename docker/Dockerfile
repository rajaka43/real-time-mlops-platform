# ─────────────────────────────────────────────
# Stage 1: Builder
# ─────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt


# ─────────────────────────────────────────────
# Stage 2: Production Image
# ─────────────────────────────────────────────
FROM python:3.11-slim

# Security: non-root user
RUN groupadd -r mlops && useradd -r -g mlops mlops

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /home/mlops/.local

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/

# Create necessary directories
RUN mkdir -p models/registry models/artifacts data/raw data/processed \
    && chown -R mlops:mlops /app

USER mlops

# Environment variables
ENV PATH=/home/mlops/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODEL_REGISTRY_PATH=/app/models/registry
ENV LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

EXPOSE 8000

# Start server
CMD ["python", "-m", "uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--log-level", "info"]
