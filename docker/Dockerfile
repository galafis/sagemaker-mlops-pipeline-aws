FROM python:3.11-slim AS base

LABEL maintainer="Gabriel Demetrios Lafis"
LABEL description="SageMaker MLOps Pipeline - Processing and Training Container"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /opt/ml/code

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY src/ ./src/
COPY config/ ./config/

# SageMaker expects the training script at a specific location
ENV SAGEMAKER_PROGRAM=src/training/train.py

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import src; print('healthy')" || exit 1

ENTRYPOINT ["python"]
CMD ["-m", "src.training.train"]
