"""Shared test fixtures for the SageMaker MLOps pipeline."""

import json
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_data():
    """Generate a synthetic classification dataset."""
    np.random.seed(42)
    n_samples = 200
    n_features = 8

    X = np.random.randn(n_samples, n_features)
    weights = np.random.randn(n_features)
    logits = X @ weights + np.random.randn(n_samples) * 0.3
    y = (logits > 0).astype(int)

    return X, y


@pytest.fixture
def sample_csv_path(temp_dir, sample_data):
    """Save sample data as CSV (target in first column)."""
    X, y = sample_data
    df = pd.DataFrame(
        np.column_stack([y, X]),
    )
    csv_path = os.path.join(temp_dir, "test_data.csv")
    df.to_csv(csv_path, index=False, header=False)
    return csv_path


@pytest.fixture
def trained_model(sample_data):
    """Train a simple model for testing."""
    X, y = sample_data
    model = GradientBoostingClassifier(
        n_estimators=20, max_depth=3, random_state=42
    )
    model.fit(X, y)
    return model


@pytest.fixture
def model_path(temp_dir, trained_model):
    """Save a trained model to disk."""
    path = os.path.join(temp_dir, "model.pkl")
    with open(path, "wb") as f:
        pickle.dump(trained_model, f)
    return path


@pytest.fixture
def model_dir(temp_dir, trained_model):
    """Create a model directory with model and metadata."""
    model_path = os.path.join(temp_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(trained_model, f)

    metrics = {
        "accuracy": 0.92,
        "precision": 0.91,
        "recall": 0.90,
        "f1": 0.905,
        "auc": 0.95,
    }
    metrics_path = os.path.join(temp_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)

    return temp_dir


@pytest.fixture
def pipeline_settings_dict():
    """Default settings for pipeline configuration."""
    return {
        "sagemaker": {
            "region": "us-east-1",
            "role_arn": "arn:aws:iam::123456789012:role/SageMakerRole",
            "default_bucket": "test-ml-bucket",
            "pipeline_name": "test-pipeline",
            "model_package_group": "test-model-group",
        },
        "training": {
            "instance_type": "ml.m5.xlarge",
            "instance_count": 1,
            "use_spot_instances": True,
            "max_runtime_seconds": 3600,
            "max_wait_seconds": 7200,
        },
        "endpoint": {
            "endpoint_name": "test-endpoint",
            "instance_type": "ml.m5.large",
            "initial_instance_count": 1,
        },
    }
