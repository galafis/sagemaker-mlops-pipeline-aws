"""Unit tests for the SageMaker training module."""

import os
import pickle

import numpy as np
import pytest

from src.training.train import SageMakerTrainer
from src.training.hyperparameters import (
    HyperparameterConfig,
    HyperparameterRange,
)


class TestSageMakerTrainer:
    """Tests for SageMakerTrainer functionality."""

    def test_initialization(self):
        trainer = SageMakerTrainer(algorithm="gradient_boosting")
        assert trainer.algorithm == "gradient_boosting"

    def test_train_gradient_boosting(self, sample_data, temp_dir):
        X, y = sample_data
        trainer = SageMakerTrainer(algorithm="gradient_boosting")
        result = trainer.train(X, y, output_dir=temp_dir)

        assert result["model"] is not None
        assert "accuracy" in result["metrics"]
        assert "f1" in result["metrics"]
        assert result["metrics"]["accuracy"] > 0.5

    def test_train_random_forest(self, sample_data, temp_dir):
        X, y = sample_data
        trainer = SageMakerTrainer(algorithm="random_forest")
        result = trainer.train(X, y, output_dir=temp_dir)

        assert result["model"] is not None
        assert result["metrics"]["accuracy"] > 0.5

    def test_train_saves_artifacts(self, sample_data, temp_dir):
        X, y = sample_data
        trainer = SageMakerTrainer(algorithm="gradient_boosting")
        trainer.train(X, y, output_dir=temp_dir)

        assert os.path.exists(os.path.join(temp_dir, "model.pkl"))
        assert os.path.exists(os.path.join(temp_dir, "metrics.json"))

    def test_model_serialization(self, sample_data, temp_dir):
        X, y = sample_data
        trainer = SageMakerTrainer(algorithm="gradient_boosting")
        result = trainer.train(X, y, output_dir=temp_dir)

        model_path = os.path.join(temp_dir, "model.pkl")
        with open(model_path, "rb") as f:
            loaded_model = pickle.load(f)

        predictions = loaded_model.predict(X[:5])
        assert len(predictions) == 5

    def test_invalid_algorithm(self):
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            trainer = SageMakerTrainer(algorithm="invalid_algo")
            trainer.train(np.array([[1]]), np.array([0]))


class TestHyperparameters:
    """Tests for hyperparameter configuration."""

    def test_range_creation(self):
        hp_range = HyperparameterRange(
            name="learning_rate",
            min_value=0.001,
            max_value=0.1,
            scaling_type="Logarithmic",
        )
        assert hp_range.name == "learning_rate"
        assert hp_range.min_value == 0.001

    def test_range_sagemaker_format(self):
        hp_range = HyperparameterRange(
            name="n_estimators",
            min_value=50,
            max_value=500,
            scaling_type="Linear",
            parameter_type="Integer",
        )
        result = hp_range.to_sagemaker_format()
        assert result["Name"] == "n_estimators"
        assert result["MinValue"] == "50"
        assert result["MaxValue"] == "500"

    def test_config_defaults(self):
        config = HyperparameterConfig()
        default_params = config.get_default_params("gradient_boosting")
        assert "n_estimators" in default_params
        assert "learning_rate" in default_params
        assert "max_depth" in default_params

    def test_config_search_spaces(self):
        config = HyperparameterConfig()
        spaces = config.get_search_space("xgboost")
        assert len(spaces) > 0
        assert all(
            isinstance(s, HyperparameterRange) for s in spaces
        )
