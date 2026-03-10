"""Unit tests for the SageMaker preprocessing module."""

import os

import numpy as np
import pandas as pd
import pytest

from src.processing.preprocessing import SageMakerPreprocessor


class TestSageMakerPreprocessor:
    """Tests for SageMakerPreprocessor functionality."""

    def test_initialization_defaults(self):
        preprocessor = SageMakerPreprocessor()
        assert preprocessor.test_size == 0.15
        assert preprocessor.validation_size == 0.15

    def test_initialization_custom(self):
        preprocessor = SageMakerPreprocessor(
            test_size=0.2, validation_size=0.1
        )
        assert preprocessor.test_size == 0.2
        assert preprocessor.validation_size == 0.1

    def test_split_data(self, sample_data):
        X, y = sample_data
        preprocessor = SageMakerPreprocessor(
            test_size=0.2, validation_size=0.1
        )
        splits = preprocessor.split_data(X, y)

        assert "X_train" in splits
        assert "X_val" in splits
        assert "X_test" in splits
        assert "y_train" in splits
        assert "y_val" in splits
        assert "y_test" in splits

        total = (
            len(splits["X_train"])
            + len(splits["X_val"])
            + len(splits["X_test"])
        )
        assert total == len(X)

    def test_split_proportions(self, sample_data):
        X, y = sample_data
        preprocessor = SageMakerPreprocessor(
            test_size=0.2, validation_size=0.2
        )
        splits = preprocessor.split_data(X, y)

        n = len(X)
        assert len(splits["X_test"]) == int(n * 0.2)
        assert len(splits["X_val"]) == int(n * 0.2)

    def test_engineer_features(self, sample_data):
        X, _ = sample_data
        preprocessor = SageMakerPreprocessor()
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

        result = preprocessor.engineer_features(df)
        assert result.shape[1] >= df.shape[1]
        assert not result.isnull().any().any()

    def test_save_splits(self, temp_dir, sample_data):
        X, y = sample_data
        preprocessor = SageMakerPreprocessor()
        splits = preprocessor.split_data(X, y)

        preprocessor.save_splits(splits, temp_dir)

        assert os.path.exists(os.path.join(temp_dir, "train.csv"))
        assert os.path.exists(os.path.join(temp_dir, "validation.csv"))
        assert os.path.exists(os.path.join(temp_dir, "test.csv"))
        assert os.path.exists(os.path.join(temp_dir, "manifest.json"))

        train_df = pd.read_csv(
            os.path.join(temp_dir, "train.csv"), header=None
        )
        assert train_df.shape[0] == len(splits["X_train"])
