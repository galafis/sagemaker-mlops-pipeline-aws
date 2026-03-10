"""Integration tests for the end-to-end SageMaker pipeline workflow."""

import json
import os
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier

from src.evaluation.model_evaluator import ModelEvaluator, QualityGate
from src.inference.inference_handler import InferenceHandler
from src.inference.serializer import deserialize_response, serialize_request
from src.processing.preprocessing import SageMakerPreprocessor
from src.training.train import SageMakerTrainer


class TestEndToEndWorkflow:
    """Integration test simulating the complete ML pipeline locally."""

    @pytest.fixture
    def raw_dataset(self, temp_dir):
        """Create a realistic raw dataset."""
        np.random.seed(42)
        n_samples = 300
        n_features = 10

        X = np.random.randn(n_samples, n_features)
        noise = np.random.randn(n_samples) * 0.2
        weights = np.random.randn(n_features)
        y = (X @ weights + noise > 0).astype(int)

        df = pd.DataFrame(
            X, columns=[f"feature_{i}" for i in range(n_features)]
        )
        df.insert(0, "target", y)

        raw_path = os.path.join(temp_dir, "raw_data.csv")
        df.to_csv(raw_path, index=False)
        return raw_path

    def test_full_pipeline(self, temp_dir, raw_dataset):
        """Run the complete pipeline: preprocess → train → eval → serve."""
        processed_dir = os.path.join(temp_dir, "processed")
        model_dir = os.path.join(temp_dir, "model")
        eval_dir = os.path.join(temp_dir, "evaluation")

        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)

        # --- Step 1: Preprocessing ---
        raw_df = pd.read_csv(raw_dataset)
        y = raw_df["target"].values
        X = raw_df.drop(columns=["target"]).values

        preprocessor = SageMakerPreprocessor(
            test_size=0.2, validation_size=0.1
        )
        splits = preprocessor.split_data(X, y)
        preprocessor.save_splits(splits, processed_dir)

        train_path = os.path.join(processed_dir, "train.csv")
        val_path = os.path.join(processed_dir, "validation.csv")
        test_path = os.path.join(processed_dir, "test.csv")

        assert os.path.exists(train_path)
        assert os.path.exists(val_path)
        assert os.path.exists(test_path)

        # --- Step 2: Training ---
        trainer = SageMakerTrainer(algorithm="gradient_boosting")
        result = trainer.train(
            splits["X_train"],
            splits["y_train"],
            output_dir=model_dir,
        )

        assert result["model"] is not None
        assert result["metrics"]["accuracy"] > 0.5
        assert os.path.exists(
            os.path.join(model_dir, "model.pkl")
        )

        # --- Step 3: Evaluation ---
        model_path = os.path.join(model_dir, "model.pkl")
        gate = QualityGate(
            min_accuracy=0.5,
            min_precision=0.5,
            min_recall=0.5,
            min_f1=0.5,
            min_auc=0.5,
        )
        evaluator = ModelEvaluator(quality_gate=gate)
        eval_result = evaluator.evaluate(
            model_path, test_path, output_dir=eval_dir
        )

        assert eval_result.passed_quality_gate is True
        assert eval_result.metrics["accuracy"] > 0.5
        assert os.path.exists(
            os.path.join(eval_dir, "evaluation_metrics.json")
        )
        assert os.path.exists(
            os.path.join(eval_dir, "quality_gate.json")
        )

        # --- Step 4: Inference ---
        handler = InferenceHandler(model_dir=model_dir)
        handler.model_fn()

        sample = splits["X_test"][:5]
        request_body = serialize_request(
            sample.tolist(), "application/json"
        )
        parsed_input = handler.input_fn(
            request_body, "application/json"
        )
        predictions = handler.predict_fn(parsed_input)

        assert "predictions" in predictions
        assert len(predictions["predictions"]) == 5
        assert all(
            p in [0, 1] for p in predictions["predictions"]
        )

        output = handler.output_fn(predictions, "application/json")
        response = deserialize_response(output, "application/json")
        assert response["predictions"] == predictions["predictions"]

    def test_quality_gate_blocks_bad_model(self, temp_dir):
        """Verify that quality gate blocks a deliberately bad model."""
        np.random.seed(99)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        model = GradientBoostingClassifier(
            n_estimators=1, max_depth=1, random_state=42
        )
        model.fit(X, y)

        model_path = os.path.join(temp_dir, "bad_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        test_df = pd.DataFrame(np.column_stack([y, X]))
        test_path = os.path.join(temp_dir, "test.csv")
        test_df.to_csv(test_path, index=False, header=False)

        gate = QualityGate(
            min_accuracy=0.95,
            min_f1=0.95,
        )
        evaluator = ModelEvaluator(quality_gate=gate)
        result = evaluator.evaluate(model_path, test_path)

        assert result.passed_quality_gate is False
        assert "REJECT" in result.recommendation
