"""Unit tests for the model evaluation module."""

import json
import os

import pytest

from src.evaluation.model_evaluator import (
    EvaluationResult,
    ModelEvaluator,
    QualityGate,
)


class TestQualityGate:
    """Tests for QualityGate defaults."""

    def test_default_thresholds(self):
        gate = QualityGate()
        assert gate.min_accuracy == 0.80
        assert gate.min_precision == 0.75
        assert gate.min_recall == 0.75
        assert gate.min_f1 == 0.78
        assert gate.min_auc == 0.80
        assert gate.max_model_size_mb == 200.0

    def test_custom_thresholds(self):
        gate = QualityGate(min_accuracy=0.90, min_f1=0.85)
        assert gate.min_accuracy == 0.90
        assert gate.min_f1 == 0.85


class TestModelEvaluator:
    """Tests for ModelEvaluator functionality."""

    def test_evaluate_returns_result(
        self, model_path, sample_csv_path
    ):
        evaluator = ModelEvaluator()
        result = evaluator.evaluate(model_path, sample_csv_path)

        assert isinstance(result, EvaluationResult)
        assert "accuracy" in result.metrics
        assert "precision" in result.metrics
        assert "recall" in result.metrics
        assert "f1" in result.metrics

    def test_evaluate_metrics_ranges(
        self, model_path, sample_csv_path
    ):
        evaluator = ModelEvaluator()
        result = evaluator.evaluate(model_path, sample_csv_path)

        for key in ["accuracy", "precision", "recall", "f1"]:
            assert 0.0 <= result.metrics[key] <= 1.0

    def test_evaluate_auc_present(
        self, model_path, sample_csv_path
    ):
        evaluator = ModelEvaluator()
        result = evaluator.evaluate(model_path, sample_csv_path)
        assert "auc" in result.metrics
        assert 0.0 <= result.metrics["auc"] <= 1.0

    def test_evaluate_model_size(
        self, model_path, sample_csv_path
    ):
        evaluator = ModelEvaluator()
        result = evaluator.evaluate(model_path, sample_csv_path)
        assert "model_size_mb" in result.metrics
        assert result.metrics["model_size_mb"] > 0

    def test_evaluate_confusion_matrix(
        self, model_path, sample_csv_path
    ):
        evaluator = ModelEvaluator()
        result = evaluator.evaluate(model_path, sample_csv_path)
        assert isinstance(result.confusion_matrix, list)
        assert len(result.confusion_matrix) == 2
        assert len(result.confusion_matrix[0]) == 2

    def test_quality_gate_pass(
        self, model_path, sample_csv_path
    ):
        gate = QualityGate(
            min_accuracy=0.5,
            min_precision=0.5,
            min_recall=0.5,
            min_f1=0.5,
            min_auc=0.5,
        )
        evaluator = ModelEvaluator(quality_gate=gate)
        result = evaluator.evaluate(model_path, sample_csv_path)

        assert result.passed_quality_gate is True
        assert "APPROVE" in result.recommendation

    def test_quality_gate_fail(
        self, model_path, sample_csv_path
    ):
        gate = QualityGate(
            min_accuracy=0.99,
            min_precision=0.99,
            min_recall=0.99,
            min_f1=0.99,
        )
        evaluator = ModelEvaluator(quality_gate=gate)
        result = evaluator.evaluate(model_path, sample_csv_path)

        assert result.passed_quality_gate is False
        assert "REJECT" in result.recommendation

    def test_gate_details(self, model_path, sample_csv_path):
        evaluator = ModelEvaluator()
        result = evaluator.evaluate(model_path, sample_csv_path)

        assert "accuracy" in result.gate_details
        assert "value" in result.gate_details["accuracy"]
        assert "threshold" in result.gate_details["accuracy"]
        assert "passed" in result.gate_details["accuracy"]

    def test_save_artifacts(
        self, model_path, sample_csv_path, temp_dir
    ):
        evaluator = ModelEvaluator()
        evaluator.evaluate(
            model_path, sample_csv_path, output_dir=temp_dir
        )

        assert os.path.exists(
            os.path.join(temp_dir, "evaluation_metrics.json")
        )
        assert os.path.exists(
            os.path.join(temp_dir, "quality_gate.json")
        )
        assert os.path.exists(
            os.path.join(temp_dir, "classification_report.txt")
        )

        with open(
            os.path.join(temp_dir, "quality_gate.json")
        ) as f:
            gate_data = json.load(f)
            assert "passed" in gate_data
            assert "details" in gate_data
            assert "recommendation" in gate_data
