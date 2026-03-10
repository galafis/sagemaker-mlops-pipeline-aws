"""Model evaluation with quality gate enforcement for SageMaker pipelines."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QualityGate:
    """Quality gate thresholds for model promotion decisions."""

    min_accuracy: float = 0.80
    min_precision: float = 0.75
    min_recall: float = 0.75
    min_f1: float = 0.78
    min_auc: float = 0.80
    max_model_size_mb: float = 200.0


@dataclass
class EvaluationResult:
    """Complete evaluation result with gate decision."""

    metrics: Dict[str, float]
    confusion_matrix: List[List[int]]
    classification_report: str
    passed_quality_gate: bool
    gate_details: Dict[str, Dict[str, float]]
    recommendation: str


class ModelEvaluator:
    """Evaluates trained models and enforces quality gates.

    Designed to run as a SageMaker Processing job step in the
    MLOps pipeline, producing evaluation artifacts.
    """

    def __init__(self, quality_gate: Optional[QualityGate] = None):
        self.quality_gate = quality_gate or QualityGate()

    def evaluate(
        self,
        model_path: str,
        test_data_path: str,
        output_dir: Optional[str] = None,
    ) -> EvaluationResult:
        """Evaluate a model on test data.

        Args:
            model_path: Path to serialized model (pkl).
            test_data_path: Path to test CSV (target first column).
            output_dir: Optional directory for evaluation artifacts.

        Returns:
            EvaluationResult with metrics and gate decision.
        """
        import pickle

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        test_df = pd.read_csv(test_data_path, header=None)
        y_test = test_df.iloc[:, 0]
        X_test = test_df.iloc[:, 1:]

        predictions = model.predict(X_test)

        metrics: Dict[str, float] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, predictions, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
        }

        if hasattr(model, "predict_proba") and len(np.unique(y_test)) == 2:
            probas = model.predict_proba(X_test)[:, 1]
            metrics["auc"] = float(roc_auc_score(y_test, probas))

        model_size = Path(model_path).stat().st_size / (1024 * 1024)
        metrics["model_size_mb"] = round(model_size, 2)

        cm = confusion_matrix(y_test, predictions).tolist()
        report = classification_report(y_test, predictions, zero_division=0)

        gate_details, passed = self._check_quality_gate(metrics)

        recommendation = (
            "APPROVE - Model meets all quality gate thresholds. "
            "Safe to deploy to production."
            if passed
            else "REJECT - Model does not meet minimum quality thresholds. "
            "Review training data and hyperparameters."
        )

        result = EvaluationResult(
            metrics=metrics,
            confusion_matrix=cm,
            classification_report=report,
            passed_quality_gate=passed,
            gate_details=gate_details,
            recommendation=recommendation,
        )

        if output_dir:
            self._save_artifacts(result, output_dir)

        status = "PASSED" if passed else "FAILED"
        logger.info(f"Evaluation {status}: {metrics}")

        return result

    def _check_quality_gate(
        self, metrics: Dict[str, float]
    ) -> tuple:
        gate = self.quality_gate
        checks = {
            "accuracy": {"value": metrics.get("accuracy", 0), "threshold": gate.min_accuracy},
            "precision": {"value": metrics.get("precision", 0), "threshold": gate.min_precision},
            "recall": {"value": metrics.get("recall", 0), "threshold": gate.min_recall},
            "f1": {"value": metrics.get("f1", 0), "threshold": gate.min_f1},
        }

        if "auc" in metrics:
            checks["auc"] = {"value": metrics["auc"], "threshold": gate.min_auc}

        if "model_size_mb" in metrics:
            checks["model_size_mb"] = {
                "value": metrics["model_size_mb"],
                "threshold": gate.max_model_size_mb,
            }

        passed = True
        for name, check in checks.items():
            if name == "model_size_mb":
                check["passed"] = check["value"] <= check["threshold"]
            else:
                check["passed"] = check["value"] >= check["threshold"]

            if not check["passed"]:
                passed = False
                logger.warning(
                    f"Quality gate FAILED for {name}: "
                    f"{check['value']:.4f} vs threshold {check['threshold']:.4f}"
                )

        return checks, passed

    def _save_artifacts(
        self, result: EvaluationResult, output_dir: str
    ) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        with open(out / "evaluation_metrics.json", "w") as f:
            json.dump(result.metrics, f, indent=2)

        with open(out / "quality_gate.json", "w") as f:
            json.dump(
                {
                    "passed": result.passed_quality_gate,
                    "details": result.gate_details,
                    "recommendation": result.recommendation,
                },
                f,
                indent=2,
            )

        with open(out / "classification_report.txt", "w") as f:
            f.write(result.classification_report)

        logger.info(f"Evaluation artifacts saved to {output_dir}")
