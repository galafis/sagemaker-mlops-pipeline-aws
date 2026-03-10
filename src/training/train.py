"""Model training module for SageMaker training jobs.

Supports XGBoost, LightGBM, and scikit-learn models with
hyperparameter management and artifact serialization.
"""

import json
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_REGISTRY = {
    "gradient_boosting": GradientBoostingClassifier,
    "random_forest": RandomForestClassifier,
}


@dataclass
class TrainingResult:
    """Training job output metadata."""

    model_path: str
    algorithm: str
    metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    training_time_seconds: float
    num_features: int
    num_train_samples: int


class SageMakerTrainer:
    """Model trainer designed for SageMaker training job compatibility.

    Reads data from SageMaker channel paths, trains the model,
    evaluates on validation set, and saves artifacts.
    """

    def __init__(
        self,
        algorithm: str = "gradient_boosting",
        hyperparameters: Optional[Dict[str, Any]] = None,
        output_dir: str = "./model_output",
    ):
        if algorithm not in MODEL_REGISTRY:
            raise ValueError(
                f"Algorithm '{algorithm}' not supported. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )
        self.algorithm = algorithm
        self.hyperparameters = hyperparameters or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._model = None

    def train(
        self,
        train_path: str,
        validation_path: Optional[str] = None,
    ) -> TrainingResult:
        """Train a model from CSV data files.

        Expects CSV with target as the first column and no header.

        Args:
            train_path: Path to training data CSV.
            validation_path: Optional path to validation data CSV.

        Returns:
            TrainingResult with model path and metrics.
        """
        logger.info(f"Training {self.algorithm} from {train_path}")
        start_time = time.time()

        train_df = pd.read_csv(train_path, header=None)
        y_train = train_df.iloc[:, 0]
        X_train = train_df.iloc[:, 1:]

        default_params = {"random_state": 42, "n_estimators": 200}
        default_params.update(self.hyperparameters)

        model_class = MODEL_REGISTRY[self.algorithm]
        valid_params = model_class().get_params()
        filtered = {k: v for k, v in default_params.items() if k in valid_params}

        self._model = model_class(**filtered)
        self._model.fit(X_train, y_train)

        training_time = time.time() - start_time

        metrics = self._evaluate(X_train, y_train, prefix="train")

        if validation_path and Path(validation_path).exists():
            val_df = pd.read_csv(validation_path, header=None)
            y_val = val_df.iloc[:, 0]
            X_val = val_df.iloc[:, 1:]
            val_metrics = self._evaluate(X_val, y_val, prefix="validation")
            metrics.update(val_metrics)

        model_path = self._save_model()

        result = TrainingResult(
            model_path=model_path,
            algorithm=self.algorithm,
            metrics=metrics,
            hyperparameters=filtered,
            training_time_seconds=round(training_time, 2),
            num_features=X_train.shape[1],
            num_train_samples=len(X_train),
        )

        metrics_path = self.output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(result.metrics, f, indent=2)

        logger.info(
            f"Training complete in {training_time:.1f}s: "
            f"train_f1={metrics.get('train_f1', 0):.4f}"
        )

        return result

    def _evaluate(
        self, X: pd.DataFrame, y: pd.Series, prefix: str = ""
    ) -> Dict[str, float]:
        predictions = self._model.predict(X)
        probas = None
        if hasattr(self._model, "predict_proba"):
            probas = self._model.predict_proba(X)

        metrics = {
            f"{prefix}_accuracy": float(accuracy_score(y, predictions)),
            f"{prefix}_precision": float(
                precision_score(y, predictions, average="weighted", zero_division=0)
            ),
            f"{prefix}_recall": float(
                recall_score(y, predictions, average="weighted", zero_division=0)
            ),
            f"{prefix}_f1": float(
                f1_score(y, predictions, average="weighted", zero_division=0)
            ),
        }

        if probas is not None and len(np.unique(y)) == 2:
            metrics[f"{prefix}_auc"] = float(roc_auc_score(y, probas[:, 1]))

        return metrics

    def _save_model(self) -> str:
        model_path = str(self.output_dir / "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(self._model, f, protocol=pickle.HIGHEST_PROTOCOL)

        size_mb = Path(model_path).stat().st_size / (1024 * 1024)
        logger.info(f"Model saved: {model_path} ({size_mb:.2f} MB)")
        return model_path


if __name__ == "__main__":
    """Entry point for SageMaker Training job."""
    train_channel = os.environ.get(
        "SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"
    )
    val_channel = os.environ.get(
        "SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation"
    )
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    hp_str = os.environ.get("SM_HPS", "{}")

    hyperparameters = json.loads(hp_str) if hp_str else {}

    train_file = next(Path(train_channel).glob("*.csv"), None)
    val_file = next(Path(val_channel).glob("*.csv"), None)

    if train_file:
        trainer = SageMakerTrainer(
            hyperparameters=hyperparameters, output_dir=model_dir
        )
        trainer.train(
            str(train_file),
            str(val_file) if val_file else None,
        )
