"""Custom inference handler for SageMaker endpoints.

Implements the four handler functions required by SageMaker:
model_fn, input_fn, predict_fn, output_fn.
"""

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class InferenceHandler:
    """Handles model loading, input parsing, prediction, and output formatting.

    Compatible with SageMaker's inference contract while also
    usable as a standalone prediction service.
    """

    def __init__(self, model_dir: str = "./model_output"):
        self.model_dir = Path(model_dir)
        self._model = None
        self._metadata: Dict[str, Any] = {}

    def model_fn(self, model_dir: Optional[str] = None) -> Any:
        """Load the model from the artifact directory."""
        dir_path = Path(model_dir) if model_dir else self.model_dir

        model_path = dir_path / "model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        with open(model_path, "rb") as f:
            self._model = pickle.load(f)

        metadata_path = dir_path / "metrics.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self._metadata = json.load(f)

        logger.info(f"Model loaded from {model_path}")
        return self._model

    def input_fn(
        self,
        request_body: Union[str, bytes, Dict],
        content_type: str = "application/json",
    ) -> np.ndarray:
        """Parse incoming request into model-ready format.

        Supports JSON and CSV content types.
        """
        if content_type == "application/json":
            if isinstance(request_body, (str, bytes)):
                data = json.loads(request_body)
            else:
                data = request_body

            if "instances" in data:
                return np.array(data["instances"])
            elif "features" in data:
                return np.array(data["features"])
            elif isinstance(data, list):
                return np.array(data)
            else:
                raise ValueError(
                    "JSON must contain 'instances', 'features', or be a list"
                )

        elif content_type == "text/csv":
            if isinstance(request_body, bytes):
                request_body = request_body.decode("utf-8")
            df = pd.read_csv(
                pd.io.common.StringIO(request_body), header=None
            )
            return df.values

        else:
            raise ValueError(f"Unsupported content type: {content_type}")

    def predict_fn(
        self, input_data: np.ndarray, model: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Generate predictions from the model.

        Returns both class predictions and probabilities when available.
        """
        model = model or self._model
        if model is None:
            raise RuntimeError("Model not loaded. Call model_fn() first.")

        predictions = model.predict(input_data)
        result: Dict[str, Any] = {
            "predictions": predictions.tolist(),
        }

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)
            result["probabilities"] = probabilities.tolist()

        return result

    def output_fn(
        self,
        prediction: Dict[str, Any],
        accept: str = "application/json",
    ) -> str:
        """Format the prediction output.

        Args:
            prediction: Prediction dictionary from predict_fn.
            accept: Desired output format.

        Returns:
            Serialized prediction string.
        """
        if accept == "application/json":
            return json.dumps(prediction)
        elif accept == "text/csv":
            preds = prediction.get("predictions", [])
            return "\n".join(str(p) for p in preds)
        else:
            raise ValueError(f"Unsupported accept type: {accept}")

    def predict(
        self,
        data: Union[Dict, List, np.ndarray, pd.DataFrame],
    ) -> Dict[str, Any]:
        """Convenience method for end-to-end prediction."""
        if self._model is None:
            self.model_fn()

        if isinstance(data, pd.DataFrame):
            input_data = data.values
        elif isinstance(data, (dict, list)):
            input_data = self.input_fn(data)
        else:
            input_data = data

        return self.predict_fn(input_data)
