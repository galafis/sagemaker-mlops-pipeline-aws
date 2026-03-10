"""Unit tests for the SageMaker inference module."""

import json

import numpy as np
import pytest

from src.inference.inference_handler import InferenceHandler
from src.inference.serializer import (
    NumpyEncoder,
    deserialize_response,
    serialize_request,
)


class TestInferenceHandler:
    """Tests for InferenceHandler functionality."""

    def test_model_fn(self, model_dir):
        handler = InferenceHandler(model_dir=model_dir)
        model = handler.model_fn()
        assert model is not None

    def test_model_fn_missing(self, temp_dir):
        handler = InferenceHandler(model_dir=temp_dir)
        with pytest.raises(FileNotFoundError):
            handler.model_fn()

    def test_input_fn_json_instances(self):
        handler = InferenceHandler()
        body = json.dumps({"instances": [[1.0, 2.0, 3.0]]})
        result = handler.input_fn(body, "application/json")
        assert result.shape == (1, 3)

    def test_input_fn_json_features(self):
        handler = InferenceHandler()
        body = json.dumps({"features": [[1.0, 2.0], [3.0, 4.0]]})
        result = handler.input_fn(body, "application/json")
        assert result.shape == (2, 2)

    def test_input_fn_json_list(self):
        handler = InferenceHandler()
        body = json.dumps([[1.0, 2.0], [3.0, 4.0]])
        result = handler.input_fn(body, "application/json")
        assert result.shape == (2, 2)

    def test_input_fn_csv(self):
        handler = InferenceHandler()
        body = "1.0,2.0,3.0\n4.0,5.0,6.0"
        result = handler.input_fn(body, "text/csv")
        assert result.shape == (2, 3)

    def test_input_fn_unsupported(self):
        handler = InferenceHandler()
        with pytest.raises(ValueError, match="Unsupported content type"):
            handler.input_fn("data", "application/xml")

    def test_predict_fn(self, model_dir, sample_data):
        X, _ = sample_data
        handler = InferenceHandler(model_dir=model_dir)
        handler.model_fn()

        result = handler.predict_fn(X[:5])
        assert "predictions" in result
        assert len(result["predictions"]) == 5

    def test_predict_fn_probabilities(self, model_dir, sample_data):
        X, _ = sample_data
        handler = InferenceHandler(model_dir=model_dir)
        handler.model_fn()

        result = handler.predict_fn(X[:3])
        assert "probabilities" in result
        assert len(result["probabilities"]) == 3

    def test_predict_fn_no_model(self):
        handler = InferenceHandler()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            handler.predict_fn(np.array([[1, 2, 3]]))

    def test_output_fn_json(self):
        handler = InferenceHandler()
        prediction = {"predictions": [0, 1, 0]}
        result = handler.output_fn(prediction, "application/json")
        parsed = json.loads(result)
        assert parsed["predictions"] == [0, 1, 0]

    def test_output_fn_csv(self):
        handler = InferenceHandler()
        prediction = {"predictions": [0, 1, 0]}
        result = handler.output_fn(prediction, "text/csv")
        assert result == "0\n1\n0"

    def test_output_fn_unsupported(self):
        handler = InferenceHandler()
        with pytest.raises(ValueError, match="Unsupported accept type"):
            handler.output_fn({}, "application/xml")

    def test_end_to_end_predict(self, model_dir, sample_data):
        X, _ = sample_data
        handler = InferenceHandler(model_dir=model_dir)
        result = handler.predict(X[:10])
        assert "predictions" in result
        assert len(result["predictions"]) == 10


class TestSerializer:
    """Tests for request/response serialization."""

    def test_serialize_json_list(self):
        features = [[1.0, 2.0], [3.0, 4.0]]
        result = serialize_request(features, "application/json")
        parsed = json.loads(result)
        assert parsed["instances"] == features

    def test_serialize_json_numpy(self):
        features = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = serialize_request(features, "application/json")
        parsed = json.loads(result)
        assert parsed["instances"] == [[1.0, 2.0], [3.0, 4.0]]

    def test_serialize_json_dict(self):
        features = {"data": [1, 2, 3]}
        result = serialize_request(features, "application/json")
        parsed = json.loads(result)
        assert parsed == features

    def test_serialize_csv(self):
        features = [[1.0, 2.0], [3.0, 4.0]]
        result = serialize_request(features, "text/csv")
        assert result == "1.0,2.0\n3.0,4.0"

    def test_serialize_csv_single_row(self):
        features = [1.0, 2.0, 3.0]
        result = serialize_request(features, "text/csv")
        assert result == "1.0,2.0,3.0"

    def test_serialize_unsupported(self):
        with pytest.raises(ValueError):
            serialize_request([[1]], "application/xml")

    def test_deserialize_json(self):
        body = json.dumps({"predictions": [0, 1]})
        result = deserialize_response(body, "application/json")
        assert result["predictions"] == [0, 1]

    def test_deserialize_csv(self):
        body = "0.5\n0.8\n0.3"
        result = deserialize_response(body, "text/csv")
        assert result["predictions"] == [0.5, 0.8, 0.3]

    def test_numpy_encoder(self):
        data = {
            "int": np.int64(42),
            "float": np.float64(3.14),
            "array": np.array([1, 2, 3]),
        }
        result = json.dumps(data, cls=NumpyEncoder)
        parsed = json.loads(result)
        assert parsed["int"] == 42
        assert parsed["float"] == 3.14
        assert parsed["array"] == [1, 2, 3]
