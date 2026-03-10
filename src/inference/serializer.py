"""Request/response serialization for SageMaker endpoints."""

import json
from typing import Any, Dict, List, Union

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def serialize_request(
    features: Union[List, np.ndarray, Dict],
    content_type: str = "application/json",
) -> str:
    """Serialize prediction request to the specified format."""
    if content_type == "application/json":
        if isinstance(features, np.ndarray):
            features = features.tolist()

        if isinstance(features, dict):
            payload = features
        else:
            payload = {"instances": features}

        return json.dumps(payload, cls=NumpyEncoder)

    elif content_type == "text/csv":
        if isinstance(features, np.ndarray):
            rows = [",".join(str(v) for v in row) for row in features]
        elif isinstance(features, list):
            if isinstance(features[0], list):
                rows = [",".join(str(v) for v in row) for row in features]
            else:
                rows = [",".join(str(v) for v in features)]
        else:
            raise ValueError("CSV serialization requires list or array")
        return "\n".join(rows)

    raise ValueError(f"Unsupported content type: {content_type}")


def deserialize_response(
    response_body: str,
    content_type: str = "application/json",
) -> Dict[str, Any]:
    """Deserialize prediction response from the endpoint."""
    if content_type == "application/json":
        return json.loads(response_body)
    elif content_type == "text/csv":
        values = [float(v.strip()) for v in response_body.strip().split("\n") if v.strip()]
        return {"predictions": values}
    raise ValueError(f"Unsupported content type: {content_type}")
