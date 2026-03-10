"""Hyperparameter management and search space definition."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class HyperparameterRange:
    """Defines a hyperparameter search range."""

    name: str
    param_type: str  # continuous, integer, categorical
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    values: Optional[List[Any]] = None
    scaling: str = "Auto"

    def to_sagemaker_format(self) -> Dict[str, Any]:
        if self.param_type == "continuous":
            return {
                "Name": self.name,
                "MinValue": str(self.min_value),
                "MaxValue": str(self.max_value),
                "ScalingType": self.scaling,
            }
        elif self.param_type == "integer":
            return {
                "Name": self.name,
                "MinValue": str(int(self.min_value)),
                "MaxValue": str(int(self.max_value)),
                "ScalingType": self.scaling,
            }
        else:
            return {
                "Name": self.name,
                "Values": [str(v) for v in (self.values or [])],
            }


class HyperparameterConfig:
    """Manages hyperparameter configurations for tuning jobs."""

    @staticmethod
    def get_xgboost_search_space() -> List[HyperparameterRange]:
        return [
            HyperparameterRange("eta", "continuous", 0.01, 0.3),
            HyperparameterRange("max_depth", "integer", 3, 10),
            HyperparameterRange("subsample", "continuous", 0.6, 1.0),
            HyperparameterRange("colsample_bytree", "continuous", 0.6, 1.0),
            HyperparameterRange("num_round", "integer", 100, 500),
            HyperparameterRange("min_child_weight", "integer", 1, 10),
            HyperparameterRange("gamma", "continuous", 0.0, 5.0),
        ]

    @staticmethod
    def get_default_params(algorithm: str = "xgboost") -> Dict[str, Any]:
        defaults = {
            "xgboost": {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "num_round": 200,
                "max_depth": 6,
                "eta": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
            "gradient_boosting": {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
            },
            "random_forest": {
                "n_estimators": 300,
                "max_depth": 12,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
            },
        }
        return defaults.get(algorithm, {})

    @staticmethod
    def merge_params(
        defaults: Dict[str, Any], overrides: Dict[str, Any]
    ) -> Dict[str, Any]:
        merged = defaults.copy()
        merged.update(overrides)
        return merged
