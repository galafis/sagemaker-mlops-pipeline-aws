"""Pipeline settings and configuration management."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class SageMakerConfig:
    """SageMaker-specific configuration."""

    role_arn: str = "arn:aws:iam::role/SageMakerExecutionRole"
    instance_type: str = "ml.m5.xlarge"
    instance_count: int = 1
    volume_size_gb: int = 30
    max_runtime_seconds: int = 86400
    use_spot_instances: bool = True
    max_wait_seconds: int = 172800
    output_bucket: str = "sagemaker-mlops-artifacts"
    output_prefix: str = "pipeline-outputs"


@dataclass
class TrainingConfig:
    """Model training configuration."""

    algorithm: str = "xgboost"
    framework_version: str = "1.7-1"
    objective: str = "binary:logistic"
    num_round: int = 200
    max_depth: int = 6
    eta: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    eval_metric: str = "auc"
    early_stopping_rounds: int = 20


@dataclass
class EndpointConfig:
    """SageMaker endpoint configuration."""

    instance_type: str = "ml.m5.large"
    initial_instance_count: int = 1
    min_instance_count: int = 1
    max_instance_count: int = 4
    target_invocations_per_instance: int = 100
    scale_in_cooldown: int = 300
    scale_out_cooldown: int = 60
    data_capture_percentage: int = 10


@dataclass
class ABTestConfig:
    """A/B testing configuration for endpoints."""

    enabled: bool = True
    champion_traffic_weight: float = 0.9
    challenger_traffic_weight: float = 0.1
    min_sample_size: int = 1000
    significance_level: float = 0.05
    metric_name: str = "auc"


@dataclass
class StepFunctionsConfig:
    """Step Functions orchestration settings."""

    state_machine_name: str = "ml-pipeline-orchestrator"
    max_retries: int = 3
    retry_interval_seconds: int = 60
    backoff_rate: float = 2.0
    timeout_seconds: int = 7200


@dataclass
class PipelineSettings:
    """Root settings aggregating all configurations."""

    sagemaker: SageMakerConfig = field(default_factory=SageMakerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    endpoint: EndpointConfig = field(default_factory=EndpointConfig)
    ab_test: ABTestConfig = field(default_factory=ABTestConfig)
    step_functions: StepFunctionsConfig = field(default_factory=StepFunctionsConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineSettings":
        config_path = Path(path)
        if not config_path.exists():
            return cls()
        with open(config_path) as f:
            raw: Dict[str, Any] = yaml.safe_load(f) or {}
        return cls(
            sagemaker=SageMakerConfig(**raw.get("sagemaker", {})),
            training=TrainingConfig(**raw.get("training", {})),
            endpoint=EndpointConfig(**raw.get("endpoint", {})),
            ab_test=ABTestConfig(**raw.get("ab_test", {})),
            step_functions=StepFunctionsConfig(**raw.get("step_functions", {})),
        )
