"""SageMaker Pipeline definition for end-to-end ML workflow.

Orchestrates preprocessing, training, evaluation, and conditional
deployment using SageMaker Pipelines native steps.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config.settings import PipelineSettings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineStep:
    """Represents a single step in the SageMaker pipeline."""

    name: str
    step_type: str
    config: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)


class SageMakerPipelineBuilder:
    """Builds and manages SageMaker Pipeline definitions.

    Constructs a multi-step pipeline with:
      - Processing step for data preparation
      - Training step with optional hyperparameter tuning
      - Evaluation step with quality gate checks
      - Conditional model registration
      - Endpoint deployment with A/B testing support
    """

    def __init__(self, settings: Optional[PipelineSettings] = None):
        self.settings = settings or PipelineSettings()
        self._steps: List[PipelineStep] = []
        self._pipeline_name = self.settings.sagemaker.pipeline_name

    def add_processing_step(
        self,
        input_s3_uri: str,
        output_s3_uri: str,
        instance_type: str = "ml.m5.xlarge",
        instance_count: int = 1,
    ) -> "SageMakerPipelineBuilder":
        """Add a data processing step to the pipeline."""
        step = PipelineStep(
            name="DataProcessing",
            step_type="Processing",
            config={
                "processor": {
                    "role": self.settings.sagemaker.role_arn,
                    "instance_type": instance_type,
                    "instance_count": instance_count,
                    "image_uri": self._get_processing_image_uri(),
                },
                "inputs": [
                    {
                        "source": input_s3_uri,
                        "destination": "/opt/ml/processing/input",
                        "input_name": "raw_data",
                    }
                ],
                "outputs": [
                    {
                        "source": "/opt/ml/processing/output/train",
                        "destination": f"{output_s3_uri}/train",
                        "output_name": "train",
                    },
                    {
                        "source": "/opt/ml/processing/output/validation",
                        "destination": f"{output_s3_uri}/validation",
                        "output_name": "validation",
                    },
                    {
                        "source": "/opt/ml/processing/output/test",
                        "destination": f"{output_s3_uri}/test",
                        "output_name": "test",
                    },
                ],
                "code": "src/processing/preprocessing.py",
            },
        )
        self._steps.append(step)
        logger.info(f"Added processing step: {step.name}")
        return self

    def add_training_step(
        self,
        train_s3_uri: str,
        validation_s3_uri: str,
        output_s3_uri: str,
        hyperparameters: Optional[Dict[str, str]] = None,
        use_spot: Optional[bool] = None,
    ) -> "SageMakerPipelineBuilder":
        """Add a model training step."""
        training_cfg = self.settings.training
        use_spot_instances = (
            use_spot if use_spot is not None else training_cfg.use_spot_instances
        )

        step = PipelineStep(
            name="ModelTraining",
            step_type="Training",
            config={
                "estimator": {
                    "role": self.settings.sagemaker.role_arn,
                    "instance_type": training_cfg.instance_type,
                    "instance_count": training_cfg.instance_count,
                    "output_path": output_s3_uri,
                    "image_uri": self._get_training_image_uri(),
                    "use_spot_instances": use_spot_instances,
                    "max_wait": training_cfg.max_wait_seconds
                    if use_spot_instances
                    else None,
                    "max_run": training_cfg.max_runtime_seconds,
                    "hyperparameters": hyperparameters
                    or training_cfg.default_hyperparameters,
                },
                "inputs": {
                    "train": train_s3_uri,
                    "validation": validation_s3_uri,
                },
            },
            depends_on=["DataProcessing"],
        )
        self._steps.append(step)
        logger.info(
            f"Added training step (spot={use_spot_instances}): {step.name}"
        )
        return self

    def add_evaluation_step(
        self,
        model_s3_uri: str,
        test_s3_uri: str,
        output_s3_uri: str,
    ) -> "SageMakerPipelineBuilder":
        """Add model evaluation with quality gate check."""
        step = PipelineStep(
            name="ModelEvaluation",
            step_type="Processing",
            config={
                "processor": {
                    "role": self.settings.sagemaker.role_arn,
                    "instance_type": "ml.m5.large",
                    "instance_count": 1,
                    "image_uri": self._get_processing_image_uri(),
                },
                "inputs": [
                    {
                        "source": model_s3_uri,
                        "destination": "/opt/ml/processing/model",
                        "input_name": "model",
                    },
                    {
                        "source": test_s3_uri,
                        "destination": "/opt/ml/processing/test",
                        "input_name": "test_data",
                    },
                ],
                "outputs": [
                    {
                        "source": "/opt/ml/processing/evaluation",
                        "destination": f"{output_s3_uri}/evaluation",
                        "output_name": "evaluation",
                    }
                ],
                "code": "src/evaluation/model_evaluator.py",
            },
            depends_on=["ModelTraining"],
        )
        self._steps.append(step)
        logger.info(f"Added evaluation step: {step.name}")
        return self

    def add_condition_step(
        self,
        metric_name: str = "f1",
        threshold: float = 0.78,
    ) -> "SageMakerPipelineBuilder":
        """Add a conditional step that gates model registration."""
        step = PipelineStep(
            name="QualityGateCheck",
            step_type="Condition",
            config={
                "conditions": [
                    {
                        "type": "GreaterThanOrEqualTo",
                        "left_value": f"steps.ModelEvaluation.properties.{metric_name}",
                        "right_value": threshold,
                    }
                ],
                "if_steps": ["RegisterModel", "DeployEndpoint"],
                "else_steps": ["NotifyFailure"],
            },
            depends_on=["ModelEvaluation"],
        )
        self._steps.append(step)
        logger.info(
            f"Added condition step: {metric_name} >= {threshold}"
        )
        return self

    def add_register_model_step(
        self,
        model_package_group: str,
        approval_status: str = "PendingManualApproval",
    ) -> "SageMakerPipelineBuilder":
        """Add model registration step to the Model Registry."""
        step = PipelineStep(
            name="RegisterModel",
            step_type="RegisterModel",
            config={
                "model_package_group_name": model_package_group,
                "approval_status": approval_status,
                "model_metrics": {
                    "source": "steps.ModelEvaluation.properties",
                },
                "inference_spec": {
                    "supported_content_types": [
                        "application/json",
                        "text/csv",
                    ],
                    "supported_response_types": ["application/json"],
                },
            },
            depends_on=["QualityGateCheck"],
        )
        self._steps.append(step)
        logger.info(f"Added model registration step: {model_package_group}")
        return self

    def add_deploy_step(
        self,
        endpoint_name: str,
        instance_type: str = "ml.m5.large",
        initial_instance_count: int = 1,
    ) -> "SageMakerPipelineBuilder":
        """Add endpoint deployment step."""
        endpoint_cfg = self.settings.endpoint
        step = PipelineStep(
            name="DeployEndpoint",
            step_type="Deploy",
            config={
                "endpoint_name": endpoint_name or endpoint_cfg.endpoint_name,
                "instance_type": instance_type or endpoint_cfg.instance_type,
                "initial_instance_count": (
                    initial_instance_count or endpoint_cfg.initial_instance_count
                ),
                "auto_scaling": {
                    "min_capacity": endpoint_cfg.auto_scaling_min,
                    "max_capacity": endpoint_cfg.auto_scaling_max,
                    "target_invocations": endpoint_cfg.auto_scaling_target_invocations,
                },
            },
            depends_on=["RegisterModel"],
        )
        self._steps.append(step)
        logger.info(f"Added deploy step: {endpoint_name}")
        return self

    def build(self) -> Dict[str, Any]:
        """Build the complete pipeline definition."""
        pipeline_def = {
            "pipeline_name": self._pipeline_name,
            "pipeline_description": (
                "End-to-end MLOps pipeline with preprocessing, "
                "training, evaluation, quality gates, and deployment"
            ),
            "steps": [
                {
                    "name": s.name,
                    "type": s.step_type,
                    "config": s.config,
                    "depends_on": s.depends_on,
                }
                for s in self._steps
            ],
        }
        logger.info(
            f"Pipeline '{self._pipeline_name}' built with "
            f"{len(self._steps)} steps"
        )
        return pipeline_def

    def save_definition(self, output_path: str) -> None:
        """Save the pipeline definition to a JSON file."""
        definition = self.build()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "w") as f:
            json.dump(definition, f, indent=2)

        logger.info(f"Pipeline definition saved to {output_path}")

    def _get_processing_image_uri(self) -> str:
        """Get the ECR image URI for processing jobs."""
        region = self.settings.sagemaker.region
        account = self.settings.sagemaker.role_arn.split(":")[4]
        return (
            f"{account}.dkr.ecr.{region}.amazonaws.com/"
            f"ml-processing:latest"
        )

    def _get_training_image_uri(self) -> str:
        """Get the ECR image URI for training jobs."""
        region = self.settings.sagemaker.region
        account = self.settings.sagemaker.role_arn.split(":")[4]
        return (
            f"{account}.dkr.ecr.{region}.amazonaws.com/"
            f"ml-training:latest"
        )


def create_default_pipeline(
    settings: Optional[PipelineSettings] = None,
) -> Dict[str, Any]:
    """Create a standard MLOps pipeline with default configuration.

    Builds the full pipeline: preprocess → train → evaluate →
    quality gate → register → deploy.
    """
    settings = settings or PipelineSettings()
    sm = settings.sagemaker
    bucket = sm.default_bucket

    builder = SageMakerPipelineBuilder(settings)
    pipeline = (
        builder.add_processing_step(
            input_s3_uri=f"s3://{bucket}/data/raw",
            output_s3_uri=f"s3://{bucket}/data/processed",
        )
        .add_training_step(
            train_s3_uri=f"s3://{bucket}/data/processed/train",
            validation_s3_uri=f"s3://{bucket}/data/processed/validation",
            output_s3_uri=f"s3://{bucket}/models",
        )
        .add_evaluation_step(
            model_s3_uri=f"s3://{bucket}/models",
            test_s3_uri=f"s3://{bucket}/data/processed/test",
            output_s3_uri=f"s3://{bucket}/evaluation",
        )
        .add_condition_step(metric_name="f1", threshold=0.78)
        .add_register_model_step(
            model_package_group=sm.model_package_group,
        )
        .add_deploy_step(
            endpoint_name=settings.endpoint.endpoint_name,
        )
        .build()
    )

    logger.info("Default pipeline created successfully")
    return pipeline
