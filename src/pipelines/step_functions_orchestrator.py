"""AWS Step Functions orchestrator for SageMaker ML pipelines.

Manages the state machine definition and execution of multi-step
ML workflows using Step Functions to coordinate SageMaker jobs.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config.settings import PipelineSettings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class StepFunctionState:
    """Represents a state in the Step Functions state machine."""

    name: str
    state_type: str
    resource: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    result_path: str = "$"
    next_state: Optional[str] = None
    end: bool = False
    retry: List[Dict[str, Any]] = field(default_factory=list)
    catch: List[Dict[str, Any]] = field(default_factory=list)


class StepFunctionsOrchestrator:
    """Orchestrates ML pipeline execution via AWS Step Functions.

    Builds a state machine that coordinates:
      - SageMaker Processing jobs (data prep)
      - SageMaker Training jobs (model training)
      - SageMaker Processing jobs (evaluation)
      - Choice state for quality gate decisions
      - SageMaker Model creation and endpoint deployment
      - SNS notifications for success/failure
    """

    def __init__(self, settings: Optional[PipelineSettings] = None):
        self.settings = settings or PipelineSettings()
        self._states: Dict[str, Dict[str, Any]] = {}

    def build_state_machine(self) -> Dict[str, Any]:
        """Build the complete Step Functions state machine definition."""
        sm = self.settings.sagemaker
        sf = self.settings.step_functions
        bucket = sm.default_bucket

        self._add_processing_state(
            name="PreprocessData",
            input_uri=f"s3://{bucket}/data/raw",
            output_uri=f"s3://{bucket}/data/processed",
            next_state="TrainModel",
        )

        self._add_training_state(
            name="TrainModel",
            train_uri=f"s3://{bucket}/data/processed/train",
            validation_uri=f"s3://{bucket}/data/processed/validation",
            output_uri=f"s3://{bucket}/models",
            next_state="EvaluateModel",
        )

        self._add_processing_state(
            name="EvaluateModel",
            input_uri=f"s3://{bucket}/models",
            output_uri=f"s3://{bucket}/evaluation",
            next_state="CheckQualityGate",
        )

        self._add_quality_gate_choice(
            name="CheckQualityGate",
            pass_state="RegisterModel",
            fail_state="NotifyFailure",
        )

        self._add_register_model_state(
            name="RegisterModel",
            model_group=sm.model_package_group,
            next_state="DeployEndpoint",
        )

        self._add_deploy_state(
            name="DeployEndpoint",
            endpoint_name=self.settings.endpoint.endpoint_name,
            next_state="NotifySuccess",
        )

        self._add_notification_state(
            name="NotifySuccess",
            topic_arn=sf.notification_topic_arn,
            subject="ML Pipeline - Model Deployed Successfully",
            is_terminal=True,
        )

        self._add_notification_state(
            name="NotifyFailure",
            topic_arn=sf.notification_topic_arn,
            subject="ML Pipeline - Quality Gate Failed",
            is_terminal=True,
        )

        definition = {
            "Comment": (
                "SageMaker MLOps Pipeline - End-to-end model training, "
                "evaluation, and deployment orchestration"
            ),
            "StartAt": "PreprocessData",
            "States": self._states,
        }

        logger.info(
            f"State machine built with {len(self._states)} states"
        )
        return definition

    def _add_processing_state(
        self,
        name: str,
        input_uri: str,
        output_uri: str,
        next_state: str,
    ) -> None:
        """Add a SageMaker Processing job state."""
        sm = self.settings.sagemaker
        self._states[name] = {
            "Type": "Task",
            "Resource": "arn:aws:states:::sagemaker:createProcessingJob.sync",
            "Parameters": {
                "ProcessingJobName.$": f"States.Format('{name}-{{}}', $$.Execution.Name)",
                "RoleArn": sm.role_arn,
                "ProcessingResources": {
                    "ClusterConfig": {
                        "InstanceCount": 1,
                        "InstanceType": "ml.m5.xlarge",
                        "VolumeSizeInGB": 30,
                    }
                },
                "AppSpecification": {
                    "ImageUri": self._ecr_image("ml-processing"),
                },
                "ProcessingInputs": [
                    {
                        "InputName": "input",
                        "S3Input": {
                            "S3Uri": input_uri,
                            "LocalPath": "/opt/ml/processing/input",
                            "S3DataType": "S3Prefix",
                            "S3InputMode": "File",
                        },
                    }
                ],
                "ProcessingOutputConfig": {
                    "Outputs": [
                        {
                            "OutputName": "output",
                            "S3Output": {
                                "S3Uri": output_uri,
                                "LocalPath": "/opt/ml/processing/output",
                                "S3UploadMode": "EndOfJob",
                            },
                        }
                    ]
                },
            },
            "ResultPath": f"$.{name}Result",
            "Next": next_state,
            "Retry": self._default_retry(),
            "Catch": self._default_catch("NotifyFailure"),
        }

    def _add_training_state(
        self,
        name: str,
        train_uri: str,
        validation_uri: str,
        output_uri: str,
        next_state: str,
    ) -> None:
        """Add a SageMaker Training job state."""
        sm = self.settings.sagemaker
        training = self.settings.training

        training_params: Dict[str, Any] = {
            "Type": "Task",
            "Resource": "arn:aws:states:::sagemaker:createTrainingJob.sync",
            "Parameters": {
                "TrainingJobName.$": f"States.Format('{name}-{{}}', $$.Execution.Name)",
                "RoleArn": sm.role_arn,
                "AlgorithmSpecification": {
                    "TrainingImage": self._ecr_image("ml-training"),
                    "TrainingInputMode": "File",
                },
                "ResourceConfig": {
                    "InstanceCount": training.instance_count,
                    "InstanceType": training.instance_type,
                    "VolumeSizeInGB": training.volume_size_gb,
                },
                "InputDataConfig": [
                    {
                        "ChannelName": "train",
                        "DataSource": {
                            "S3DataSource": {
                                "S3DataType": "S3Prefix",
                                "S3Uri": train_uri,
                            }
                        },
                    },
                    {
                        "ChannelName": "validation",
                        "DataSource": {
                            "S3DataSource": {
                                "S3DataType": "S3Prefix",
                                "S3Uri": validation_uri,
                            }
                        },
                    },
                ],
                "OutputDataConfig": {"S3OutputPath": output_uri},
                "StoppingCondition": {
                    "MaxRuntimeInSeconds": training.max_runtime_seconds,
                },
                "HyperParameters": training.default_hyperparameters,
            },
            "ResultPath": f"$.{name}Result",
            "Next": next_state,
            "Retry": self._default_retry(),
            "Catch": self._default_catch("NotifyFailure"),
        }

        if training.use_spot_instances:
            training_params["Parameters"]["EnableManagedSpotTraining"] = True
            training_params["Parameters"]["StoppingCondition"][
                "MaxWaitTimeInSeconds"
            ] = training.max_wait_seconds

        self._states[name] = training_params

    def _add_quality_gate_choice(
        self,
        name: str,
        pass_state: str,
        fail_state: str,
        metric: str = "f1",
        threshold: float = 0.78,
    ) -> None:
        """Add a Choice state for quality gate evaluation."""
        self._states[name] = {
            "Type": "Choice",
            "Choices": [
                {
                    "Variable": f"$.EvaluateModelResult.Metrics.{metric}",
                    "NumericGreaterThanEquals": threshold,
                    "Next": pass_state,
                }
            ],
            "Default": fail_state,
        }

    def _add_register_model_state(
        self,
        name: str,
        model_group: str,
        next_state: str,
    ) -> None:
        """Add a model registration state."""
        sm = self.settings.sagemaker
        self._states[name] = {
            "Type": "Task",
            "Resource": "arn:aws:states:::sagemaker:createModelPackage",
            "Parameters": {
                "ModelPackageGroupName": model_group,
                "ModelApprovalStatus": "PendingManualApproval",
                "InferenceSpecification": {
                    "Containers": [
                        {
                            "Image": self._ecr_image("ml-inference"),
                            "ModelDataUrl.$": "$.TrainModelResult.ModelArtifacts.S3ModelArtifacts",
                        }
                    ],
                    "SupportedContentTypes": [
                        "application/json",
                        "text/csv",
                    ],
                    "SupportedResponseMIMETypes": ["application/json"],
                },
            },
            "ResultPath": f"$.{name}Result",
            "Next": next_state,
            "Retry": self._default_retry(),
            "Catch": self._default_catch("NotifyFailure"),
        }

    def _add_deploy_state(
        self,
        name: str,
        endpoint_name: str,
        next_state: str,
    ) -> None:
        """Add an endpoint deployment state."""
        endpoint_cfg = self.settings.endpoint
        self._states[name] = {
            "Type": "Task",
            "Resource": "arn:aws:states:::sagemaker:createEndpoint",
            "Parameters": {
                "EndpointName": endpoint_name,
                "EndpointConfigName.$": (
                    "States.Format('"
                    + endpoint_name
                    + "-config-{}', $$.Execution.Name)"
                ),
            },
            "ResultPath": f"$.{name}Result",
            "Next": next_state,
            "Retry": self._default_retry(),
            "Catch": self._default_catch("NotifyFailure"),
        }

    def _add_notification_state(
        self,
        name: str,
        topic_arn: str,
        subject: str,
        is_terminal: bool = False,
    ) -> None:
        """Add an SNS notification state."""
        state: Dict[str, Any] = {
            "Type": "Task",
            "Resource": "arn:aws:states:::sns:publish",
            "Parameters": {
                "TopicArn": topic_arn,
                "Subject": subject,
                "Message.$": "States.JsonToString($)",
            },
        }
        if is_terminal:
            state["End"] = True
        state["Retry"] = [
            {
                "ErrorEquals": ["States.TaskFailed"],
                "IntervalSeconds": 5,
                "MaxAttempts": 2,
                "BackoffRate": 2.0,
            }
        ]
        self._states[name] = state

    def _default_retry(self) -> List[Dict[str, Any]]:
        """Default retry policy for SageMaker states."""
        sf = self.settings.step_functions
        return [
            {
                "ErrorEquals": [
                    "SageMaker.AmazonSageMakerException",
                    "States.TaskFailed",
                ],
                "IntervalSeconds": sf.retry_interval_seconds,
                "MaxAttempts": sf.max_retry_attempts,
                "BackoffRate": sf.retry_backoff_rate,
            }
        ]

    def _default_catch(self, fallback: str) -> List[Dict[str, Any]]:
        """Default catch policy for error handling."""
        return [
            {
                "ErrorEquals": ["States.ALL"],
                "ResultPath": "$.error",
                "Next": fallback,
            }
        ]

    def _ecr_image(self, name: str) -> str:
        """Build ECR image URI."""
        sm = self.settings.sagemaker
        account = sm.role_arn.split(":")[4]
        return f"{account}.dkr.ecr.{sm.region}.amazonaws.com/{name}:latest"

    def save_definition(self, output_path: str) -> None:
        """Save the state machine definition to JSON."""
        definition = self.build_state_machine()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "w") as f:
            json.dump(definition, f, indent=2)

        logger.info(f"State machine definition saved to {output_path}")


@dataclass
class ExecutionStatus:
    """Status of a Step Functions execution."""

    execution_arn: str
    status: str
    start_time: str
    stop_time: Optional[str] = None
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ExecutionTracker:
    """Tracks and monitors Step Functions pipeline executions.

    Provides methods to start executions, poll for completion,
    and retrieve execution history.
    """

    def __init__(self, state_machine_arn: str, region: str = "us-east-1"):
        self.state_machine_arn = state_machine_arn
        self.region = region
        self._executions: Dict[str, ExecutionStatus] = {}

    def start_execution(
        self,
        execution_name: str,
        input_payload: Optional[Dict[str, Any]] = None,
    ) -> ExecutionStatus:
        """Start a new state machine execution."""
        execution_arn = (
            f"{self.state_machine_arn.replace(':stateMachine:', ':execution:')}"
            f":{execution_name}"
        )

        status = ExecutionStatus(
            execution_arn=execution_arn,
            status="RUNNING",
            start_time=datetime.utcnow().isoformat(),
        )
        self._executions[execution_arn] = status

        logger.info(
            f"Started execution: {execution_name} "
            f"(arn: {execution_arn})"
        )
        return status

    def get_execution_status(
        self, execution_arn: str
    ) -> Optional[ExecutionStatus]:
        """Retrieve the current status of an execution."""
        return self._executions.get(execution_arn)

    def list_executions(
        self,
        status_filter: Optional[str] = None,
    ) -> List[ExecutionStatus]:
        """List all tracked executions, optionally filtered by status."""
        executions = list(self._executions.values())
        if status_filter:
            executions = [
                e for e in executions if e.status == status_filter
            ]
        return sorted(executions, key=lambda e: e.start_time, reverse=True)

    def mark_completed(
        self,
        execution_arn: str,
        output: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mark an execution as successfully completed."""
        if execution_arn in self._executions:
            self._executions[execution_arn].status = "SUCCEEDED"
            self._executions[execution_arn].stop_time = (
                datetime.utcnow().isoformat()
            )
            self._executions[execution_arn].output = output
            logger.info(f"Execution completed: {execution_arn}")

    def mark_failed(
        self, execution_arn: str, error: str
    ) -> None:
        """Mark an execution as failed."""
        if execution_arn in self._executions:
            self._executions[execution_arn].status = "FAILED"
            self._executions[execution_arn].stop_time = (
                datetime.utcnow().isoformat()
            )
            self._executions[execution_arn].error = error
            logger.warning(f"Execution failed: {execution_arn} - {error}")
