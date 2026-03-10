"""Unit tests for the SageMaker pipeline builder."""

import json

import pytest

from src.pipelines.sagemaker_pipeline import (
    SageMakerPipelineBuilder,
    create_default_pipeline,
)


class TestSageMakerPipelineBuilder:
    """Tests for pipeline construction."""

    def test_empty_pipeline(self):
        builder = SageMakerPipelineBuilder()
        pipeline = builder.build()
        assert pipeline["pipeline_name"] is not None
        assert pipeline["steps"] == []

    def test_add_processing_step(self):
        builder = SageMakerPipelineBuilder()
        builder.add_processing_step(
            input_s3_uri="s3://bucket/input",
            output_s3_uri="s3://bucket/output",
        )
        pipeline = builder.build()
        assert len(pipeline["steps"]) == 1
        assert pipeline["steps"][0]["name"] == "DataProcessing"
        assert pipeline["steps"][0]["type"] == "Processing"

    def test_add_training_step(self):
        builder = SageMakerPipelineBuilder()
        builder.add_training_step(
            train_s3_uri="s3://bucket/train",
            validation_s3_uri="s3://bucket/val",
            output_s3_uri="s3://bucket/models",
        )
        pipeline = builder.build()
        assert len(pipeline["steps"]) == 1
        assert pipeline["steps"][0]["name"] == "ModelTraining"

    def test_add_evaluation_step(self):
        builder = SageMakerPipelineBuilder()
        builder.add_evaluation_step(
            model_s3_uri="s3://bucket/model",
            test_s3_uri="s3://bucket/test",
            output_s3_uri="s3://bucket/eval",
        )
        pipeline = builder.build()
        assert pipeline["steps"][0]["name"] == "ModelEvaluation"

    def test_add_condition_step(self):
        builder = SageMakerPipelineBuilder()
        builder.add_condition_step(
            metric_name="accuracy", threshold=0.90
        )
        pipeline = builder.build()
        step = pipeline["steps"][0]
        assert step["name"] == "QualityGateCheck"
        assert step["config"]["conditions"][0]["right_value"] == 0.90

    def test_full_pipeline_chain(self):
        builder = SageMakerPipelineBuilder()
        pipeline = (
            builder.add_processing_step("s3://b/in", "s3://b/out")
            .add_training_step("s3://b/train", "s3://b/val", "s3://b/m")
            .add_evaluation_step("s3://b/m", "s3://b/test", "s3://b/e")
            .add_condition_step()
            .add_register_model_step("model-group")
            .add_deploy_step("endpoint")
            .build()
        )
        assert len(pipeline["steps"]) == 6

        step_names = [s["name"] for s in pipeline["steps"]]
        assert "DataProcessing" in step_names
        assert "ModelTraining" in step_names
        assert "ModelEvaluation" in step_names
        assert "QualityGateCheck" in step_names
        assert "RegisterModel" in step_names
        assert "DeployEndpoint" in step_names

    def test_step_dependencies(self):
        builder = SageMakerPipelineBuilder()
        builder.add_processing_step("s3://b/in", "s3://b/out")
        builder.add_training_step(
            "s3://b/train", "s3://b/val", "s3://b/m"
        )
        pipeline = builder.build()
        training_step = pipeline["steps"][1]
        assert "DataProcessing" in training_step["depends_on"]

    def test_save_definition(self, temp_dir):
        import os

        builder = SageMakerPipelineBuilder()
        builder.add_processing_step("s3://b/in", "s3://b/out")
        path = os.path.join(temp_dir, "pipeline.json")
        builder.save_definition(path)

        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
            assert "steps" in data

    def test_spot_instance_config(self):
        builder = SageMakerPipelineBuilder()
        builder.add_training_step(
            "s3://b/train",
            "s3://b/val",
            "s3://b/m",
            use_spot=True,
        )
        pipeline = builder.build()
        config = pipeline["steps"][0]["config"]["estimator"]
        assert config["use_spot_instances"] is True


class TestDefaultPipeline:
    """Tests for the default pipeline factory."""

    def test_create_default_pipeline(self):
        pipeline = create_default_pipeline()
        assert len(pipeline["steps"]) == 6
        assert pipeline["pipeline_name"] is not None
