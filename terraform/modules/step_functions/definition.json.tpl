{
  "Comment": "SageMaker MLOps Pipeline - End-to-end ML workflow",
  "StartAt": "PreprocessData",
  "States": {
    "PreprocessData": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sagemaker:createProcessingJob.sync",
      "Parameters": {
        "ProcessingJobName.$": "States.Format('preprocess-{}', $$.Execution.Name)",
        "RoleArn": "${sagemaker_role}",
        "ProcessingResources": {
          "ClusterConfig": {
            "InstanceCount": 1,
            "InstanceType": "ml.m5.xlarge",
            "VolumeSizeInGB": 30
          }
        },
        "AppSpecification": {
          "ImageUri": "${account_id}.dkr.ecr.${region}.amazonaws.com/ml-processing:latest"
        },
        "ProcessingInputs": [
          {
            "InputName": "raw-data",
            "S3Input": {
              "S3Uri": "s3://${data_bucket}/data/raw",
              "LocalPath": "/opt/ml/processing/input",
              "S3DataType": "S3Prefix",
              "S3InputMode": "File"
            }
          }
        ],
        "ProcessingOutputConfig": {
          "Outputs": [
            {
              "OutputName": "processed-data",
              "S3Output": {
                "S3Uri": "s3://${data_bucket}/data/processed",
                "LocalPath": "/opt/ml/processing/output",
                "S3UploadMode": "EndOfJob"
              }
            }
          ]
        }
      },
      "ResultPath": "$.PreprocessResult",
      "Next": "TrainModel",
      "Retry": [
        {
          "ErrorEquals": ["SageMaker.AmazonSageMakerException"],
          "IntervalSeconds": 30,
          "MaxAttempts": 2,
          "BackoffRate": 2.0
        }
      ],
      "Catch": [
        {
          "ErrorEquals": ["States.ALL"],
          "ResultPath": "$.error",
          "Next": "NotifyFailure"
        }
      ]
    },
    "TrainModel": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sagemaker:createTrainingJob.sync",
      "Parameters": {
        "TrainingJobName.$": "States.Format('train-{}', $$.Execution.Name)",
        "RoleArn": "${sagemaker_role}",
        "AlgorithmSpecification": {
          "TrainingImage": "${account_id}.dkr.ecr.${region}.amazonaws.com/ml-training:latest",
          "TrainingInputMode": "File"
        },
        "ResourceConfig": {
          "InstanceCount": 1,
          "InstanceType": "ml.m5.xlarge",
          "VolumeSizeInGB": 30
        },
        "EnableManagedSpotTraining": true,
        "StoppingCondition": {
          "MaxRuntimeInSeconds": 3600,
          "MaxWaitTimeInSeconds": 7200
        },
        "InputDataConfig": [
          {
            "ChannelName": "train",
            "DataSource": {
              "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": "s3://${data_bucket}/data/processed/train"
              }
            }
          },
          {
            "ChannelName": "validation",
            "DataSource": {
              "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": "s3://${data_bucket}/data/processed/validation"
              }
            }
          }
        ],
        "OutputDataConfig": {
          "S3OutputPath": "s3://${data_bucket}/models"
        }
      },
      "ResultPath": "$.TrainResult",
      "Next": "EvaluateModel",
      "Retry": [
        {
          "ErrorEquals": ["SageMaker.AmazonSageMakerException"],
          "IntervalSeconds": 60,
          "MaxAttempts": 2,
          "BackoffRate": 2.0
        }
      ],
      "Catch": [
        {
          "ErrorEquals": ["States.ALL"],
          "ResultPath": "$.error",
          "Next": "NotifyFailure"
        }
      ]
    },
    "EvaluateModel": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sagemaker:createProcessingJob.sync",
      "Parameters": {
        "ProcessingJobName.$": "States.Format('evaluate-{}', $$.Execution.Name)",
        "RoleArn": "${sagemaker_role}",
        "ProcessingResources": {
          "ClusterConfig": {
            "InstanceCount": 1,
            "InstanceType": "ml.m5.large",
            "VolumeSizeInGB": 10
          }
        },
        "AppSpecification": {
          "ImageUri": "${account_id}.dkr.ecr.${region}.amazonaws.com/ml-processing:latest"
        },
        "ProcessingInputs": [
          {
            "InputName": "model",
            "S3Input": {
              "S3Uri.$": "$.TrainResult.ModelArtifacts.S3ModelArtifacts",
              "LocalPath": "/opt/ml/processing/model",
              "S3DataType": "S3Prefix",
              "S3InputMode": "File"
            }
          },
          {
            "InputName": "test-data",
            "S3Input": {
              "S3Uri": "s3://${data_bucket}/data/processed/test",
              "LocalPath": "/opt/ml/processing/test",
              "S3DataType": "S3Prefix",
              "S3InputMode": "File"
            }
          }
        ],
        "ProcessingOutputConfig": {
          "Outputs": [
            {
              "OutputName": "evaluation",
              "S3Output": {
                "S3Uri": "s3://${data_bucket}/evaluation",
                "LocalPath": "/opt/ml/processing/evaluation",
                "S3UploadMode": "EndOfJob"
              }
            }
          ]
        }
      },
      "ResultPath": "$.EvaluateResult",
      "Next": "CheckQualityGate",
      "Catch": [
        {
          "ErrorEquals": ["States.ALL"],
          "ResultPath": "$.error",
          "Next": "NotifyFailure"
        }
      ]
    },
    "CheckQualityGate": {
      "Type": "Choice",
      "Choices": [
        {
          "Variable": "$.EvaluateResult.Metrics.f1",
          "NumericGreaterThanEquals": 0.78,
          "Next": "RegisterModel"
        }
      ],
      "Default": "NotifyFailure"
    },
    "RegisterModel": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sagemaker:createModelPackage",
      "Parameters": {
        "ModelPackageGroupName": "sagemaker-mlops-models",
        "ModelApprovalStatus": "PendingManualApproval",
        "InferenceSpecification": {
          "Containers": [
            {
              "Image": "${account_id}.dkr.ecr.${region}.amazonaws.com/ml-inference:latest",
              "ModelDataUrl.$": "$.TrainResult.ModelArtifacts.S3ModelArtifacts"
            }
          ],
          "SupportedContentTypes": ["application/json", "text/csv"],
          "SupportedResponseMIMETypes": ["application/json"]
        }
      },
      "ResultPath": "$.RegisterResult",
      "Next": "NotifySuccess",
      "Catch": [
        {
          "ErrorEquals": ["States.ALL"],
          "ResultPath": "$.error",
          "Next": "NotifyFailure"
        }
      ]
    },
    "NotifySuccess": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sns:publish",
      "Parameters": {
        "TopicArn": "${sns_topic_arn}",
        "Subject": "ML Pipeline - Model Registered Successfully",
        "Message.$": "States.JsonToString($)"
      },
      "End": true
    },
    "NotifyFailure": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sns:publish",
      "Parameters": {
        "TopicArn": "${sns_topic_arn}",
        "Subject": "ML Pipeline - Execution Failed",
        "Message.$": "States.JsonToString($)"
      },
      "End": true
    }
  }
}
