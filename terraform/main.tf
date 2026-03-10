###############################################################################
# SageMaker MLOps Pipeline - Terraform Infrastructure
# Provisions all AWS resources for the end-to-end ML pipeline.
###############################################################################

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.30"
    }
  }

  backend "s3" {
    bucket         = "ml-terraform-state"
    key            = "sagemaker-mlops/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-locks"
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "sagemaker-mlops-pipeline"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

# ─── Data Sources ────────────────────────────────────────────────────────────

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# ─── S3 Buckets ──────────────────────────────────────────────────────────────

module "s3" {
  source = "./modules/s3"

  project_name = var.project_name
  environment  = var.environment
}

# ─── IAM Roles ───────────────────────────────────────────────────────────────

module "sagemaker" {
  source = "./modules/sagemaker"

  project_name       = var.project_name
  environment        = var.environment
  data_bucket_arn    = module.s3.data_bucket_arn
  model_bucket_arn   = module.s3.model_bucket_arn
  ecr_repository_arn = aws_ecr_repository.ml_images.arn
}

# ─── ECR Repository ─────────────────────────────────────────────────────────

resource "aws_ecr_repository" "ml_images" {
  name                 = "${var.project_name}-${var.environment}"
  image_tag_mutability = "MUTABLE"
  force_delete         = var.environment == "dev"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = {
    Name = "${var.project_name}-ecr"
  }
}

resource "aws_ecr_lifecycle_policy" "cleanup" {
  repository = aws_ecr_repository.ml_images.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep only last 10 images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 10
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

# ─── Step Functions ──────────────────────────────────────────────────────────

module "step_functions" {
  source = "./modules/step_functions"

  project_name     = var.project_name
  environment      = var.environment
  sagemaker_role   = module.sagemaker.execution_role_arn
  sns_topic_arn    = aws_sns_topic.pipeline_notifications.arn
  data_bucket_name = module.s3.data_bucket_name
}

# ─── SNS Notifications ──────────────────────────────────────────────────────

resource "aws_sns_topic" "pipeline_notifications" {
  name = "${var.project_name}-${var.environment}-notifications"

  tags = {
    Name = "Pipeline Notifications"
  }
}

resource "aws_sns_topic_subscription" "email" {
  count     = var.notification_email != "" ? 1 : 0
  topic_arn = aws_sns_topic.pipeline_notifications.arn
  protocol  = "email"
  endpoint  = var.notification_email
}

# ─── CloudWatch ──────────────────────────────────────────────────────────────

resource "aws_cloudwatch_log_group" "pipeline" {
  name              = "/aws/sagemaker/${var.project_name}-${var.environment}"
  retention_in_days = var.log_retention_days

  tags = {
    Name = "Pipeline Logs"
  }
}

resource "aws_cloudwatch_metric_alarm" "training_failures" {
  alarm_name          = "${var.project_name}-training-failures"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "TrainingJobFailures"
  namespace           = "SageMaker/MLOps"
  period              = 300
  statistic           = "Sum"
  threshold           = 0
  alarm_description   = "Alarm when SageMaker training jobs fail"
  alarm_actions       = [aws_sns_topic.pipeline_notifications.arn]
}
