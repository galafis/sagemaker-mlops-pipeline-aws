###############################################################################
# Step Functions Module — ML Pipeline State Machine
###############################################################################

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# ─── Step Functions Execution Role ──────────────────────────────────────────

resource "aws_iam_role" "step_functions" {
  name = "${var.project_name}-${var.environment}-sfn-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "states.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy" "sfn_sagemaker" {
  name = "sfn-sagemaker-access"
  role = aws_iam_role.step_functions.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "sagemaker:CreateProcessingJob",
          "sagemaker:CreateTrainingJob",
          "sagemaker:CreateModel",
          "sagemaker:CreateEndpointConfig",
          "sagemaker:CreateEndpoint",
          "sagemaker:UpdateEndpoint",
          "sagemaker:CreateModelPackage",
          "sagemaker:DescribeProcessingJob",
          "sagemaker:DescribeTrainingJob",
          "sagemaker:DescribeEndpoint",
          "sagemaker:AddTags",
        ]
        Resource = ["*"]
      },
      {
        Effect   = "Allow"
        Action   = ["iam:PassRole"]
        Resource = [var.sagemaker_role]
      },
      {
        Effect   = "Allow"
        Action   = ["sns:Publish"]
        Resource = [var.sns_topic_arn]
      },
      {
        Effect = "Allow"
        Action = [
          "events:PutTargets",
          "events:PutRule",
          "events:DescribeRule",
        ]
        Resource = [
          "arn:aws:events:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:rule/StepFunctionsGetEventsForSageMaker*"
        ]
      }
    ]
  })
}

# ─── State Machine ──────────────────────────────────────────────────────────

resource "aws_sfn_state_machine" "ml_pipeline" {
  name     = "${var.project_name}-${var.environment}-pipeline"
  role_arn = aws_iam_role.step_functions.arn

  definition = templatefile("${path.module}/definition.json.tpl", {
    sagemaker_role = var.sagemaker_role
    data_bucket    = var.data_bucket_name
    sns_topic_arn  = var.sns_topic_arn
    region         = data.aws_region.current.name
    account_id     = data.aws_caller_identity.current.account_id
  })

  logging_configuration {
    log_destination        = "${aws_cloudwatch_log_group.sfn.arn}:*"
    include_execution_data = true
    level                  = "ALL"
  }

  tags = {
    Name = "ML Pipeline State Machine"
  }
}

resource "aws_cloudwatch_log_group" "sfn" {
  name              = "/aws/vendedlogs/states/${var.project_name}-${var.environment}"
  retention_in_days = 30
}

# ─── Variables and Outputs ──────────────────────────────────────────────────

variable "project_name" {
  type = string
}

variable "environment" {
  type = string
}

variable "sagemaker_role" {
  type = string
}

variable "sns_topic_arn" {
  type = string
}

variable "data_bucket_name" {
  type = string
}

output "state_machine_arn" {
  value = aws_sfn_state_machine.ml_pipeline.arn
}
