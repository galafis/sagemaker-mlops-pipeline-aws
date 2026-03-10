###############################################################################
# SageMaker Module — IAM Roles and Execution Permissions
###############################################################################

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# ─── SageMaker Execution Role ───────────────────────────────────────────────

resource "aws_iam_role" "sagemaker_execution" {
  name = "${var.project_name}-${var.environment}-sagemaker-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = {
    Name = "SageMaker Execution Role"
  }
}

resource "aws_iam_role_policy" "sagemaker_s3" {
  name = "sagemaker-s3-access"
  role = aws_iam_role.sagemaker_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket",
          "s3:DeleteObject",
        ]
        Resource = [
          var.data_bucket_arn,
          "${var.data_bucket_arn}/*",
          var.model_bucket_arn,
          "${var.model_bucket_arn}/*",
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy" "sagemaker_ecr" {
  name = "sagemaker-ecr-access"
  role = aws_iam_role.sagemaker_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchGetImage",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchCheckLayerAvailability",
        ]
        Resource = [var.ecr_repository_arn]
      },
      {
        Effect   = "Allow"
        Action   = ["ecr:GetAuthorizationToken"]
        Resource = ["*"]
      }
    ]
  })
}

resource "aws_iam_role_policy" "sagemaker_logs" {
  name = "sagemaker-cloudwatch-logs"
  role = aws_iam_role.sagemaker_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams",
        ]
        Resource = [
          "arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:log-group:/aws/sagemaker/*"
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_full" {
  role       = aws_iam_role.sagemaker_execution.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

# ─── Variables and Outputs ──────────────────────────────────────────────────

variable "project_name" {
  type = string
}

variable "environment" {
  type = string
}

variable "data_bucket_arn" {
  type = string
}

variable "model_bucket_arn" {
  type = string
}

variable "ecr_repository_arn" {
  type = string
}

output "execution_role_arn" {
  value = aws_iam_role.sagemaker_execution.arn
}
