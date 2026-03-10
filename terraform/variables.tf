###############################################################################
# Input Variables
###############################################################################

variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project identifier used in resource naming"
  type        = string
  default     = "sagemaker-mlops"
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "notification_email" {
  description = "Email address for pipeline notifications"
  type        = string
  default     = ""
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 30
}

variable "training_instance_type" {
  description = "SageMaker training instance type"
  type        = string
  default     = "ml.m5.xlarge"
}

variable "endpoint_instance_type" {
  description = "SageMaker endpoint instance type"
  type        = string
  default     = "ml.m5.large"
}

variable "enable_spot_training" {
  description = "Enable managed spot training for cost savings"
  type        = bool
  default     = true
}
