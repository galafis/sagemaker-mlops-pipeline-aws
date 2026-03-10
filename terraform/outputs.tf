###############################################################################
# Outputs
###############################################################################

output "sagemaker_role_arn" {
  description = "ARN of the SageMaker execution role"
  value       = module.sagemaker.execution_role_arn
}

output "data_bucket_name" {
  description = "Name of the S3 data bucket"
  value       = module.s3.data_bucket_name
}

output "model_bucket_name" {
  description = "Name of the S3 model artifacts bucket"
  value       = module.s3.model_bucket_name
}

output "ecr_repository_url" {
  description = "ECR repository URL for ML container images"
  value       = aws_ecr_repository.ml_images.repository_url
}

output "step_functions_arn" {
  description = "ARN of the Step Functions state machine"
  value       = module.step_functions.state_machine_arn
}

output "sns_topic_arn" {
  description = "ARN of the SNS notification topic"
  value       = aws_sns_topic.pipeline_notifications.arn
}

output "cloudwatch_log_group" {
  description = "CloudWatch log group for pipeline logs"
  value       = aws_cloudwatch_log_group.pipeline.name
}
