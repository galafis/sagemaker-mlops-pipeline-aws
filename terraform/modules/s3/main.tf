###############################################################################
# S3 Module — Data and Model Artifact Buckets
###############################################################################

resource "aws_s3_bucket" "data" {
  bucket        = "${var.project_name}-${var.environment}-data"
  force_destroy = var.environment == "dev"

  tags = {
    Name    = "ML Data Bucket"
    Purpose = "Raw and processed training data"
  }
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket                  = aws_s3_bucket.data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "data" {
  bucket = aws_s3_bucket.data.id

  rule {
    id     = "archive-old-data"
    status = "Enabled"

    transition {
      days          = 90
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 180
      storage_class = "GLACIER"
    }
  }
}

# ─── Model Artifacts Bucket ─────────────────────────────────────────────────

resource "aws_s3_bucket" "models" {
  bucket        = "${var.project_name}-${var.environment}-models"
  force_destroy = var.environment == "dev"

  tags = {
    Name    = "ML Model Artifacts"
    Purpose = "Trained model artifacts and evaluation results"
  }
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "models" {
  bucket = aws_s3_bucket.models.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "models" {
  bucket                  = aws_s3_bucket.models.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ─── Outputs ─────────────────────────────────────────────────────────────────

variable "project_name" {
  type = string
}

variable "environment" {
  type = string
}

output "data_bucket_name" {
  value = aws_s3_bucket.data.bucket
}

output "data_bucket_arn" {
  value = aws_s3_bucket.data.arn
}

output "model_bucket_name" {
  value = aws_s3_bucket.models.bucket
}

output "model_bucket_arn" {
  value = aws_s3_bucket.models.arn
}
