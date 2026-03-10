"""AWS service helper utilities."""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AWSConfig:
    """AWS connection configuration."""

    region: str = "us-east-1"
    profile: Optional[str] = None
    role_arn: Optional[str] = None
    endpoint_url: Optional[str] = None

    @classmethod
    def from_env(cls) -> "AWSConfig":
        return cls(
            region=os.getenv("AWS_REGION", "us-east-1"),
            profile=os.getenv("AWS_PROFILE"),
            role_arn=os.getenv("AWS_ROLE_ARN"),
            endpoint_url=os.getenv("AWS_ENDPOINT_URL"),
        )


def get_boto3_session(config: Optional[AWSConfig] = None):
    """Create a boto3 session with the given configuration."""
    import boto3

    config = config or AWSConfig.from_env()

    session_kwargs: Dict[str, Any] = {"region_name": config.region}
    if config.profile:
        session_kwargs["profile_name"] = config.profile

    session = boto3.Session(**session_kwargs)

    if config.role_arn:
        sts = session.client("sts")
        assumed = sts.assume_role(
            RoleArn=config.role_arn,
            RoleSessionName="sagemaker-pipeline-session",
        )
        credentials = assumed["Credentials"]
        session = boto3.Session(
            aws_access_key_id=credentials["AccessKeyId"],
            aws_secret_access_key=credentials["SecretAccessKey"],
            aws_session_token=credentials["SessionToken"],
            region_name=config.region,
        )

    return session


def get_s3_client(config: Optional[AWSConfig] = None):
    """Get an S3 client."""
    session = get_boto3_session(config)
    kwargs = {}
    if config and config.endpoint_url:
        kwargs["endpoint_url"] = config.endpoint_url
    return session.client("s3", **kwargs)


def upload_to_s3(
    local_path: str,
    bucket: str,
    key: str,
    config: Optional[AWSConfig] = None,
) -> str:
    """Upload a file to S3 and return the S3 URI."""
    s3 = get_s3_client(config)
    s3.upload_file(local_path, bucket, key)
    uri = f"s3://{bucket}/{key}"
    logger.info(f"Uploaded {local_path} to {uri}")
    return uri


def download_from_s3(
    bucket: str,
    key: str,
    local_path: str,
    config: Optional[AWSConfig] = None,
) -> str:
    """Download a file from S3."""
    s3 = get_s3_client(config)
    s3.download_file(bucket, key, local_path)
    logger.info(f"Downloaded s3://{bucket}/{key} to {local_path}")
    return local_path
