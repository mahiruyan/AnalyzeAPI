import os
import boto3
from botocore.client import Config
from datetime import datetime, timedelta, timezone


def _s3_client():
    region = os.getenv("S3_REGION", "us-east-1")
    return boto3.client(
        "s3",
        region_name=region,
        config=Config(s3={"addressing_style": "virtual"}),
    )


def generate_presigned_url(key: str, expires_minutes: int = 60) -> str:
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        raise RuntimeError("S3_BUCKET belirtilmeli")
    client = _s3_client()
    url = client.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires_minutes * 60,
    )
    return url


def put_bytes(key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        raise RuntimeError("S3_BUCKET belirtilmeli")
    client = _s3_client()
    client.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)


def generate_presigned_put_url(key: str, content_type: str = "application/octet-stream", expires_minutes: int = 60) -> str:
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        raise RuntimeError("S3_BUCKET belirtilmeli")
    client = _s3_client()
    return client.generate_presigned_url(
        ClientMethod="put_object",
        Params={"Bucket": bucket, "Key": key, "ContentType": content_type},
        ExpiresIn=expires_minutes * 60,
    )


