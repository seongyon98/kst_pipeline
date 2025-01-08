import boto3
import os
from config import S3_BUCKET_NAME
import logging

logger = logging.getLogger(__name__)

def create_s3_client():
    """
    S3 클라이언트를 생성하여 반환.
    """
    try:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION"),
        )
        logger.info("S3 클라이언트 생성 완료.")
        return s3_client
    except Exception as e:
        logger.error(f"S3 클라이언트 생성 실패: {e}")
        raise

def download_image_from_s3(s3_client, bucket_name, object_key):
    """
    S3에서 이미지를 다운로드하여 /tmp에 저장.

    Parameters:
    - s3_client: 생성된 S3 클라이언트
    - bucket_name: S3 버킷 이름
    - object_key: S3 객체 키 (파일 경로)
    
    Returns:
    - local_download_path: 로컬에 다운로드된 파일 경로
    """
    try:
        local_download_path = f"/tmp/{os.path.basename(object_key)}"
        s3_client.download_file(bucket_name, object_key, local_download_path)
        logger.info(f"S3에서 {object_key}를 {local_download_path}로 다운로드 완료.")
        return local_download_path
    except Exception as e:
        logger.error(f"S3에서 이미지 다운로드 실패: {e}")
        raise

def upload_cropped_images_to_s3(s3_client, local_dir, bucket_name, object_key_prefix):
    """
    로컬 디렉토리에서 크롭된 이미지를 S3에 업로드.

    Parameters:
    - s3_client: 생성된 S3 클라이언트
    - local_dir: 로컬 디렉토리 경로 (/tmp)
    - bucket_name: S3 버킷 이름
    - object_key_prefix: 원본 이미지 경로와 동일한 접두어
    """
    try:
        for file_name in os.listdir(local_dir):
            local_file_path = os.path.join(local_dir, file_name)
            s3_object_key = os.path.join(object_key_prefix, file_name)
            s3_client.upload_file(local_file_path, bucket_name, s3_object_key)
            logger.info(f"로컬 파일 {local_file_path}를 S3 {s3_object_key}에 업로드 완료.")
    except Exception as e:
        logger.error(f"S3에 크롭된 이미지 업로드 실패: {e}")
        raise
