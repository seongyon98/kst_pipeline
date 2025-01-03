import boto3
import os
import tempfile

# S3 설정
s3_client = boto3.client("s3")


# 3. S3에서 이미지 목록 가져오기
def list_images_in_s3(bucket_name, prefix):
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        if "Contents" not in response:
            raise ValueError("No files found in the specified bucket and prefix.")
        files = [
            obj["Key"]
            for obj in response["Contents"]
            if obj["Key"].endswith((".png", ".jpg"))
        ]
        print(f"Found {len(files)} image files.")
        return files
    except Exception as e:
        print(f"[ERROR] Failed to list images from S3: {e}")
        raise e


# 4. S3에서 이미지 다운로드
def download_image_from_s3(bucket_name, s3_key, local_path):
    """
    S3에서 지정된 이미지를 local_path로 다운로드합니다.
    """
    s3 = boto3.client("s3")
    try:
        # local_path에 직접 다운로드
        with open(local_path, "wb") as f:
            s3.download_fileobj(bucket_name, s3_key, f)
        print(f"Image downloaded from S3: {s3_key} to {local_path}")
    except Exception as e:
        print(f"[ERROR] Failed to download image {s3_key} to {local_path}: {e}")
        raise
