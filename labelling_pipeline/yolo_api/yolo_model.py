import os
import boto3
import tempfile
from ultralytics import YOLO

s3_client = boto3.client("s3")
MODEL_BUCKET_NAME = os.getenv("MODEL_BUCKET_NAME")
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH")
LOCAL_YOLO_PATH = "/tmp/temp_yolo.pt"


def load_yolo_model():
    try:
        # 임시 디렉토리에 파일 생성
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as temp_file:
            LOCAL_YOLO_PATH = temp_file.name

        print(f"[INFO] Temporary file created at {LOCAL_YOLO_PATH}")

        # S3에서 YOLO 모델 다운로드
        s3_client.download_file(MODEL_BUCKET_NAME, YOLO_MODEL_PATH, LOCAL_YOLO_PATH)
        print(f"[INFO] YOLO model downloaded to {LOCAL_YOLO_PATH}")

        # YOLO 모델 로드
        model = YOLO(LOCAL_YOLO_PATH)

        # 임시 파일 삭제
        os.remove(LOCAL_YOLO_PATH)
        print(f"[INFO] Temporary file {LOCAL_YOLO_PATH} deleted")

        return model

    except Exception as e:
        print(f"[ERROR] Failed to load YOLO model: {e}")
        raise e
