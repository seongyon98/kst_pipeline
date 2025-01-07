import os
from dotenv import load_dotenv
import logging

# 환경 변수 로드
load_dotenv()

# S3 버킷 이름 및 경로 설정
S3_BUCKET_NAME = "big9-project-02-training-bucket"
S3_IMAGE_PREFIX = "test/images"

# YOLO 모델 경로 (로컬 경로에 YOLO 모델 파일이 있어야 함)
LOCAL_YOLO_MODEL_PATH = "./models/yolov8_text_nontext.pt"

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
