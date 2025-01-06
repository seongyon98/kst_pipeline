# src/config.py
import os
from dotenv import load_dotenv
import logging

# 환경 변수 로드
load_dotenv(dotenv_path="/pipeline/.env", override=True)

# S3 설정
S3_BUCKET_NAME = os.getenv("MODEL_BUCKET_NAME")
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH")  # S3 경로
LOCAL_YOLO_MODEL_PATH = "./models/yolov8_text_nontext.pt"  # 로컬 경로에 YOLO 모델 저장

# 디렉토리 경로 설정
OUTPUT_DIR = "./models/Yolov8/Result/processed"
CROPPED_IMAGES_DIR = "./cropped_images/"
DOWNLOADS_DIR = "./downloads/"

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
