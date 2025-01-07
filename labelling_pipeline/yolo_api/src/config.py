import os
from dotenv import load_dotenv
import logging

# # 환경 변수 로드
# load_dotenv()

# 출력 디렉토리 경로 설정
OUTPUT_DIR = "./output"
CROPPED_IMAGES_DIR = "./cropped_images"
DOWNLOADS_DIR = "./downloads"
coordinates_dir = os.path.join(OUTPUT_DIR, "coordinates")
failed_boxes_dir = os.path.join(OUTPUT_DIR, "failed_boxes")

# YOLO 모델 경로
LOCAL_YOLO_MODEL_PATH = "./models/yolov8_text_nontext.pt"

# 디렉토리 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CROPPED_IMAGES_DIR, exist_ok=True)
os.makedirs(DOWNLOADS_DIR, exist_ok=True)
os.makedirs(coordinates_dir, exist_ok=True)
os.makedirs(failed_boxes_dir, exist_ok=True)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
