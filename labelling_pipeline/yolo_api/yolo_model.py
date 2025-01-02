import os
import cv2
import numpy as np
import boto3
from craft_text_detector import Craft
from ultralytics import YOLO
from io import BytesIO

import boto3
import os
from io import BytesIO
from ultralytics import YOLO
from craft_text_detector import Craft

# -----------------------------------------------------------
# S3에서 모델 파일을 다운로드하여 사용
# -----------------------------------------------------------
S3_BUCKET_NAME = "big9-project-02-model-bucket"
YOLO_MODEL_PATH = "yolov8_text_nontext.pt"  # S3 경로
LOCAL_YOLO_MODEL_PATH = "./models/yolov8_text_nontext.pt"  # 로컬 경로에 YOLO 모델 저장

S3_IMAGE_BUCKET = "big9-project-02-question-bucket"
S3_IMAGE_PATH = "image/P3_1_01_21114_49495.png"  # 이미지 S3 경로

from dotenv import load_dotenv

load_dotenv(override=True)

# S3 클라이언트 생성
s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
)

# CRAFT 모델 초기화
craft = Craft(
    output_dir="./Model/Yolov8/Result/processed",  # 상대 경로로 변경
    crop_type="box",
)


def download_file_from_s3(bucket_name, file_key, local_path):
    """
    S3에서 파일을 다운로드하여 로컬에 저장
    :param bucket_name: S3 버킷 이름
    :param file_key: S3 경로
    :param local_path: 로컬 경로
    """
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)  # 디렉토리 생성
        s3_client.download_file(bucket_name, file_key, local_path)
        print(f"Downloaded: {file_key} to {local_path}")
    except Exception as e:
        print(f"Error downloading {file_key}: {e}")
        raise


def load_yolo_model():
    """
    YOLO 모델을 로컬에 저장 후 로드
    """
    print("Downloading YOLO model from S3...")
    download_file_from_s3(S3_BUCKET_NAME, YOLO_MODEL_PATH, LOCAL_YOLO_MODEL_PATH)
    print("Loading YOLO model...")
    return YOLO(LOCAL_YOLO_MODEL_PATH)


# YOLO 모델 로드
yolo_model = load_yolo_model()


def save_coordinates(coordinates, file_path):
    """텍스트 좌표를 저장"""
    with open(file_path, "w", encoding="utf-8") as f:
        for pts in coordinates:
            f.write(",".join(map(str, pts.flatten())) + "\n")


def save_failed_boxes(failed_boxes, file_path):
    """CRAFT 실패 영역 정보 저장"""
    with open(file_path, "w", encoding="utf-8") as f:
        for box in failed_boxes:
            f.write(
                f"Class: {box['class']}, Confidence: {box['conf']:.2f}, BBox: {box['bbox']}\n"
            )


def process_image_with_craft(image_path):
    """CRAFT로 텍스트 영역을 감지 후 (boxes, score_text 등) 리턴"""
    try:
        craft_result = craft.detect_text(image_path)
        return craft_result
    except Exception as e:
        print(f"CRAFT Error: {e}")
        return None


def process_image_with_yolo_and_craft(image, target_classes=None, conf_thresh=0.5):
    """YOLO로 텍스트 감지 후 CRAFT 수행"""
    if target_classes is None:
        target_classes = ["text"]  # 본인의 YOLO 클래스명

    height, width = image.shape[:2]
    original_image = image.copy()

    results = yolo_model.predict(image, conf=conf_thresh)
    if len(results) == 0 or len(results[0].boxes) == 0:
        print(f"No objects detected by YOLO. Skipping...")
        return [], []

    boxes = results[0].boxes
    all_text_boxes = []
    failed_boxes_info = []

    for box in boxes:
        cls_id = int(box.cls[0].item())  # 클래스 인덱스
        cls_conf = float(box.conf[0].item())  # confidence
        cls_name = results[0].names[cls_id]

        if cls_name in target_classes and cls_conf >= conf_thresh:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)

            # YOLO로 잡은 영역 Crop (원본 이미지에서 크롭)
            cropped_region = image[y1:y2, x1:x2]

            # CRAFT 수행
            try:
                craft_result = craft.detect_text(cropped_region)
                text_bboxes = [
                    np.array(pt).astype(np.int32)
                    for pt in craft_result["boxes"]
                    if pt is not None and len(pt) > 0
                ]

                for pts in text_bboxes:
                    pts[:, 0] += x1
                    pts[:, 1] += y1
                    all_text_boxes.append(pts)

            except Exception:
                failed_boxes_info.append(
                    {"class": cls_name, "conf": cls_conf, "bbox": (x1, y1, x2, y2)}
                )

    return all_text_boxes, failed_boxes_info
