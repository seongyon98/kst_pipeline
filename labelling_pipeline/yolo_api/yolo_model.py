from pathlib import Path
import boto3
from ultralytics import YOLO
from dotenv import load_dotenv
import os
import tempfile
from PIL import Image
import io
import cv2
import numpy as np

# 환경 변수 로드
load_dotenv(override=True)

# S3 설정
s3_client = boto3.client("s3")
MODEL_BUCKET_NAME = os.getenv("MODEL_BUCKET_NAME")
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH")
IMAGE_BUCKET_NAME = os.getenv("IMAGE_BUCKET_NAME")
IMAGE_PREFIX = os.getenv("IMAGE_PREFIX")  # 이미지 경로(prefix)
LOCAL_YOLO_PATH = "/tmp/temp_yolo.pt"  # YOLO 모델 임시 저장 경로


# 1. YOLO 모델 로드
def load_yolo_model_from_s3():
    try:
        # 임시 디렉토리에 YOLO 모델 다운로드
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


# YOLO 모델을 사용해 바운딩 박스를 추출하는 함수
def upscale_image(image, scale=2):
    height, width = image.shape[:2]
    new_dim = (width * scale, height * scale)
    upscaled_image = cv2.resize(image, new_dim, interpolation=cv2.INTER_CUBIC)
    return upscaled_image


def enhance_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_image = cv2.merge((cl, a, b))
    final_image = cv2.cvtColor(enhanced_image, cv2.COLOR_LAB2BGR)
    return final_image


def preprocess_image(image):
    # 이미지 전처리 과정 (스케일 업, 대비 향상)
    image = upscale_image(image, scale=2)
    image = enhance_contrast(image)
    return image


def extract_bboxes(yolo_model, image):
    try:
        # 이미지를 numpy array로 변환
        open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # 이미지 전처리
        preprocessed_image = preprocess_image(open_cv_image)

        # YOLO 모델 추론
        results = yolo_model(preprocessed_image, conf=0.3, iou=0.4, verbose=False)

        bboxes = []
        for result in results[0].boxes.data:
            x1, y1, x2, y2, conf, cls = result
            if int(cls) == 0:  # 클래스가 0인 경우만 처리
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                bboxes.append(
                    [0, x_center.item(), y_center.item(), width.item(), height.item()]
                )

        # 바운딩 박스가 없는 경우 기본값 반환 (just 연결 테스트용!)
        if not bboxes:
            print("[WARNING] No bounding boxes found. Returning default bounding box.")
            bboxes.append([0, 0, 0, 0, 0])

        return bboxes

    except Exception as e:
        print(f"[ERROR] Failed to extract bounding boxes: {e}")
        return [[0, 0, 0, 0, 0]]  # 오류 발생 시 기본값 반환 (just 연결 테스트용!)
