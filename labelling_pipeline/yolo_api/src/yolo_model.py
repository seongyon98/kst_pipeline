# src/yolo_model.py
import os
import cv2
import numpy as np
from craft_text_detector import Craft
from ultralytics import YOLO
from itertools import chain
from config import OUTPUT_DIR, CROPPED_IMAGES_DIR, S3_BUCKET_NAME
from s3_utils import create_s3_client
import logging

logger = logging.getLogger(__name__)

# CRAFT 모델 초기화
craft = Craft(
    output_dir=OUTPUT_DIR,  # 설정 파일에서 가져온 경로 사용
    crop_type="box",
)

# YOLO 모델 변수 초기화
yolo_model = None

def initialize_models(yolo_model_path):
    """YOLO 모델을 로드합니다."""
    global yolo_model
    try:
        yolo_model = YOLO(yolo_model_path)
        logger.info("YOLO 모델 로드 완료.")
    except Exception as e:
        logger.error(f"YOLO 모델 로드 실패: {e}")
        raise

def download_image_from_s3(bucket_name, s3_key, local_path):
    """
    S3에서 지정된 이미지를 local_path로 다운로드하고 이미지를 반환합니다.
    """
    s3_client = create_s3_client()
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            s3_client.download_fileobj(bucket_name, s3_key, f)
        logger.info(f"Image downloaded from S3: {s3_key} to {local_path}")
        image = cv2.imread(local_path)
        return image
    except Exception as e:
        logger.error(f"[ERROR] Failed to download image {s3_key} to {local_path}: {e}")
        raise

def convert_numpy_to_python(data):
    """numpy 배열이나 numpy 타입을 Python 기본 타입으로 변환"""
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.generic):
        return data.item()
    return data

def flatten(data):
    """주어진 data가 numpy 배열일 경우 flatten을 호출하고, list일 경우 평탄화"""
    if isinstance(data, np.ndarray):
        return data.flatten()
    elif isinstance(data, list):
        return list(chain.from_iterable(data))
    return data

def save_coordinates(coordinates, file_path):
    """텍스트 좌표를 저장"""
    with open(file_path, "w", encoding="utf-8") as f:
        for pts in coordinates:
            f.write(",".join(map(str, pts.flatten())) + "\n")

def save_failed_boxes(failed_boxes, file_path):
    """CRAFT 실패 영역 정보를 저장"""
    with open(file_path, "w", encoding="utf-8") as f:
        for box in failed_boxes:
            f.write(f"Class: {box['class']}, Confidence: {box['conf']:.2f}, BBox: {box['bbox']}\n")

def save_cropped_image(image, bbox, save_dir, image_name, box_id):
    """크롭된 이미지를 저장"""
    x1, y1, x2, y2 = bbox
    cropped_image = image[y1:y2, x1:x2]
    cropped_image_path = os.path.join(save_dir, f"{image_name}_box_{box_id}.png")
    cv2.imwrite(cropped_image_path, cropped_image)
    return cropped_image_path

def process_image_with_craft(image_path):
    """CRAFT로 텍스트 영역을 감지 후 결과 반환"""
    try:
        craft_result = craft.detect_text(image_path)
        return craft_result
    except Exception as e:
        logger.error(f"CRAFT Error: {e}")
        return None

def process_image_with_yolo_and_craft(yolo_model, image, image_name, target_classes=None, conf_thresh=0.5):
    """YOLO로 텍스트 감지 후 CRAFT 수행 및 크롭된 이미지 저장"""
    if target_classes is None:
        target_classes = ["text"]  # 본인의 YOLO 클래스명

    height, width = image.shape[:2]
    original_image = image.copy()

    results = yolo_model.predict(image, conf=conf_thresh)
    if len(results) == 0 or len(results[0].boxes) == 0:
        logger.info("No objects detected by YOLO. Skipping...")
        return [], [], []

    boxes = results[0].boxes
    all_text_boxes = []
    failed_boxes_info = []
    cropped_image_paths = []

    for idx, box in enumerate(boxes):
        cls_id = int(box.cls[0].item())  # 클래스 인덱스
        cls_conf = float(box.conf[0].item())  # confidence
        cls_name = results[0].names[cls_id]

        if cls_name in target_classes and cls_conf >= conf_thresh:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)

            # YOLO로 잡은 영역 Crop (원본 이미지에서 크롭)
            cropped_image_path = save_cropped_image(
                image, (x1, y1, x2, y2), CROPPED_IMAGES_DIR, image_name, idx
            )
            cropped_image_paths.append(cropped_image_path)

            # CRAFT 수행
            try:
                craft_result = craft.detect_text(cropped_image_path)
                text_bboxes = [
                    np.array(pt).astype(np.int32)
                    for pt in craft_result["boxes"]
                    if pt is not None and len(pt) > 0
                ]

                for pts in text_bboxes:
                    pts[:, 0] += x1
                    pts[:, 1] += y1
                    all_text_boxes.append(pts)

            except Exception as e:
                logger.error(f"CRAFT 수행 중 오류: {e}")
                failed_boxes_info.append(
                    {"class": cls_name, "conf": cls_conf, "bbox": (x1, y1, x2, y2)}
                )

    return all_text_boxes, failed_boxes_info, cropped_image_paths
