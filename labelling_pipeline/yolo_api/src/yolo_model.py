import os
import cv2
import numpy as np
from craft_text_detector import Craft
from ultralytics import YOLO
from config import LOCAL_YOLO_MODEL_PATH, OUTPUT_DIR, CROPPED_IMAGES_DIR
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CRAFT 모델 초기화
craft = Craft(
    output_dir=OUTPUT_DIR,
    crop_type="box",  # box 형태로 크롭
)
logger.info("CRAFT 모델 초기화 완료")

# YOLO 모델 초기화
yolo_model = YOLO(LOCAL_YOLO_MODEL_PATH)
logger.info(f"YOLO 모델 초기화 완료: {LOCAL_YOLO_MODEL_PATH}")


def save_cropped_image(image, bbox, save_dir, image_name, box_id, prefix):
    """
    크롭된 이미지를 저장하고 경로를 반환.
    
    Parameters:
    - image: 원본 이미지 (numpy 배열)
    - bbox: 크롭할 바운딩 박스 좌표 (x1, y1, x2, y2)
    - save_dir: 크롭된 이미지를 저장할 디렉토리 경로
    - image_name: 이미지 이름 (확장자 제외)
    - box_id: 바운딩 박스 ID (정수)
    - prefix: YOLO 또는 CRAFT 구분을 위한 접두어
    
    Returns:
    - cropped_image_path: 저장된 크롭 이미지 경로 (문자열)
    """
    x1, y1, x2, y2 = bbox
    cropped_image = image[y1:y2, x1:x2]
    cropped_image_path = os.path.join(save_dir, f"{image_name}_{prefix}_box_{box_id}.png")
    cv2.imwrite(cropped_image_path, cropped_image)
    return cropped_image_path


def process_image_with_craft(image, image_name, save_dir):
    """
    CRAFT로 텍스트 영역을 감지 후 크롭된 이미지를 저장.

    Parameters:
    - image: 원본 이미지 (numpy 배열)
    - image_name: 이미지 이름 (확장자 제외)
    - save_dir: 크롭된 이미지를 저장할 디렉토리 경로

    Returns:
    - cropped_image_paths: 저장된 크롭 이미지 경로 리스트
    """
    cropped_image_paths = []
    try:
        craft_result = craft.detect_text(image)
        if craft_result and "boxes" in craft_result:
            for idx, box in enumerate(craft_result["boxes"]):
                x1, y1, x2, y2 = np.array(box).astype(int).flatten()
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image.shape[1] - 1, x2), min(image.shape[0] - 1, y2)

                # 크롭된 이미지를 저장
                cropped_image_path = save_cropped_image(
                    image, (x1, y1, x2, y2), save_dir, image_name, idx, "craft"
                )
                cropped_image_paths.append(cropped_image_path)
    except Exception as e:
        logger.error(f"CRAFT 수행 중 오류: {e}")
    
    return cropped_image_paths


def process_image_with_yolo_and_craft(image, image_name, save_dir, conf_thresh=0.5):
    """
    YOLO로 텍스트 감지 후 크롭된 이미지를 저장.

    Parameters:
    - image: 원본 이미지 (numpy 배열)
    - image_name: 이미지 이름 (확장자 제외)
    - save_dir: 크롭된 이미지를 저장할 디렉토리 경로
    - conf_thresh: YOLO confidence threshold

    Returns:
    - cropped_image_paths: 저장된 크롭 이미지 경로 리스트
    """
    cropped_image_paths = []
    height, width = image.shape[:2]

    # YOLO 모델로 텍스트 영역 감지
    results = yolo_model.predict(image, conf=conf_thresh)
    if len(results) == 0 or len(results[0].boxes) == 0:
        logger.info("No objects detected by YOLO.")
        return cropped_image_paths

    boxes = results[0].boxes

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width - 1, x2), min(height - 1, y2)

        # 크롭된 이미지를 저장
        cropped_image_path = save_cropped_image(
            image, (x1, y1, x2, y2), save_dir, image_name, idx, "yolo"
        )
        cropped_image_paths.append(cropped_image_path)

    return cropped_image_paths
