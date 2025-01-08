import os
import cv2
import numpy as np
from craft_text_detector import Craft
from ultralytics import YOLO
from config import LOCAL_YOLO_MODEL_PATH, S3_BUCKET_NAME
from s3_utils import upload_cropped_images_to_s3
import logging

logger = logging.getLogger(__name__)

# CRAFT 모델 초기화
craft = Craft(
    output_dir="/tmp",  # 로컬 임시 디렉토리 사용
    crop_type="box",  # box 형태로 크롭
)
logger.info("CRAFT 모델 초기화 완료")

# YOLO 모델 초기화
yolo_model = YOLO(LOCAL_YOLO_MODEL_PATH)
logger.info(f"YOLO 모델 초기화 완료: {LOCAL_YOLO_MODEL_PATH}")


def save_cropped_image(image, bbox, local_dir, image_name, box_id, prefix):
    """
    크롭된 이미지를 로컬에 저장하고 경로를 반환.
    
    Parameters:
    - image: 원본 이미지 (numpy 배열)
    - bbox: 크롭할 바운딩 박스 좌표 (x1, y1, x2, y2)
    - local_dir: 로컬 디렉토리 경로
    - image_name: 이미지 이름 (확장자 제외)
    - box_id: 바운딩 박스 ID (정수)
    - prefix: YOLO 또는 CRAFT 구분을 위한 접두어
    
    Returns:
    - local_cropped_path: 저장된 크롭 이미지의 로컬 경로 (문자열)
    """
    x1, y1, x2, y2 = bbox
    cropped_image = image[y1:y2, x1:x2]
    local_cropped_path = os.path.join(local_dir, f"{image_name}_{prefix}_box_{box_id}.png")
    cv2.imwrite(local_cropped_path, cropped_image)
    
    logger.info(f"Saved cropped image: {local_cropped_path}")
    return local_cropped_path


def process_image_with_craft(image, image_name, local_dir):
    """
    CRAFT로 텍스트 영역을 감지 후 크롭된 이미지를 저장.
    
    Parameters:
    - image: 원본 이미지 (numpy 배열)
    - image_name: 이미지 이름 (확장자 제외)
    - local_dir: 로컬 디렉토리 경로
    
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
                    image, (x1, y1, x2, y2), local_dir, image_name, idx, "craft"
                )
                cropped_image_paths.append(cropped_image_path)
    except Exception as e:
        logger.error(f"CRAFT 수행 중 오류: {e}")
    
    return cropped_image_paths


def process_image_with_yolo_and_craft(s3_client, bucket_name, object_key, download_path):
    """
    YOLO와 CRAFT로 텍스트 영역을 감지 후 크롭된 이미지를 S3에 업로드.
    
    Parameters:
    - s3_client: S3 클라이언트 객체
    - bucket_name: S3 버킷 이름
    - object_key: S3에서 가져온 원본 이미지의 객체 키
    - download_path: 로컬에 다운로드된 이미지 경로
    """
    image = cv2.imread(download_path)
    if image is None:
        logger.error(f"이미지를 로드할 수 없습니다: {download_path}")
        return

    image_name = os.path.splitext(os.path.basename(object_key))[0]
    local_dir = os.path.join("/tmp", image_name)
    os.makedirs(local_dir, exist_ok=True)

    # 1차: CRAFT로 텍스트 영역 감지
    cropped_image_paths = process_image_with_craft(image, image_name, local_dir)

    # 2차: YOLO로 추가 텍스트 영역 감지
    if not cropped_image_paths:
        logger.info("CRAFT에서 감지된 텍스트가 없습니다. YOLO로 대체합니다.")
        height, width = image.shape[:2]
        results = yolo_model.predict(image, conf=0.5)
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width - 1, x2), min(height - 1, y2)

                # 크롭된 이미지를 저장
                cropped_image_path = save_cropped_image(
                    image, (x1, y1, x2, y2), local_dir, image_name, idx, "yolo"
                )
                cropped_image_paths.append(cropped_image_path)

    # S3에 크롭된 이미지 업로드
    s3_object_key_prefix = os.path.join(os.path.dirname(object_key), image_name)
    upload_cropped_images_to_s3(s3_client, local_dir, bucket_name, s3_object_key_prefix)
    logger.info(f"YOLO+CRAFT 처리 완료 및 S3 업로드 완료: {s3_object_key_prefix}")
