import os
import cv2
import numpy as np
import boto3
from craft_text_detector import Craft
from ultralytics import YOLO
from io import BytesIO
import asyncio
from itertools import chain


# CRAFT 모델 초기화
craft = Craft(
    output_dir="./Model/Yolov8/Result/processed",  # 상대 경로로 변경
    crop_type="box",
)

# S3 클라이언트 생성
s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
)


# S3에서 이미지 목록 가져오기
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


# S3에서 이미지 다운로드
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


# numpy 데이터를 기본 Python 타입으로 변환하는 함수
def convert_numpy_to_python(data):
    """numpy 배열이나 numpy 타입을 Python 기본 타입으로 변환"""
    if isinstance(data, np.ndarray):
        return data.tolist()  # numpy 배열을 리스트로 변환
    elif isinstance(data, np.generic):  # numpy 데이터 타입인 경우
        return data.item()  # Python 기본 데이터 타입으로 변환
    return data


# 리스트 평탄화 함수
def flatten(data):
    """주어진 data가 numpy 배열일 경우 flatten을 호출하고, list일 경우 list comprehension을 사용해 평평하게 만듦"""
    if isinstance(data, np.ndarray):  # numpy 배열인 경우
        return data.flatten()
    elif isinstance(data, list):  # list인 경우
        return list(
            chain.from_iterable(data)
        )  # itertools.chain을 사용하여 평평하게 만들기
    return data


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


def process_image_with_yolo_and_craft(
    yolo_model, image, target_classes=None, conf_thresh=0.5
):
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
