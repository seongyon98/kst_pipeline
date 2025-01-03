import os
import cv2
import numpy as np
import boto3
from craft_text_detector import Craft
from ultralytics import YOLO
from io import BytesIO
import asyncio


# CRAFT 모델 초기화
craft = Craft(
    output_dir="./Model/Yolov8/Result/processed",  # 상대 경로로 변경
    crop_type="box",
)


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
