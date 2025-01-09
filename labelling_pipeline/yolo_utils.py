import os
import cv2
import boto3
import numpy as np
from craft_text_detector import Craft
from ultralytics import YOLO


# S3에서 YOLO 및 OCR 모델 다운로드
def download_model_from_s3(s3_client, bucket, s3_key, local_path):
    if os.path.exists(local_path):
        print(f"[INFO] Model file already exists: {local_path}. Skipping download.")
    else:
        print(f"[INFO] Downloading model '{s3_key}' from S3 bucket '{bucket}'...")
        s3_client.download_file(bucket, s3_key, local_path)
        print(f"[INFO] Model downloaded to: {local_path}")


# 모델 초기화 함수들
def initialize_yolo_model(yolo_local_path):
    return YOLO(yolo_local_path)


def initialize_craft_model(output_dir):
    return Craft(output_dir=output_dir, crop_type="box", export_extra=False)


# 디렉토리 설정
def create_directories(base_dir):
    output_dir = os.path.join(base_dir, "output", "craft")
    coordinates_dir = os.path.join(output_dir, "coordinates")
    failed_boxes_dir = os.path.join(output_dir, "failed_boxes")
    cropped_dir = os.path.join(output_dir, "cropped")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(coordinates_dir, exist_ok=True)
    os.makedirs(failed_boxes_dir, exist_ok=True)
    os.makedirs(cropped_dir, exist_ok=True)

    return output_dir, coordinates_dir, failed_boxes_dir, cropped_dir


def download_selected_images_from_s3(
    s3_client, bucket, image_s3_keys, local_image_folder
):
    print(f"[INFO] Retrieving selected images from s3://{bucket}...")

    if not image_s3_keys:
        print("[WARN] No image keys provided.")
        return []

    # 기존에 존재하는 파일 삭제 (모든 파일을 덮어쓰므로 기존 파일 삭제)
    for existing_file in os.listdir(local_image_folder):
        existing_file_path = os.path.join(local_image_folder, existing_file)
        if os.path.isfile(existing_file_path):
            os.remove(existing_file_path)
            print(f"[INFO] Removed existing file: {existing_file_path}")

    # 선택된 이미지 파일 다운로드
    downloaded_images = []
    for image_s3_key in image_s3_keys:
        file_name = os.path.basename(image_s3_key)
        local_path = os.path.join(local_image_folder, file_name)
        print(f"[INFO] Downloading image: s3://{bucket}/{image_s3_key} -> {local_path}")
        s3_client.download_file(bucket, image_s3_key, local_path)
        print(f"[INFO] Downloaded image to: {local_path}")
        downloaded_images.append(local_path)

    return downloaded_images


# CRAFT 및 YOLO를 이용한 이미지 처리 함수들
def save_coordinates(coordinates, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for pts in coordinates:
            f.write(",".join(map(str, pts.flatten())) + "\n")


def save_failed_boxes(failed_boxes, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for box in failed_boxes:
            f.write(
                f"Class: {box['class']}, Confidence: {box['conf']:.2f}, BBox: {box['bbox']}\n"
            )


def process_image_with_craft(image_path, craft_model):
    try:
        return craft_model.detect_text(image_path)
    except Exception as e:
        print(f"[ERROR] CRAFT Error: {e}")
        return None


def process_image_with_yolo_and_craft(
    image,
    file_name,
    target_classes=None,
    conf_thresh=0.5,
    yolo_model=None,
    craft_model=None,
):
    if target_classes is None:
        target_classes = ["text"]

    results = yolo_model.predict(image, conf=conf_thresh)
    if len(results) == 0 or len(results[0].boxes) == 0:
        print(f"[WARN] No objects detected by YOLO in {file_name}. Skipping...")
        return [], []

    boxes = results[0].boxes
    all_text_boxes = []
    failed_boxes_info = []

    height, width = image.shape[:2]

    for box in boxes:
        cls_id = int(box.cls[0].item())
        cls_conf = float(box.conf[0].item())
        cls_name = results[0].names[cls_id]

        if cls_name in target_classes and cls_conf >= conf_thresh:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)

            cropped_region = image[y1:y2, x1:x2]

            try:
                craft_result = craft_model.detect_text(cropped_region)
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
                print(f"[ERROR] CRAFT on cropped_region: {e}")
                failed_boxes_info.append(
                    {"class": cls_name, "conf": cls_conf, "bbox": (x1, y1, x2, y2)}
                )

    return all_text_boxes, failed_boxes_info


def save_cropped_image(cropped_image, cropped_image_path):
    if os.path.exists(cropped_image_path):
        print(f"[INFO] Already exists: {cropped_image_path}. Skipping save.")
    else:
        cv2.imwrite(cropped_image_path, cropped_image)
        print(f"[INFO] Saved cropped image: {cropped_image_path}")
