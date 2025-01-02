from fastapi import FastAPI, File, UploadFile
from typing import List
from pydantic import BaseModel
import shutil
import os
import cv2
import boto3
from io import BytesIO
import numpy as np
from yolo_model import (
    process_image_with_yolo_and_craft,
    process_image_with_craft,
    save_coordinates,
    save_failed_boxes,
)

app = FastAPI()


class CoordinatesResponse(BaseModel):
    coordinates: List[List[int]]


# S3 클라이언트 생성
s3_client = boto3.client("s3")


def download_image_from_s3(bucket_name, file_key):
    """S3에서 이미지를 다운로드"""
    file_obj = BytesIO()
    s3_client.download_fileobj(bucket_name, file_key, file_obj)
    file_obj.seek(0)
    img_array = np.asarray(bytearray(file_obj.read()), dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return image


@app.post("/extract_bboxes/")
async def extract_bboxes_from_image():
    try:
        # S3에서 이미지 다운로드
        image = download_image_from_s3(
            "big9-project-02-question-bucket", "image/P3_1_01_21114_49495.png"
        )
        if image is None:
            return {"error": "Failed to load image from S3"}

        # CRAFT 또는 YOLO+CRAFT 결과 얻기
        craft_result = process_image_with_craft(image)
        coordinates = []
        if craft_result is not None and len(craft_result["boxes"]) > 0:
            coordinates = [
                np.array(box).astype(np.int32)
                for box in craft_result["boxes"]
                if box is not None and len(box) > 0
            ]
            save_coordinates(coordinates, "coordinates.txt")
        else:
            all_text_boxes, failed_boxes = process_image_with_yolo_and_craft(image)
            save_coordinates(all_text_boxes, "coordinates.txt")
            if failed_boxes:
                save_failed_boxes(failed_boxes, "failed_boxes.txt")

        return {
            "message": "Text detection completed successfully",
            "coordinates": coordinates,
        }

    except Exception as e:
        return {"error": str(e)}
