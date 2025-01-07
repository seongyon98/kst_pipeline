from typing import List
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import os
import logging
import cv2
import numpy as np
from yolo_model import process_image_with_yolo_and_craft
from s3_utils import create_s3_client, download_image_from_s3, upload_cropped_images_to_s3
from config import S3_BUCKET_NAME

# FastAPI 인스턴스 생성
app = FastAPI(title="YOLO+CRAFT S3 Image Cropping API")

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# S3 클라이언트 생성
s3_client = create_s3_client()

# 응답 모델 정의
class CroppedImageResponse(BaseModel):
    message: str
    cropped_image_paths: List[str]


@app.post("/crop_image/", response_model=CroppedImageResponse)
async def crop_image(object_key: str = Query(..., description="S3에 저장된 이미지의 객체 키")):
    """
    S3에 저장된 이미지 경로를 받아 YOLO와 CRAFT를 이용해 텍스트 영역을 감지하고,
    크롭된 이미지를 S3에 업로드합니다.
    """
    try:
        # 임시 다운로드 및 처리 디렉토리 설정
        image_name = os.path.splitext(os.path.basename(object_key))[0]
        download_path = f"/tmp/{image_name}.png"
        save_dir = f"/tmp/{image_name}"
        os.makedirs(save_dir, exist_ok=True)

        # S3에서 이미지 다운로드
        download_image_from_s3(s3_client, S3_BUCKET_NAME, object_key, download_path)

        # 이미지 로드
        image = cv2.imread(download_path)
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to load image from S3")

        # YOLO+CRAFT로 텍스트 감지 및 크롭된 이미지 저장
        cropped_image_paths = process_image_with_yolo_and_craft(image, image_name, save_dir)

        if not cropped_image_paths:
            return {"message": "No text areas detected", "cropped_image_paths": []}

        # S3에 크롭된 이미지 업로드
        upload_cropped_images_to_s3(s3_client, save_dir, S3_BUCKET_NAME, f"cropped/{image_name}")

        return {"message": "Cropping completed and uploaded to S3", "cropped_image_paths": cropped_image_paths}

    except HTTPException as he:
        logger.error(f"HTTP Exception: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Error during image processing: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
