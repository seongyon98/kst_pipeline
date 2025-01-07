from typing import List
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import os
import logging
import cv2
import numpy as np
from yolo_model import process_image_with_yolo_and_craft
from config import CROPPED_IMAGES_DIR

# FastAPI 인스턴스 생성
app = FastAPI(title="YOLO+CRAFT Image Cropping API")

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 응답 모델 정의
class CroppedImageResponse(BaseModel):
    message: str
    cropped_image_paths: List[str]


@app.post("/crop_image/", response_model=CroppedImageResponse)
async def crop_image(file: UploadFile = File(...)):
    """
    업로드된 이미지에서 YOLO와 CRAFT를 이용해 텍스트 영역을 감지하고 크롭된 이미지를 저장합니다.
    """
    try:
        # 업로드된 파일 읽기
        image_data = await file.read()
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")

        # 이미지 이름 추출 (확장자 제외)
        image_name = os.path.splitext(file.filename)[0]

        # 저장 디렉터리 설정 (이미지마다 별도의 디렉터리 생성)
        save_dir = os.path.join(CROPPED_IMAGES_DIR, image_name)
        os.makedirs(save_dir, exist_ok=True)

        # YOLO+CRAFT를 이용해 텍스트 감지 및 크롭된 이미지 저장
        cropped_image_paths = process_image_with_yolo_and_craft(image, image_name, save_dir)

        if not cropped_image_paths:
            return {"message": "No text areas detected", "cropped_image_paths": []}

        return {"message": "Cropping completed", "cropped_image_paths": cropped_image_paths}

    except HTTPException as he:
        logger.error(f"HTTP Exception: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Error during image processing: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
