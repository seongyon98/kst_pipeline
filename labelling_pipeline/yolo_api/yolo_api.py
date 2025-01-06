from typing import List
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import asyncio
import boto3
import os
from contextlib import asynccontextmanager
from io import BytesIO
import cv2
from ultralytics import YOLO
from dotenv import load_dotenv
from yolo_model import (
    convert_numpy_to_python,
    download_image_from_s3,
    flatten,
    process_image_with_yolo_and_craft,
    process_image_with_craft,
    save_coordinates,
    save_failed_boxes,
)


load_dotenv(dotenv_path="/pipeline/.env", override=True)

S3_BUCKET_NAME = os.getenv("MODEL_BUCKET_NAME")
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH")  # S3 경로
LOCAL_YOLO_MODEL_PATH = "./models/"  # 로컬 경로에 YOLO 모델 저장

# S3 클라이언트 생성
s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("서버 시작 중...")

    # 비동기적으로 모델 다운로드 및 로드
    try:
        # S3에서 모델 파일 다운로드
        os.makedirs(os.path.dirname(LOCAL_YOLO_MODEL_PATH), exist_ok=True)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: s3_client.download_file(
                S3_BUCKET_NAME, YOLO_MODEL_PATH, LOCAL_YOLO_MODEL_PATH
            ),
        )
        print(f"Downloaded: {YOLO_MODEL_PATH} to {LOCAL_YOLO_MODEL_PATH}")
        # 모델을 global로 설정
        global yolo_model

        # YOLO 모델 로드
        print("Loading YOLO model...")
        yolo_model = await loop.run_in_executor(None, YOLO, LOCAL_YOLO_MODEL_PATH)
        print("YOLO 모델 로드 완료.")

        # 모델 로드 후 앱을 계속 실행
        yield

    except Exception as e:
        print(f"Error during model download or load: {e}")

    finally:
        print("서버 종료 중...")


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {"message": "FastAPI 서버가 실행 중입니다!"}


class CoordinatesResponse(BaseModel):
    coordinates: List[List[int]]


@app.post("/extract_bboxes/")
async def extract_bboxes_from_image(
    image_path: str = Query(..., title="S3 Image Path")
):
    """
    이미지의 경로를 받아 S3에서 해당 이미지를 다운로드하고, YOLO 모델을 사용하여 바운딩 박스를 추출합니다.
    """
    try:
        # S3에서 이미지 다운로드
        image = download_image_from_s3(S3_BUCKET_NAME, image_path)
        if image is None:
            raise HTTPException(status_code=400, detail="S3에서 이미지 로드 실패")

        # CRAFT 또는 YOLO+CRAFT 결과 얻기
        craft_result = process_image_with_craft(image)
        coordinates = []
        if craft_result is not None and len(craft_result["boxes"]) > 0:
            coordinates = [
                np.array(box).astype(np.int32)
                for box in craft_result["boxes"]
                if box is not None and len(box) > 0
            ]
            # numpy 데이터를 Python 기본 타입으로 변환
            coordinates = [convert_numpy_to_python(coord) for coord in coordinates]
            save_coordinates(coordinates, "coordinates.txt")
        else:
            all_text_boxes, failed_boxes = process_image_with_yolo_and_craft(
                yolo_model, image
            )
            # numpy 데이터를 Python 기본 타입으로 변환
            all_text_boxes = [
                convert_numpy_to_python(coord) for coord in all_text_boxes
            ]
            # 평탄화
            all_text_boxes = flatten(all_text_boxes)  # 이 부분에서 평탄화
            save_coordinates(all_text_boxes, "coordinates.txt")
            if failed_boxes:
                save_failed_boxes(failed_boxes, "failed_boxes.txt")

        return {
            "message": "텍스트 감지 완료",
            "coordinates": coordinates,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
