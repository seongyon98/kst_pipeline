from typing import List
import numpy as np
from fastapi import FastAPI
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
    process_image_with_yolo_and_craft,
    process_image_with_craft,
    save_coordinates,
    save_failed_boxes,
)
from itertools import chain

load_dotenv(override=True)

S3_BUCKET_NAME = "big9-project-02-model-bucket"
YOLO_MODEL_PATH = "yolov8_text_nontext.pt"  # S3 경로
LOCAL_YOLO_MODEL_PATH = "./models/yolov8_text_nontext.pt"  # 로컬 경로에 YOLO 모델 저장

S3_IMAGE_BUCKET = "big9-project-02-question-bucket"
S3_IMAGE_PATH = "image/P3_1_01_21114_49495.png"  # 이미지 S3 경로

# S3 클라이언트 생성
s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
)


# S3에서 이미지를 다운로드하는 함수
def download_image_from_s3(bucket_name, file_key):
    file_obj = BytesIO()
    s3_client.download_fileobj(bucket_name, file_key, file_obj)
    file_obj.seek(0)
    img_array = np.asarray(bytearray(file_obj.read()), dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return image


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
async def extract_bboxes_from_image():
    try:
        # S3에서 이미지 다운로드
        image = download_image_from_s3(
            "big9-project-02-question-bucket", "image/P3_1_01_21114_49495.png"
        )
        if image is None:
            return {"error": "S3에서 이미지 로드 실패"}

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
            all_text_boxes, failed_boxes = process_image_with_yolo_and_craft(image)
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
        return {"error": str(e)}
