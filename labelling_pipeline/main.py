import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
import cv2
import boto3
from dotenv import load_dotenv

# 제공된 파일들에서 함수 임포트
from ocr_utils import load_ocr_model_from_s3, extract_text_from_folders
from yolo_utils import (
    download_model_from_s3,
    initialize_yolo_model,
    initialize_craft_model,
    create_directories,
    download_latest_image_from_s3,
    process_image_with_yolo_and_craft,
    save_coordinates,
    save_failed_boxes,
    save_cropped_image,
)
from llm_utils import process_multiple_problems


app = FastAPI()

# 로깅 설정
logger = logging.getLogger("uvicorn.error")

# 환경 변수 로드
load_dotenv(override=True)

# AWS 설정
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
MODEL_BUCKET = os.getenv("MODEL_BUCKET_NAME")
OCR_S3_KEY = f"ocr/final_model/{os.getenv('OCR_S3_KEY')}"
QUESTION_BUCKET = os.getenv("IMAGE_BUCKET_NAME")
YOLO_S3_KEY = os.getenv("YOLO_MODEL_PATH")

# 로컬 저장 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트 경로
LOCAL_MODEL_DIR = os.path.join(BASE_DIR, "ocr_model")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CROPPED_IMAGES_DIR = os.path.join(BASE_DIR, "cropped_images")
MODEL_FOLDER = os.path.join(BASE_DIR, "models")
YOLO_LOCAL_PATH = os.path.join(MODEL_FOLDER, "yolov8_text_nontext.pt")
OCR_LOCAL_PATH = os.path.join(MODEL_FOLDER, "final_model_1")  # OCR 디렉토리
LOCAL_IMAGE_FOLDER = os.path.join(BASE_DIR, "local_images")
os.makedirs(LOCAL_IMAGE_FOLDER, exist_ok=True)

# S3 클라이언트 초기화
try:
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )
except Exception as e:
    logger.error(f"S3 클라이언트 초기화 실패: {str(e)}")
    raise RuntimeError("S3 클라이언트 초기화에 실패했습니다.")

# 전역 변수로 모델 캐싱
yolo_model = None
craft_model = None
ocr_model = None
ocr_tokenizer = None
ocr_processor = None


@app.on_event("startup")
async def load_models():
    """
    서버 시작 시 YOLO, CRAFT, OCR 모델을 한 번 로드하여 재사용.
    """
    global yolo_model, craft_model, ocr_model, ocr_tokenizer, ocr_processor

    logger.info("[STARTUP] Initializing models...")

    try:
        download_model_from_s3(s3_client, MODEL_BUCKET, YOLO_S3_KEY, YOLO_LOCAL_PATH)
        yolo_model = initialize_yolo_model(YOLO_LOCAL_PATH)
        if not yolo_model:
            raise RuntimeError("YOLO 모델 초기화 실패")

        OUTPUT_DIR, _, _, _ = create_directories(BASE_DIR)
        craft_model = initialize_craft_model(OUTPUT_DIR)
        if not craft_model:
            raise RuntimeError("CRAFT 모델 초기화 실패")

        ocr_model, ocr_tokenizer, ocr_processor = load_ocr_model_from_s3(
            MODEL_BUCKET, OCR_S3_KEY, OCR_LOCAL_PATH, CROPPED_IMAGES_DIR
        )
        if not all([ocr_model, ocr_tokenizer, ocr_processor]):
            raise RuntimeError("OCR 모델 초기화 실패")

        logger.info("[STARTUP] Models initialized successfully.")
    except Exception as e:
        logger.error(f"[STARTUP] Model initialization failed: {str(e)}")
        raise RuntimeError("모델 초기화 중 오류가 발생했습니다.")


class ProcessRequest(BaseModel):
    image_s3_key: str


class ProcessResponse(BaseModel):
    question_text: str
    major_category: str
    label_category: str
    label_time: float


@app.post("/process_image", response_model=ProcessResponse)
async def process_image(request: ProcessRequest):
    try:
        logger.info("[INFO] Starting image processing...")

        if (
            not yolo_model
            or not craft_model
            or not all([ocr_model, ocr_tokenizer, ocr_processor])
        ):
            raise RuntimeError("모델이 초기화되지 않았습니다.")

        # 디렉토리 생성
        OUTPUT_DIR, COORDINATES_DIR, FAILED_BOXES_DIR, CROPPED_DIR = create_directories(
            BASE_DIR
        )

        # 최신 이미지 다운로드
        logger.info("[INFO] Downloading the latest image...")
        local_image_path = download_latest_image_from_s3(
            s3_client, QUESTION_BUCKET, request.image_s3_key, LOCAL_IMAGE_FOLDER
        )

        if not local_image_path or not os.path.exists(local_image_path):
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없음: {local_image_path}")

        # YOLO와 CRAFT로 이미지 처리
        logger.info("[INFO] Processing image with YOLO and CRAFT...")
        image = cv2.imread(local_image_path)
        if image is None:
            raise ValueError(f"이미지를 읽을 수 없음: {local_image_path}")

        text_boxes, failed_boxes = process_image_with_yolo_and_craft(
            image,
            os.path.basename(local_image_path),
            yolo_model=yolo_model,
            craft_model=craft_model,
        )

        # 결과 저장 및 이미지 크롭
        logger.info("[INFO] Saving results and cropping images...")
        save_coordinates(
            text_boxes,
            os.path.join(
                COORDINATES_DIR, f"{os.path.basename(local_image_path)}_coordinates.txt"
            ),
        )
        save_failed_boxes(
            failed_boxes,
            os.path.join(
                FAILED_BOXES_DIR,
                f"{os.path.basename(local_image_path)}_failed_boxes.txt",
            ),
        )

        for i, box in enumerate(text_boxes):
            x, y, w, h = cv2.boundingRect(box)
            cropped_image = image[y : y + h, x : x + w]
            cropped_image_path = os.path.join(
                CROPPED_DIR, f"{os.path.basename(local_image_path)}_crop_{i}.jpg"
            )
            save_cropped_image(cropped_image, cropped_image_path)

        # OCR 수행
        logger.info("[INFO] Performing OCR on cropped images...")
        ocr_results = extract_text_from_folders(ocr_model, ocr_tokenizer, ocr_processor)
        if isinstance(ocr_results, dict) and "ocr_results" in ocr_results:
            # OCR 결과에서 "text" 값만 추출
            question_texts = [
                item["text"]
                for item in ocr_results["ocr_results"].values()
                if "text" in item
            ]
            if not question_texts:
                raise ValueError(
                    "[ERROR] OCR 결과에서 'text' 데이터를 추출하지 못했습니다."
                )
        else:
            raise ValueError("[ERROR] OCR 결과가 올바르지 않습니다.")

        # LLM에 전달할 문제 텍스트 생성
        problem_text = " ".join(question_texts)
        if not problem_text.strip():
            raise ValueError("[ERROR] OCR 결과에서 문제 텍스트를 생성하지 못했습니다.")

        # 수학 개념 추출 및 대분류 결정
        logger.info("[INFO] Extracting math concepts and determining major category...")
        category, leaf_category, category_time, leaf_time = process_multiple_problems(
            request.image_s3_key, QUESTION_BUCKET, problem_text
        )

        logger.info("[INFO] Image processing completed successfully.")
        return ProcessResponse(
            question_text=problem_text,
            major_category=category,
            label_category=leaf_category,
            label_time=category_time + leaf_time,
        )

    except Exception as e:
        logger.error(f"[ERROR] Image processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
