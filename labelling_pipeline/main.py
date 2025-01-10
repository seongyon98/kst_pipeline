from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import os
import boto3
import cv2
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from typing import List

# 제공된 파일들에서 함수 임포트
from ocr_utils import load_ocr_model_from_s3, extract_text_from_folders
from yolo_utils import (
    download_model_from_s3,
    initialize_yolo_model,
    initialize_craft_model,
    create_directories,
    download_selected_images_from_s3,
    process_image_with_yolo_and_craft,
    save_coordinates,
    save_failed_boxes,
    save_cropped_image,
)
from llm_utils import process_multiple_problems

# FastAPI 초기화
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
OCR_S3_KEY = f"ocr/final_model/{os.getenv('OCR_S3_KEY')}"  # 실제 모델 저장 경로로 수정(models/ocr_training/)
QUESTION_BUCKET = os.getenv("IMAGE_BUCKET_NAME")
YOLO_S3_KEY = f"{os.getenv('YOLO_MODEL_PATH')}"  # 실제 모델 저장 경로로 수정(models/yolo_training/)

# 로컬 저장 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트 경로
LOCAL_MODEL_DIR = os.path.join(BASE_DIR, "ocr_model")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CROPPED_IMAGES_DIR = os.path.join(BASE_DIR, "cropped_images")
MODEL_FOLDER = os.path.join(BASE_DIR, "models")
YOLO_LOCAL_PATH = os.path.join(MODEL_FOLDER, f"{os.getenv('YOLO_MODEL_PATH')}")
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan 컨텍스트를 활용하여 모델을 초기화하고 종료 시 리소스를 정리.
    """
    global yolo_model, craft_model, ocr_model, ocr_tokenizer, ocr_processor

    logger.info("[LIFESPAN] Initializing models...")

    try:
        # YOLO 모델 다운로드 및 초기화
        download_model_from_s3(s3_client, MODEL_BUCKET, YOLO_S3_KEY, YOLO_LOCAL_PATH)
        yolo_model = initialize_yolo_model(YOLO_LOCAL_PATH)
        if not yolo_model:
            raise RuntimeError("YOLO 모델 초기화 실패")

        # CRAFT 모델 초기화
        OUTPUT_DIR, _, _, _ = create_directories(BASE_DIR)
        craft_model = initialize_craft_model(OUTPUT_DIR)
        if not craft_model:
            raise RuntimeError("CRAFT 모델 초기화 실패")

        # OCR 모델 다운로드 및 초기화
        ocr_model, ocr_tokenizer, ocr_processor = load_ocr_model_from_s3(
            MODEL_BUCKET, OCR_S3_KEY, OCR_LOCAL_PATH, CROPPED_IMAGES_DIR
        )
        if not all([ocr_model, ocr_tokenizer, ocr_processor]):
            raise RuntimeError("OCR 모델 초기화 실패")

        logger.info("[LIFESPAN] Models initialized successfully.")
        yield  # 애플리케이션 실행

    except Exception as e:
        logger.error(f"[LIFESPAN] Model initialization failed: {str(e)}")
        raise RuntimeError("모델 초기화 중 오류가 발생했습니다.")
    finally:
        logger.info("[LIFESPAN] Cleaning up resources...")


# FastAPI 앱에 lifespan 설정
app = FastAPI(lifespan=lifespan)


class ProcessRequest(BaseModel):
    image_s3_keys: list  # 여러 개의 이미지 S3 키를 받도록 수정


class ImageResult(BaseModel):
    image_s3_key: str
    question_text: str
    major_category: str
    label_category: str
    label_time: float


class ProcessResponse(BaseModel):
    image_results: list[ImageResult]  # 여러 개의 이미지 결과를 포함하는 리스트


@app.post("/process_images", response_model=ProcessResponse)
async def process_images(request: ProcessRequest):
    try:
        logger.info("[INFO] Starting batch image processing...")

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

        # 이미지 다운로드 및 처리
        all_image_results = []  # 여러 이미지 결과를 담을 리스트

        for image_s3_key in request.image_s3_keys:
            logger.info(f"[INFO] Downloading image {image_s3_key}...")

            # 각 이미지 키에 대해 다운로드 함수 호출
            local_image_paths = download_selected_images_from_s3(
                s3_client, QUESTION_BUCKET, [image_s3_key], LOCAL_IMAGE_FOLDER
            )

            # 다운로드된 이미지 경로가 없으면 에러 발생
            if not local_image_paths or not os.path.exists(local_image_paths[0]):
                raise FileNotFoundError(
                    f"이미지 파일을 찾을 수 없음: {local_image_paths[0]}"
                )

            # 여러 이미지에 대해 처리
            for local_image_path in local_image_paths:
                logger.info(f"[INFO] Processing image {local_image_path}...")

                # YOLO와 CRAFT로 이미지 처리
                logger.info(
                    f"[INFO] Processing image {image_s3_key} with YOLO and CRAFT..."
                )
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
                logger.info(
                    f"[INFO] Saving results and cropping images for {image_s3_key}..."
                )
                save_coordinates(
                    text_boxes,
                    os.path.join(
                        COORDINATES_DIR,
                        f"{os.path.basename(local_image_path)}_coordinates.txt",
                    ),
                )
                save_failed_boxes(
                    failed_boxes,
                    os.path.join(
                        FAILED_BOXES_DIR,
                        f"{os.path.basename(local_image_path)}_failed_boxes.txt",
                    ),
                )

                cropped_image_base_dir = os.path.join(
                    CROPPED_DIR, os.path.splitext(os.path.basename(local_image_path))[0]
                )
                os.makedirs(cropped_image_base_dir, exist_ok=True)

                for i, box in enumerate(text_boxes):
                    x, y, w, h = cv2.boundingRect(box)
                    cropped_image = image[y : y + h, x : x + w]
                    cropped_image_path = os.path.join(
                        cropped_image_base_dir, f"crop_{i}.jpg"
                    )
                    save_cropped_image(cropped_image, cropped_image_path)

                # OCR 수행
                logger.info(
                    f"[INFO] Performing OCR on cropped images for {image_s3_key}..."
                )
                ocr_results = extract_text_from_folders(
                    ocr_model, ocr_tokenizer, ocr_processor, cropped_image_base_dir
                )

                if isinstance(ocr_results, dict) and "ocr_results" in ocr_results:
                    ocr_result_items = ocr_results["ocr_results"]
                    question_texts = [
                        text.strip()
                        for text in ocr_result_items
                        if isinstance(text, str) and text.strip()
                    ]
                else:
                    raise ValueError(
                        f"[ERROR] OCR 결과가 올바르지 않습니다: {ocr_results}"
                    )

                # LLM에 전달할 문제 텍스트 생성
                problem_text = " ".join(question_texts)
                if not problem_text.strip():
                    raise ValueError(
                        "[ERROR] OCR 결과에서 문제 텍스트를 생성하지 못했습니다."
                    )

                # 수학 개념 추출 및 대분류 결정
                logger.info(
                    "[INFO] Extracting math concepts and determining major category..."
                )
                category, leaf_category, category_time, leaf_time = (
                    process_multiple_problems(
                        request.image_s3_keys, QUESTION_BUCKET, problem_text
                    )
                )

                # 각 이미지에 대한 결과를 리스트에 추가
                image_result = ImageResult(
                    image_s3_key=image_s3_key,
                    question_text=problem_text,
                    major_category=category,
                    label_category=leaf_category,
                    label_time=category_time + leaf_time,
                )
                all_image_results.append(image_result)

        logger.info("[INFO] Batch image processing completed successfully.")
        return ProcessResponse(image_results=all_image_results)  # 여러 이미지 결과 반환

    except Exception as e:
        logger.error(f"[ERROR] Batch image processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


class TextProcessRequest(BaseModel):
    text_s3_keys: List[str]  # S3 키 리스트로 입력받도록 설계


class TextResult(BaseModel):
    question_text: str
    major_category: str
    label_category: str
    label_time: float


class ProcessTextResponse(BaseModel):
    text_results: list[TextResult]


@app.post("/process_texts", response_model=ProcessTextResponse)
async def process_texts(request: TextProcessRequest):
    """
    S3에서 텍스트 데이터를 읽어와 LLM을 통해 결과 반환
    """
    try:
        logger.info("[INFO] Starting text processing from S3...")

        all_text_results = []  # 텍스트 처리 결과를 담을 리스트

        for text_s3_key in request.text_s3_keys:
            logger.info(f"[INFO] Downloading text from S3: {text_s3_key}")

            # S3에서 텍스트 파일 다운로드
            text_file_path = os.path.join(
                LOCAL_IMAGE_FOLDER, os.path.basename(text_s3_key)
            )
            s3_client.download_file(QUESTION_BUCKET, text_s3_key, text_file_path)

            # 텍스트 파일 읽기
            if not os.path.exists(text_file_path):
                raise FileNotFoundError(
                    f"다운로드된 파일이 존재하지 않습니다: {text_file_path}"
                )

            with open(text_file_path, "r", encoding="utf-8") as file:
                text = file.read().strip()

            if not text:
                raise ValueError(f"파일이 비어 있습니다: {text_s3_key}")

            logger.info(f"[INFO] Processing text: {text}")

            # LLM에 전달하여 처리
            category, leaf_category, category_time, leaf_time = (
                process_multiple_problems(request.text_s3_keys, QUESTION_BUCKET, text)
            )

            # 처리 결과 저장
            text_result = TextResult(
                question_text=text,
                major_category=category,
                label_category=leaf_category,
                label_time=category_time + leaf_time,
            )
            all_text_results.append(text_result)

        logger.info("[INFO] Text processing from S3 completed successfully.")
        return ProcessTextResponse(text_results=all_text_results)

    except Exception as e:
        logger.error(f"[ERROR] Text processing from S3 failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
