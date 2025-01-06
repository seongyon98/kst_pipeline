import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List
from ocr_model import (
    load_ocr_model_from_s3,  # S3에서 바로 로드
    perform_ocr_on_cropped_images,
)
import os

app = FastAPI()

# OCR 모델 관련 설정
MODEL_BUCKET = os.getenv("MODEL_BUCKET")
OCR_S3_KEY = "ocr/final_model/final_model_1/"
CROPPED_IMAGES_PATH = "/tmp/cropped_images"

# 1. S3에서 OCR 모델 로드
ocr_model, ocr_tokenizer, ocr_processor = load_ocr_model_from_s3(MODEL_BUCKET, OCR_S3_KEY)
if not ocr_model or not ocr_tokenizer or not ocr_processor:
    raise RuntimeError("[ERROR] OCR 모델 로드에 실패했습니다.")

# 2. FastAPI 엔드포인트 정의
@app.post("/extract_text/")
async def extract_text_from_images(files: List[UploadFile] = File(...)):
    try:
        # 크롭된 이미지 저장 디렉토리 생성
        if not os.path.exists(CROPPED_IMAGES_PATH):
            os.makedirs(CROPPED_IMAGES_PATH, exist_ok=True)

        image_paths = []
        for file in files:
            # 파일 읽기 및 로컬 저장
            file_data = await file.read()
            image_path = os.path.join(CROPPED_IMAGES_PATH, file.filename)
            with open(image_path, "wb") as f:
                f.write(file_data)
            image_paths.append(image_path)

        # OCR 수행
        all_texts = perform_ocr_on_cropped_images(
            image_paths=image_paths,
            model=ocr_model,
            tokenizer=ocr_tokenizer,
            image_processor=ocr_processor,
            image_size=384,
        )

        # 최종 텍스트 합성
        final_text = " ".join(all_texts).strip()
        print(f"[INFO] Combined text: {final_text}")

        return JSONResponse(content={"text": final_text})

    finally:
        # 디렉토리 정리: 처리 후 크롭된 이미지 삭제
        if os.path.exists(CROPPED_IMAGES_PATH):
            shutil.rmtree(CROPPED_IMAGES_PATH)  # 디렉토리와 파일 모두 삭제
            print(f"[INFO] Temporary directory {CROPPED_IMAGES_PATH} has been cleaned up.")