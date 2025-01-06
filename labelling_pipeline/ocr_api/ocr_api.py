# [로컬에서 ocr 모델 로드]
import os
import shutil
import json
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List
from ocr_model import (
    load_ocr_model,  # 로컬 디렉터리에서 모델 로드
    perform_ocr_on_cropped_images,
)
from yolo_api.src.config import CROPPED_IMAGES_DIR  # 로컬 디렉터리에서 크롭된 이미지 로드

app = FastAPI()

# OCR 모델 관련 설정
LOCAL_MODEL_PATH = "C:/Users/user/Desktop/final_project/github/labelling_pipeline/ocr_api/models/final_model_1"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# 결과 저장 디렉토리 생성
os.makedirs(RESULTS_DIR, exist_ok=True)

# 1. 로컬 디렉터리에서 OCR 모델 로드
ocr_model, ocr_tokenizer, ocr_processor = load_ocr_model(LOCAL_MODEL_PATH)
if not ocr_model or not ocr_tokenizer or not ocr_processor:
    raise RuntimeError(f"[ERROR] OCR 모델 로드에 실패했습니다. 경로를 확인하세요: {LOCAL_MODEL_PATH}")

# 2. FastAPI 엔드포인트 정의
@app.post("/extract_text/")
async def extract_text_from_images(files: List[UploadFile] = File(...)):
    try:
        # 크롭된 이미지 저장 디렉토리 생성
        if not os.path.exists(CROPPED_IMAGES_DIR):
            os.makedirs(CROPPED_IMAGES_DIR, exist_ok=True)

        image_paths = []
        for file in files:
            # 파일 읽기 및 로컬 저장
            file_data = await file.read()
            image_path = os.path.join(CROPPED_IMAGES_DIR, file.filename)
            with open(image_path, "wb") as f:
                f.write(file_data)
            image_paths.append(image_path)

        if not image_paths:
            return JSONResponse(content={"error": "No valid images provided."}, status_code=400)

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

        # 결과 JSON 생성 및 저장
        result = {
            "cropped_images": [os.path.basename(path) for path in image_paths],
            "ocr_result": final_text
        }

        output_json_path = os.path.join(RESULTS_DIR, "ocr_results.json")
        with open(output_json_path, "w", encoding="utf-8") as json_file:
            json.dump(result, json_file, ensure_ascii=False, indent=4)

        print(f"[INFO] OCR results saved to: {output_json_path}")

        return JSONResponse(content={"text": final_text, "result_file": output_json_path})

    except Exception as e:
        print(f"[ERROR] Failed to process images: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

    finally:
        # 디렉토리 정리: 처리 후 크롭된 이미지 삭제
        if os.path.exists(CROPPED_IMAGES_DIR):
            shutil.rmtree(CROPPED_IMAGES_DIR)  # 디렉토리와 파일 모두 삭제
            print(f"[INFO] Temporary directory {CROPPED_IMAGES_DIR} has been cleaned up.")