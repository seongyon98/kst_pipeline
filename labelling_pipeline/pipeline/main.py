from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import httpx  # httpx를 사용하여 비동기 HTTP 요청 처리
from http_client import call_yolo_api, call_ocr_api, call_llm_api
from llm_api.llm_main import process_multiple_problems

app = FastAPI()


# 요청받을 데이터 형식 정의
class ImageRequest(BaseModel):
    file_name: str
    image_path: str


# 이미지 경로를 통해 문제 텍스트 추출 후 라벨링하는 API 엔드포인트
@app.post("/process-image/")
async def process_image(request: ImageRequest):
    try:
        # 1. YOLO API 호출: 이미지에서 객체 인식
        yolo_result = await call_yolo_api(request.image_path)
        if "error" in yolo_result:
            raise HTTPException(
                status_code=400, detail=f"YOLO API 호출 실패: {yolo_result['error']}"
            )

        # 2. OCR API 호출: 이미지에서 텍스트 추출
        ocr_result = await call_ocr_api(request.image_path)
        if "error" in ocr_result:
            raise HTTPException(
                status_code=400, detail=f"OCR API 호출 실패: {ocr_result['error']}"
            )

        # 추출된 텍스트에서 문제 텍스트 얻기
        question_text = ocr_result.get("text", "")
        if not question_text:
            raise HTTPException(status_code=400, detail="OCR에서 텍스트 추출 실패")

        # 3. LLM API 호출: 문제 텍스트로 라벨링
        llm_result = await call_llm_api(question_text)
        if "error" in llm_result:
            raise HTTPException(
                status_code=400, detail=f"LLM API 호출 실패: {llm_result['error']}"
            )

        # 라벨링된 카테고리와 최하위 카테고리 추출
        category_label = llm_result.get("category_label")
        leaf_label = llm_result.get("leaf_label")

        if not category_label or not leaf_label:
            raise HTTPException(
                status_code=400, detail="LLM에서 카테고리 및 라벨 추출 실패"
            )

        # 4. 결과를 PostgreSQL에 저장
        process_multiple_problems(
            request.file_name,
            {
                "question_text": question_text,
                "category_label": category_label,
                "leaf_label": leaf_label,
            },
        )

        return {
            "file_name": request.file_name,
            "category_label": category_label,
            "leaf_label": leaf_label,
            "status": "success",
        }

    except httpx.RequestError as e:  # httpx의 예외로 수정
        # 요청 오류 처리: 네트워크 문제, API 연결 실패 등
        raise HTTPException(status_code=502, detail=f"요청 오류: {str(e)}")

    except HTTPException as e:
        # 이미 처리된 HTTP 오류 예외
        raise e

    except Exception as e:
        # 다른 일반적인 오류 처리
        raise HTTPException(status_code=500, detail=f"내부 서버 오류: {str(e)}")


# 여러 문제를 처리하는 엔드포인트 (예시)
@app.post("/process-multiple/")
async def process_multiple(request: Dict[str, Any]):
    try:
        file_name = request.get("file_name")
        processed_data = request.get("processed_data")

        if not file_name or not processed_data:
            raise HTTPException(status_code=400, detail="필수 데이터 누락")

        # 여러 문제 처리
        process_multiple_problems(file_name, processed_data)

        return {"status": "success", "message": "문제 처리 완료"}

    except Exception as e:
        # 예외 처리: 세부적인 오류 메시지를 반환
        raise HTTPException(status_code=500, detail=f"내부 서버 오류: {str(e)}")
