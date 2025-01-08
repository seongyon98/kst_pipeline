from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
import uvicorn
from llm_utils import process_multiple_problems

# FastAPI 애플리케이션 생성
app = FastAPI()


# API에서 사용할 입력 데이터 모델 정의
class ProblemData(BaseModel):
    file_name: str
    processed_data: dict


# LLM API 엔드포인트 정의
@app.post("/process_problem")
async def process_problem(problem: ProblemData):
    """
    문제 텍스트를 처리하여 라벨링하고 PostgreSQL에 저장
    """
    try:
        # LLM을 통해 문제 처리 및 라벨링
        category, leaf_category, category_time, leaf_time = process_multiple_problems(
            problem.file_name, problem.processed_data
        )
        return {
            "file_name": problem.file_name,
            "category": category,
            "leaf_category": leaf_category,
            "category_time": category_time,
            "leaf_time": leaf_time,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"문제 처리 실패: {e}")
