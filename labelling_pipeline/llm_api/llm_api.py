from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llm_main import process_multiple_problems
from typing import Optional
import os
import uvicorn

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
        process_multiple_problems(problem.file_name, problem.processed_data)
        return {"message": "문제 처리가 완료되었습니다."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"문제 처리 실패: {e}")


# 서버 시작을 위한 명령
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
