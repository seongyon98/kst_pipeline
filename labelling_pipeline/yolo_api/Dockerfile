# Python 3.9 기반 이미지
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 시스템 라이브러리 설치 (예: OpenCV, PyTorch 등)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies 설치
COPY requirements.txt . 
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# 소스 코드 및 모델 복사
COPY src/ ./src
COPY models/ ./models

# 다운로드 및 크롭된 이미지 디렉토리 생성
RUN mkdir -p /tmp/downloads /tmp/cropped_images

# 환경 변수 설정
ENV PYTHONPATH=/app/src

# 서버 실행 명령어
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
