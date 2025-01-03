FROM apache/airflow:2.6.3-python3.9

USER root
RUN apt-get update && apt-get install -y nano bash
USER airflow

# 필요하면 추가 패키지
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Airflow 환경변수
ENV AIRFLOW_HOME=/opt/airflow

# dags, scripts, preprocessing, yolo, ocr 폴더는 docker-compose.yml에서 마운트
