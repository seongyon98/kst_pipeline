import boto3
import psycopg2
import os
import requests
from dotenv import load_dotenv
import logging

# 환경 변수 로드
load_dotenv(override=True)

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# S3 및 DB 클라이언트 초기화
s3_client = boto3.client("s3")

# 데이터베이스 연결 설정
def get_db_connection():
    """데이터베이스 연결을 반환하는 함수"""
    return psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
    )

# FastAPI 서버 URL
container_url = f"http://{os.getenv('EC2_IP')}:8000/process_images"

def check_processed_in_db(cursor, file_name):
    """
    데이터베이스에서 파일이 처리되었는지 확인
    """
    try:
        cursor.execute(
            "SELECT status FROM upload_filerecord WHERE file_name = %s", (file_name,)
        )
        result = cursor.fetchone()
        return result and result[0] == "complete"
    except Exception as e:
        logging.error(f"Error checking database for file {file_name}: {e}")
        return False


def send_files_to_container(container_url, file_names, retries=3):
    """
    FastAPI 서버로 파일 처리 요청을 보내는 함수
    """
    for attempt in range(retries):
        try:
            logging.info(f"Sending files to FastAPI: {file_names}")
            response = requests.post(
                container_url,
                json={"image_s3_keys": file_names},
                timeout=120
            )
            logging.info(f"FastAPI Response: {response.status_code}, {response.text}")
            if response.status_code == 200:
                return response
            else:
                logging.error(
                    f"FastAPI Error: Status {response.status_code}, {response.text}"
                )
        except Exception as e:
            logging.error(f"Failed to send request to FastAPI: {e}")

        # 재시도 대기
        if attempt < retries - 1:
            logging.info(f"Retrying FastAPI request... Attempt {attempt + 1}")

    raise Exception(f"Failed to send request to FastAPI after {retries} attempts")


def lambda_handler(event, context):
    """
    Lambda 함수: S3 이벤트를 처리하고 데이터베이스에서 처리되지 않은 파일만 FastAPI로 전달
    """
    db_connection = None
    try:
        # 데이터베이스 연결
        db_connection = get_db_connection()
        cursor = db_connection.cursor()

        files_to_process = []

        for record in event['Records']:
            bucket_name = record['s3']['bucket']['name']
            file_name = record['s3']['object']['key']
            logging.info(f"New file uploaded: {file_name}")

            # S3 키를 "image/<file_name>" 형식으로 변경
            s3_key_with_path = f"{file_name}"

            # 데이터베이스에서 처리 상태 확인
            if not check_processed_in_db(cursor, file_name):
                files_to_process.append(s3_key_with_path)
            else:
                logging.info(f"File {file_name} already processed. Skipping.")

        if files_to_process:
            # FastAPI로 파일 처리 요청
            response = send_files_to_container(container_url, files_to_process)
            logging.info(f"Files processed successfully: {response.json()}")

            # 처리된 파일을 데이터베이스에 업데이트
            for file_name in files_to_process:
                cursor.execute(
                    "UPDATE upload_filerecord SET status = %s WHERE file_name = %s",
                    ("complete", file_name.split("/")[-1]),  # 파일 이름만 DB에 업데이트
                )

            # 트랜잭션 커밋
            db_connection.commit()
        else:
            logging.info("No new files to process")

    except Exception as e:
        logging.error(f"Error processing files: {e}")
        if db_connection:
            db_connection.rollback()  # 트랜잭션 롤백

    finally:
        # 데이터베이스 연결 닫기
        if db_connection:
            cursor.close()
            db_connection.close()

    return {"status": "Lambda execution complete"}

# [Environment Variables]
# ECP_IP: 172.31.17.163 (EC2 Private IP)
# POSTGRES_DB: labelling_db
# POSTGRES_HOST: 172.31.17.163 (EC2 Private IP)
# POSTGRES_PORT: 5432
# POSTGRES_USER: user