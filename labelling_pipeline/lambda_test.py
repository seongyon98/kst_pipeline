import boto3
import time
from datetime import datetime, timedelta
import psycopg2
import os
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

# S3 클라이언트 초기화
s3_client = boto3.client("s3")
ec2_client = boto3.client("ec2")
ssm_client = boto3.client("ssm")  # SSM 클라이언트 초기화

# 데이터베이스 연결 설정
db_connection = psycopg2.connect(
    dbname=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
    host=os.getenv("POSTGRES_HOST"),
    port=os.getenv("POSTGRES_PORT"),
)

cursor = db_connection.cursor()

# 환경 변수에서 설정
bucket_name = os.getenv("IMAGE_BUCKET_NAME")
wait_time = 60  # 대기 시간 (초)
min_files = 10  # 최소 파일 수
max_files = 50  # 한 번에 전달할 최대 파일 수
ec2_instance_id = os.getenv("EC2_INSTANCE_ID")  # EC2 인스턴스 ID 환경 변수에서 가져오기
container_name = "labelling-container"
container_url = f"http://{os.getenv("EC2_IP")}:8000/process_images"  # FastAPI 서버 URL


def get_files_in_s3(bucket_name):
    """S3 버킷에서 파일 목록을 가져오는 함수"""
    response = s3_client.list_objects_v2(Bucket=bucket_name)
    return response.get("Contents", [])


def check_processed_in_db(file_name):
    """데이터베이스에서 파일이 처리되었는지 확인"""
    cursor.execute(
        "SELECT status FROM upload_filerecord WHERE file_name = %s", (file_name,)
    )
    result = cursor.fetchone()
    return result and result[0] == "complete"


def start_ec2_instance_with_files(instance_id, file_names):
    """EC2 인스턴스를 시작하고 처리할 파일들을 전달하는 함수"""
    # EC2에 여러 파일을 전달하는 방식 예시 (파일 목록을 EC2로 전달)
    ec2_client.start_instances(InstanceIds=[instance_id])
    print(f"EC2 instance {instance_id} started for files: {file_names}")

    # EC2 인스턴스가 준비된 후 명령 실행
    # FastAPI 서버에 파일 처리 요청 보내기
    response = send_files_to_container(container_url, file_names)
    print(f"Files sent to container: {response.status_code}")

    return {"status": "Files are being processed in the container"}


def send_files_to_container(container_url, file_names):
    """FastAPI 서버로 파일 처리 요청을 보내는 함수"""
    response = requests.post(
        f"{container_url}/process-files",
        json={"files": file_names},
    )
    return response


def lambda_handler(event, context):
    """Lambda 함수"""
    start_time = datetime.now()

    while True:
        files = get_files_in_s3(bucket_name)
        unprocessed_files = [
            file for file in files if not check_processed_in_db(file["Key"])
        ]

        file_count = len(unprocessed_files)

        # 최소 파일 수가 되거나 시간 초과 시 EC2 인스턴스 시작
        if file_count >= min_files or (datetime.now() - start_time) > timedelta(
            seconds=wait_time
        ):
            # 파일을 최대 수까지 묶어서 EC2로 전달
            files_to_process = unprocessed_files[: min(file_count, max_files)]
            file_names = [file["Key"] for file in files_to_process]
            start_ec2_instance_with_files(ec2_instance_id, file_names)
            break  # 한 번에 처리할 파일을 EC2로 전달 후 종료

        time.sleep(10)  # 10초마다 체크

    return {"status": "EC2 instance started and files are being processed"}
