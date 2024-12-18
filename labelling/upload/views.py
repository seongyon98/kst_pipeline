from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.exceptions import NotFound, ValidationError
from rest_framework.parsers import MultiPartParser
from rest_framework import status
import boto3
from dotenv import load_dotenv
import os
from pymongo import MongoClient


# .env 파일에서 환경 변수 로드
load_dotenv()

# S3 연결 설정
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET = os.getenv("S3_BUCKET_NAME")  # S3_BUCKET 변수로 수정
region_name = os.getenv("AWS_REGION")

# 클라이언트 및 세션 생성
s3_client = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region_name,
)

# MongoDB 클라이언트 생성
USERNAME = os.getenv("MONGO_USER")
PASSWORD = os.getenv("MONGO_PASSWORD")
HOST = os.getenv("MONGO_HOST")
PORT = os.getenv("MONGO_PORT")
DATABASE = os.getenv("MONGO_DB")
MONGO_URI = f"mongodb://<{USERNAME}>:<{PASSWORD}>@<{HOST}>:<{PORT}>/<{DATABASE}>"
client = MongoClient(MONGO_URI)
db = client[os.getenv("DATABASE_NAME")]
collection = db[os.getenv("COLLECTION_NAME")]


# HTML 페이지 렌더링 뷰
def upload_page(request):
    return render(request, "upload.html")


# API VIEW: 파일 업로드
class UploadFileAPIView(APIView):
    """
    파일 업로드 API

    Endpoint: POST /api/upload/
    Features:
    - 파일 업로드 후 S3에 저장
    - 성공 시 파일명 반환
    """

    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser]  # 멀티파트 파일 업로드 지원

    def post(self, request):
        file = request.FILES.get("file")
        if not file:
            raise ValidationError("No file provided")

        # S3에 파일 업로드
        try:
            s3_client.upload_fileobj(file, S3_BUCKET, file.name)
            return Response(
                {"message": "File uploaded successfully", "file_name": file.name},
                status=status.HTTP_200_OK,
            )
        except Exception as e:
            return Response(
                {"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


# API VIEW: 결과 조회
class GetResultAPIView(APIView):
    """
    파일에 대한 라벨링 결과를 MongoDB에서 조회하는 API

    Endpoint: GET /api/results/{file_name}/
    Features:
    - MongoDB에서 파일명에 해당하는 결과 검색
    """

    permission_classes = [IsAuthenticated]

    def get(self, request, file_name):
        try:
            # MongoDB에서 결과 데이터 찾기
            result = collection.find_one({"file_name": file_name}, {"_id": 0})

            if result:
                return Response(result, status=status.HTTP_200_OK)
            raise NotFound("결과를 찾을 수 없습니다.")

        except Exception as e:
            return Response(
                {"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
