from django.shortcuts import render, redirect
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from rest_framework.exceptions import NotFound
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
import boto3
import os
import time
from dotenv import load_dotenv
from .serializers import FileUploadSerializer
from labelling.settings import (
    AWS_ACCESS_KEY,
    AWS_SECRET_ACCESS_KEY,
    S3_BUCKET,
    S3_REGION,
)
from .models import FileRecord, LabellingResult  # PostgreSQL 모델 임포트

load_dotenv()

# S3 연결 설정
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=S3_REGION,
)


# 기본 페이지 뷰 추가
def home(request):
    return render(request, "home.html")


# HTML 페이지 렌더링 뷰
def upload_page(request):
    return render(request, "upload.html")


def result_page(request, file_name):
    """
    업로드 후 해당 파일에 대한 결과를 조회하는 페이지를 보여주는 뷰입니다.
    """
    return render(request, "result.html", {"file_name": file_name})


def list_page(request):
    """
    저장된 파일 목록을 조회하는 페이지를 렌더링하는 뷰입니다.
    """
    return render(request, "list.html")  # list.html로 파일 목록 페이지 렌더링


class UploadFileAPIView(APIView):
    """
    파일 업로드 또는 텍스트 입력을 받아 처리하는 API
    """

    permission_classes = [AllowAny]
    parser_classes = [
        MultiPartParser,
        FormParser,
    ]  # 파일 업로드를 위해 MultiPartParser 추가

    @swagger_auto_schema(
        operation_description="파일 업로드 또는 텍스트 입력 처리 API",
        request_body=FileUploadSerializer,  # Serializer 적용
        responses={
            200: openapi.Response(description="성공적으로 업로드되었습니다."),
            400: openapi.Response(description="파일이나 텍스트가 제공되지 않았습니다."),
        },
    )
    def post(self, request):
        # 여러 파일 업로드 처리
        files = request.data.getlist("file")
        text = request.data.getlist("text")  # 여러 텍스트 처리

        if files:
            try:
                file_names = []  # 업로드된 파일 이름을 저장할 리스트
                for file in files:
                    file_extension = file.name.split(".")[-1]
                    if file_extension.lower() in [
                        "jpg",
                        "jpeg",
                        "png",
                    ]:  # 이미지 파일일 경우
                        s3_key = f"image/{file.name}"
                    else:  # 텍스트 파일일 경우
                        s3_key = f"text/{file.name}"

                    s3_client.upload_fileobj(file, S3_BUCKET, s3_key)

                    # PostgreSQL에 업로드된 파일 정보 저장 (Django ORM 사용)
                    FileRecord.objects.create(
                        file_name=file.name, s3_key=s3_key, status="processing"
                    )
                    file_names.append(file.name)  # 파일 이름 추가

                return Response(
                    {"message": "파일 업로드 성공", "file_names": file_names},
                    status=status.HTTP_200_OK,
                )

            except Exception as e:
                return Response(
                    {"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

        if text:
            try:
                text_file_names = []  # 업로드된 텍스트 파일 이름을 저장할 리스트
                for t in text:
                    timestamp = int(time.time())
                    file_name = f"uploaded_text_{timestamp}.txt"
                    s3_key = f"text/{file_name}"

                    # 텍스트 파일 S3에 저장
                    s3_client.put_object(Body=t, Bucket=S3_BUCKET, Key=s3_key)

                    # PostgreSQL에 텍스트 정보 저장 (Django ORM 사용)
                    FileRecord.objects.create(
                        file_name=file_name, s3_key=s3_key, status="processing"
                    )
                    text_file_names.append(file_name)  # 텍스트 파일 이름 추가

                return Response(
                    {"message": "텍스트 업로드 성공", "file_names": text_file_names},
                    status=status.HTTP_200_OK,
                )

            except Exception as e:
                return Response(
                    {"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

        return Response(
            {"error": "파일이나 텍스트가 제공되지 않았습니다."},
            status=status.HTTP_400_BAD_REQUEST,
        )


class GetAllResultsAPIView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        try:
            file_names = request.data.get("file_names", [])
            results = FileRecord.objects.filter(file_name__in=file_names)
            results_list = []
            for file_record in results:
                labeling_result = LabellingResult.objects.filter(
                    file_record=file_record
                ).first()
                result_data = {
                    "file_name": file_record.file_name,
                    "status": file_record.status,
                    "labeling_result": (
                        labeling_result.result
                        if labeling_result
                        else "라벨링 결과 없음"
                    ),
                }
                results_list.append(result_data)
            return Response(results_list, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {"error": f"데이터베이스 조회 실패: {e}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class GetResultAPIView(APIView):
    permission_classes = [AllowAny]

    def get(self, request, file_name):
        try:
            # 파일 정보 조회
            file_record = FileRecord.objects.filter(file_name=file_name).first()
            if not file_record:
                raise NotFound("파일을 찾을 수 없습니다.")

            # 해당 파일에 대한 라벨링 결과 조회
            labeling_result = LabellingResult.objects.filter(
                file_record=file_record
            ).first()

            if labeling_result:
                return Response(
                    {
                        "file_name": file_record.file_name,
                        "status": file_record.status,
                        "labeling_result": labeling_result.label,
                    },
                    status=status.HTTP_200_OK,
                )
            else:
                raise NotFound("라벨링 결과를 찾을 수 없습니다.")
        except Exception as e:
            return Response(
                {"error": f"데이터베이스 조회 실패: {e}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class ListFilesAPIView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        try:
            # PostgreSQL에서 모든 파일 목록 조회 (Django ORM 사용)
            files = FileRecord.objects.all().values("file_name")
            files_list = [file["file_name"] for file in files]
            return Response({"files": files_list}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {"error": f"데이터베이스 조회 실패: {e}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
