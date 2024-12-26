from django.urls import path
from .views import (
    home,
    UploadFileAPIView,
    GetResultAPIView,
    GetAllResultsAPIView,
    upload_page,
    result_page,
    ListFilesAPIView,
    list_page,
)
from drf_yasg.views import get_schema_view
from drf_yasg import openapi
from rest_framework.permissions import AllowAny

# Swagger UI를 위한 설정
schema_view = get_schema_view(
    openapi.Info(
        title="자동 라벨링 시스템 API",
        default_version="v1",
        description="자동 라벨링 시스템의 API 문서입니다.",
        terms_of_service="https://www.google.com/policies/terms/",
        contact=openapi.Contact(email="contact@myapi.com"),
        license=openapi.License(name="BSD License"),
    ),
    public=True,
    permission_classes=[AllowAny],
)

# app_name을 설정하여 네임스페이스를 정의
app_name = "upload"

urlpatterns = [
    path("", home, name="home"),
    path("upload/", upload_page, name="upload_page"),  # HTML 페이지 렌더링
    path(
        "api/upload/", UploadFileAPIView.as_view(), name="upload_file"
    ),  # 파일 업로드 API
    path(
        "api/results/<str:file_name>/", GetResultAPIView.as_view(), name="get_result"
    ),  # 결과 조회 API
    path("api/all-results/", GetAllResultsAPIView.as_view(), name="get_all_results"),
    path(
        "result/<str:file_name>/", result_page, name="result_page_view"
    ),  # 업로드 후 해당 파일의 결과 페이지
    path(
        "api/files/", ListFilesAPIView.as_view(), name="list_files"
    ),  # 저장된 파일 목록 조회 API
    path("file-list/", list_page, name="file_list_page"),  # 저장된 파일 목록 페이지
    path(
        "swagger/", schema_view.with_ui("swagger", cache_timeout=0), name="swagger-ui"
    ),
]
