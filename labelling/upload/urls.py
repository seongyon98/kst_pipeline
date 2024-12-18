from django.urls import path
from .views import UploadFileAPIView, GetResultAPIView, upload_page

urlpatterns = [
    path("upload/", upload_page, name="upload_page"),  # HTML 페이지 렌더링
    path(
        "api/upload/", UploadFileAPIView.as_view(), name="upload_file"
    ),  # 파일 업로드 API
    path(
        "api/results/<str:file_name>/", GetResultAPIView.as_view(), name="get_result"
    ),  # 결과 조회 API
]
