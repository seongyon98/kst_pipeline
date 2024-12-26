from django.db import models


class FileRecord(models.Model):
    file_name = models.CharField(max_length=255)
    s3_key = models.CharField(max_length=255)
    status = models.CharField(max_length=50)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.file_name


class LabellingResult(models.Model):
    file_record = models.ForeignKey(
        FileRecord, on_delete=models.CASCADE
    )  # 연결된 파일 정보
    label = models.CharField(
        max_length=255
    )  # 라벨링된 결과
    processed_at = models.DateTimeField(auto_now_add=True)  # 라벨링이 처리된 시간

    def __str__(self):
        return f"Result for {self.file_record.file_name} - {self.label}"
