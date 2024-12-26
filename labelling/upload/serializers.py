from rest_framework import serializers


class FileUploadSerializer(serializers.Serializer):
    file = serializers.FileField(required=False, help_text="업로드할 파일")
    text = serializers.CharField(required=False, help_text="입력할 텍스트")

    def validate(self, data):
        # 파일이나 텍스트 둘 중 하나는 반드시 있어야 함
        if not data.get("file") and not data.get("text"):
            raise serializers.ValidationError("파일 또는 텍스트를 제공해야 합니다.")
        return data
