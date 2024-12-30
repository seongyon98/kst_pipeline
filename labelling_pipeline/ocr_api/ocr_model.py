import boto3
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from dotenv import load_dotenv
import os
from PIL import Image
import tempfile

# 환경 변수 로드
load_dotenv(override=True)

# S3 설정
s3_client = boto3.client("s3")
MODEL_BUCKET_NAME = os.getenv("MODEL_BUCKET_NAME")
OCR_MODEL_PATH = os.getenv("OCR_MODEL_PATH")
LOCAL_OCR_PATH = "/tmp/temp_ocr/"  # OCR 모델 임시 저장 경로


def load_finetuned_trocr_model_from_s3():
    try:
        # 임시 디렉토리 생성
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"[INFO] Temporary directory created at {temp_dir}")

            # S3에서 필요한 파일 리스트 정의
            trocr_files = [
                "config.json",
                "pytorch_model.bin",
                "tokenizer.json",
                "tokenizer_config.json",
                "preprocessor_config.json",
                "generation_config.json",
            ]

            # 파일 개별 다운로드
            for file_name in trocr_files:
                s3_file_path = f"{OCR_MODEL_PATH}/{file_name}"
                local_file_path = os.path.join(temp_dir, file_name)

                # S3에서 파일 다운로드
                try:
                    s3_client.download_file(
                        MODEL_BUCKET_NAME, s3_file_path, local_file_path
                    )
                    print(f"[INFO] Downloaded {file_name} to {local_file_path}")
                except Exception as e:
                    print(f"[ERROR] Failed to download {file_name}: {e}")
                    raise e

            # 모델과 프로세서 로드
            try:
                processor = TrOCRProcessor.from_pretrained(temp_dir)
                model = VisionEncoderDecoderModel.from_pretrained(temp_dir)
                print(f"[INFO] Fine-tuned TrOCR model loaded from {temp_dir}")
                return processor, model
            except Exception as e:
                print(f"[ERROR] Failed to load TrOCR model: {e}")
                raise e

    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
        raise e


def extract_text_from_bboxes(processor, model, image, bboxes):
    try:
        image = image.resize((384, 384))  # TrOCR 입력 크기로 리사이즈
        question_text = []

        for bbox in bboxes:
            _, x_center, y_center, width, height = bbox

            x_min = max(0, int((x_center - width / 2) * 384))
            y_min = max(0, int((y_center - height / 2) * 384))
            x_max = min(384, int((x_center + width / 2) * 384))
            y_max = min(384, int((y_center + height / 2) * 384))

            if x_min >= x_max or y_min >= y_max:
                print(
                    "[WARNING] Skipping invalid bounding box with zero or negative area."
                )
                continue

            cropped_image = image.crop((x_min, y_min, x_max, y_max))
            inputs = processor(images=cropped_image, return_tensors="pt").pixel_values
            outputs = model.generate(inputs)
            text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            question_text.append(text.strip())

        # 텍스트가 없을 경우 기본 메시지 추가
        if not question_text:
            question_text.append("테스트 메시지입니다")

        return question_text

    except Exception as e:
        print(f"[ERROR] Failed to extract text from bounding boxes: {e}")
        return ["테스트 메시지입니다"]  # 오류 발생 시 기본값 반환
